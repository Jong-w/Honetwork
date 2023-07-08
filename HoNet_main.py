import torch
from logger import Logger
from HoNet import HONET, mp_loss
from utils import make_envs, take_action, init_obj
from storage import Storage
import wandb
import torch.optim as optim
import torch.nn.utils as torch_utils

import argparse
import torch
import cv2

parser = argparse.ArgumentParser(description='Honet')
# GENERIC RL/MODEL PARAMETERS
parser.add_argument('--dynamic', type=int, default=0,
                    help='dynamic_neural_network or not')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--env-name', type=str, default='FrostbiteNoFrameskip-v4',  #'MiniGrid-FourRooms-v0' 'MiniGrid-DoorKey-5x5-v0' 'MiniGrid-Empty-16x16-v0'
                    help='gym environment name')
parser.add_argument('--num-workers', type=int, default=64,
                    help='number of parallel environments to run')
parser.add_argument('--num-steps', type=int, default=1000,
                    help='number of steps the agent takes before updating')
parser.add_argument('--max-steps', type=int, default=int(1e8),
                    help='maximum number of training steps in total')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Add cuda')

parser.add_argument('--grad-clip', type=float, default=5.,
                    help='Gradient clipping (recommended).')
parser.add_argument('--entropy-coef', type=float, default=0.2,
                    help='Entropy coefficient to encourage exploration.')
parser.add_argument('--mlp', type=int, default=1,
                    help='toggle to feedforward ML architecture')
parser.add_argument('--whole', type=int, default=1,
                    help='use whole information of the env')
parser.add_argument('--reward-reg', type=int, default=5000,
                    help='reward regulaizer')
parser.add_argument('--env-max-step', type=int, default=5000,
                    help='max step for environment typically same as reward-reg')

parser.add_argument('--grid-size', type=int, default=19,
                    help='setting grid size')

# SPECIFIC FEUDALNET PARAMETERS
parser.add_argument('--time-horizon_manager', type=int, default=20,
                    help=' horizon (c_m)')
parser.add_argument('--time-horizon_supervisor', type=int, default=10,
                    help=' horizon (c_s)')
parser.add_argument('--hidden-dim-manager', type=int, default=256,
                    help='Hidden dim (d)')
parser.add_argument('--hidden-dim-supervisor', type=int, default=128,
                    help='Hidden dim for supervisor (k)')
parser.add_argument('--hidden-dim-worker', type=int, default=64,
                    help='Hidden dim for worker (k)')
parser.add_argument('--gamma-5', type=float, default=0.99,
                    help="discount factor worker")
parser.add_argument('--gamma-4', type=float, default=0.95,
                    help="discount factor supervisor")
parser.add_argument('--gamma-3', type=float, default=0.9,
                    help="discount factor manager")
parser.add_argument('--gamma-2', type=float, default=0.85,
                    help="discount factor worker")
parser.add_argument('--gamma-1', type=float, default=0.8,
                    help="discount factor supervisor")
parser.add_argument('--alpha', type=float, default=0.2,
                    help='Intrinsic reward coefficient in [0, 1]')
parser.add_argument('--eps', type=float, default=float(2e-1),
                    help='Random Gausian goal for exploration')

parser.add_argument('--dilation_manager', type=int, default=20,
                    help='Dilation parameter for manager LSTM.')
parser.add_argument('--dilation_supervisor', type=int, default=10,
                    help='Dilation parameter for manager LSTM.')

parser.add_argument('--hidden-dim-Hierarchies', type=int, default=[84, 84, 84, 84, 84],
                    help='Hidden dim (d)')
parser.add_argument('--time_horizon_Hierarchies', type=int, default=[1, 5, 10, 15, 20, 25],
                    help=' horizon (c_s)')

# EXPERIMENT RELATED PARAMS
parser.add_argument('--run-name', type=str, default='melting_self',
                    help='run name for the logger.')
parser.add_argument('--seed', type=int, default=0,
                    help='reproducibility seed.')

args = parser.parse_args()

def experiment(args):

    save_steps = list(torch.arange(0, int(args.max_steps),
                                   int(args.max_steps) // 1000).numpy())

    # logger = Logger(args.run_name, args)
    logger = Logger(args.env_name, 'THEFUN_64', args)
    cuda_is_available = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda_is_available else "cpu")
    args.device = device

    torch.manual_seed(args.seed)
    if cuda_is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    envs = make_envs(args.env_name, args.num_workers)
    MPnet = HONET(
        num_workers=args.num_workers,
        input_dim=envs.observation_space.shape,
        hidden_dim_Hierarchies = args.hidden_dim_Hierarchies,
        time_horizon_Hierarchies=args.time_horizon_Hierarchies,
        n_actions=envs.single_action_space.n,
        dynamic=0,
        device=device,
        args=args)


    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # In orther to avoid gradient exploding, we apply gradient clipping.

    optimizer = torch.optim.RMSprop(MPnet.parameters(), lr=args.lr,alpha=0.99, eps=1e-5)

    goals_5, states_total, goals_4, goals_3, goals_2, masks = MPnet.init_obj()

    x = envs.reset()
    step = 0
    while step < args.max_steps:
        # Detaching LSTMs and goals_m
        MPnet.repackage_hidden()
        goals_5 = [g.detach() for g in goals_5]
        goals_4 = [g.detach() for g in goals_4]
        goals_3 = [g.detach() for g in goals_3]
        goals_2 = [g.detach() for g in goals_2]

        storage = Storage(size=args.num_steps,
                          keys=['r_i', 'v_5', 'v_4', 'v_3', 'v_2', 'v_1', 'ret_5', 'ret_4', 'ret_3', 'ret_2', 'ret_1',
                                'logp', 'entropy', 'state_goal_5_cos', 'state_goal_4_cos', 'state_goal_3_cos', 'state_goal_2_cos',
                                'hierarchy_selected' 'mask', 'ret_5', 'ret_4', 'ret_3', 'ret_2', 'ret_1'])



        for _ in range(args.num_steps):
            action_dist, goals_5, states_total, value_5, goals_4, value_4, goals_3, value_3, goals_2, value_2, value_1, hierarchies_selected \
                = MPnet(x, goals_5, states_total, goals_4, goals_3, goals_2, masks[-1], step)

            # Take a step, log the info, get the next state
            action, logp, entropy = take_action(action_dist)
            x, reward, done, info = envs.step(action)

            logger.log_episode(info, step)
            wandb.log({'training reward':reward})

            mask = torch.FloatTensor(1 - done).unsqueeze(-1).to(args.device)
            masks.pop(0)
            masks.append(mask)

            add_ = {'r': torch.FloatTensor(reward).unsqueeze(-1).to(device),
                'r_i': MPnet.intrinsic_reward(states_total, goals_2, masks),
                'logp': logp.unsqueeze(-1),
                'entropy': entropy.unsqueeze(-1),
                'hierarchy_selected': hierarchies_selected,
                'm': mask}


            add_['v_5'] = value_5
            add_['state_goal_5_cos'] = MPnet.state_goal_cosine(states_total, goals_5, masks, 5)
            add_['v_4'] = value_4
            add_['state_goal_4_cos'] = MPnet.state_goal_cosine(states_total, goals_4, masks, 4)
            add_['v_3'] = value_3
            add_['state_goal_3_cos'] = MPnet.state_goal_cosine(states_total, goals_3, masks, 3)
            add_['v_2'] = value_2
            add_['state_goal_2_cos'] = MPnet.state_goal_cosine(states_total, goals_2, masks, 2)
            add_['v_1'] = value_1

            storage.add(add_)

            step += args.num_workers

        with torch.no_grad():
            _, _, _, next_v_5, _, next_v_4, _, next_v_3, _, next_v_2, next_v_1, _ = MPnet(x, goals_5, states_total,
                                                                                           goals_4, goals_3, goals_2, masks[-1], step, save = False)

            next_v_5 = next_v_5.detach()
            next_v_4 = next_v_4.detach()
            next_v_3 = next_v_3.detach()
            next_v_2 = next_v_2.detach()
            next_v_1 = next_v_1.detach()

        optimizer.zero_grad()
        loss, loss_dict = mp_loss(storage, next_v_5, next_v_4, next_v_3, next_v_2, next_v_1, args)
        wandb.log(loss_dict)
        loss.backward()
        torch.nn.utils.clip_grad_value_(MPnet.parameters(), clip_value=args.grad_clip)
        optimizer.step()
        logger.log_scalars(loss_dict, step)

    envs.close()
    torch.save({
        'model': MPnet.state_dict(),
        'args': args,
        'processor_mean': MPnet.preprocessor.rms.mean,
        'optim': optimizer.state_dict()},
        f'models/{args.env_name}_{args.run_name}_steps={step}.pt')


def main(args):
    run_name = args.run_name
    seed_size_ori = [args.hidden_dim_manager, args.hidden_dim_worker]
    seed_size = [[128, 64], [256, 128], [512, 256]]
    seed = 1

    for seed in range(5):
        wandb.init(project="fourroom1919_0408",
                   config=args.__dict__
                   )
        args.seed = seed
        wandb.run.name = f"{run_name}_runseed={seed}"
        experiment(args)
        wandb.finish()


if __name__ == '__main__':
    main(args)
