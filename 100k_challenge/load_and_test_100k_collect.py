from logger import Logger
import gym
from MDM_no_hd import MDM_no_hd
from MDM import MDM
from a3c import a3c
from feudalnet import FeudalNetwork
from utils import make_envs, take_action, init_obj
from storage import Storage
import wandb
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import torch
import gc
import numpy as np

parser = argparse.ArgumentParser(description='MDM')

parser = argparse.ArgumentParser(description='Feudal Nets')
# GENERIC RL/MODEL PARAMETERS
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--env-name', type=str, default='FrostbiteNoFrameskip-v4',
                    help='gym environment name')
parser.add_argument('--num-workers', type=int, default=1,
                    help='number of parallel environments to run')
parser.add_argument('--num-steps', type=int, default=400,
                    help='number of steps the agent takes before updating')
parser.add_argument('--max-steps', type=int, default=int(1e5),
                    help='maximum number of training steps in total')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Add cuda')
parser.add_argument('--grad-clip', type=float, default=5.,
                    help='Gradient clipping (recommended).')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='Entropy coefficient to encourage exploration.')
parser.add_argument('--mlp', type=int, default=0,
                    help='toggle to feedforward ML architecture')

# SPECIFIC FEUDALNET PARAMETERS
parser.add_argument('--time-horizon', type=int, default=10,
                    help='Manager horizon (c)')
parser.add_argument('--hidden-dim-manager', type=int, default=256,
                    help='Hidden dim (d)')
parser.add_argument('--hidden-dim-worker', type=int, default=16,
                    help='Hidden dim for worker (k)')
parser.add_argument('--gamma-w', type=float, default=0.95,
                    help="discount factor worker")
parser.add_argument('--gamma-m', type=float, default=0.99,
                    help="discount factor manager")
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Intrinsic reward coefficient in [0, 1]')
parser.add_argument('--eps', type=float, default=int(1e-5),
                    help='Random Gausian goal for exploration')
parser.add_argument('--dilation', type=int, default=10,
                    help='Dilation parameter for manager LSTM.')

# EXPERIMENT RELATED PARAMS
parser.add_argument('--run-name', type=str, default='',
                    help='run name for the logger.')
parser.add_argument('--seed', type=int, default=0,
                    help='reproducibility seed.')

args = parser.parse_args()

def experiment(args):

    # logger = Logger(args.run_name, args)
    logger = Logger(args.env_name, 'MDM_64', args)
    cuda_is_available = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda_is_available else "cpu")
    args.device = device

    if cuda_is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    envs = make_envs(args.env_name, args.num_workers)

    if args.model_name == 'FuN':
        model = FeudalNetwork(
            num_workers=args.num_workers,
            input_dim=envs.observation_space.shape,
            hidden_dim_manager=args.hidden_dim_manager,
            hidden_dim_worker=args.hidden_dim_worker,
            n_actions=envs.single_action_space.n,
            time_horizon=args.time_horizon,
            dilation=args.dilation,
            device=device,
            mlp=args.mlp,
            args=args)
    if args.model_name == 'a3c':
        model = a3c(
            num_workers=args.num_workers,
            input_dim=envs.observation_space.shape,
            hidden_dim_manager=args.hidden_dim_manager,
            hidden_dim_worker=args.hidden_dim_worker,
            n_actions=envs.single_action_space.n,
            time_horizon=args.time_horizon,
            dilation=args.dilation,
            device=device,
            mlp=args.mlp,
            args=args)

    if args.model_name == 'FuN':
        path = '100k_challenge/models_new_testing_fun/' + args.env_name + "_" + args.model_name + "_steps=102400.pt"
    if args.model_name == 'a3c':
        path = '100k_challenge/models/' + args.env_name + "_" + args.model_name + "_steps=102400.pt"
    model.load_state_dict(torch.load(path)['model'])
    model.eval()

    # In orther to avoid gradient exploding, we apply gradient clipping.
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-5)

    goals, states, masks = model.init_obj()

    x = envs.reset()
    step = 0
    step_t_ep = 0
    break_flag = False
    while step < args.max_steps:

        # Detaching LSTMs and goals
        model.repackage_hidden()
        goals = [g.detach() for g in goals]
        storage = Storage(size=args.num_steps,
                          keys=['r', 'r_i', 'v_w', 'v_m', 'logp', 'entropy',
                                's_goal_cos', 'mask', 'ret_w', 'ret_m',
                                'adv_m', 'adv_w'])

        for _ in range(args.num_steps):
            action_dist, goals, states, value_m, value_w \
                = model(x, goals, states, masks[-1])

            # Take a step, log the info, get the next state
            action, logp, entropy = take_action(action_dist)
            x, reward, done, info = envs.step(action)

            #            infos =[ ]
            #           for i in range(len(done)):
            #                # empty dict
            #                info_temp = {}

            #                for dict_name, dict_array in info.items():
            # add dict_name and dict_array[i] to info_temp
            #                    info_temp[dict_name] = dict_array[i]

            #                infos.append(info_temp)

            # logger.log_episode(info, step)

            mask = torch.FloatTensor(1 - done).unsqueeze(-1).to(args.device)
            masks.pop(0)
            masks.append(mask)

            storage.add({
                'r': torch.FloatTensor(reward).unsqueeze(-1).to(device),
                'r_i': model.intrinsic_reward(states, goals, masks),
                'v_w': value_w,
                'v_m': value_m,
                'logp': logp.unsqueeze(-1),
                'entropy': entropy.unsqueeze(-1),
                's_goal_cos': model.state_goal_cosine(states, goals, masks),
                'm': mask
            })
            #            for _i in range(len(done)):
            #                if done[_i] or truncated[_i]:
            #                    wandb.log(
            #                    {"training/episode/reward": infos[_i]['final_info']['returns/episodic_reward'],
            #                     "training/episode/length": infos[_i]['final_info']['returns/episodic_length'],
            #                     "training/episode/reward_sign": int(infos[_i]['final_info']['returns/episodic_reward']!=-1000)
            #                     },step=step)
            for _i in range(len(done)):
                if done[_i]:
                    wandb.log(
                        {"training/episode/reward": info[_i]['returns/episodic_reward'],
                         "training/episode/length": info[_i]['returns/episodic_length']
                         }, step=step)
                    break_flag = True
            step += args.num_workers

            if break_flag:
                break

        #with torch.no_grad():
        #    _, _, _, next_v_5, _, next_v_4, _, next_v_3, _, next_v_2, next_v_1, _, _ = model(x, goals_5, states_total, goals_4, goals_3, goals_2, masks[-1], step, train_eps=0, save = False)


        #    next_v_5 = next_v_5.detach()
        #    next_v_4 = next_v_4.detach()
        #    next_v_3 = next_v_3.detach()
        #    next_v_2 = next_v_2.detach()
        #    next_v_1 = next_v_1.detach()

        #optimizer.zero_grad()
        #loss, loss_dict = mp_loss(storage, next_v_5, next_v_4, next_v_3, next_v_2, next_v_1, args)
        #wandb.log(loss_dict)
        #loss.backward()
        #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=args.grad_clip)
        #optimizer.step()
        #logger.log_scalars(loss_dict, step)
        if break_flag:
            break

    envs.close()
    #torch.save({
    #    'model': model.state_dict(),
    #    'args': args,
    #    'processor_mean': model.preprocessor.rms.mean,
    #    'optim': optimizer.state_dict()},
    #    f'models/{args.env_name}_{args.run_name}_steps={step}.pt')


def main(args):
    all_envs = gym.envs.registry.all()
    noframeskip_v4_no_ram_envs = [env.id for env in all_envs if
                                  ((env.id.endswith('NoFrameskip-v4')) and ('-ram' not in env.id) and ('Defender' not in env.id))]

    run_name = args.run_name

    seeds_ = np.random.randint(-1000, 1000, 100)

    runs = wandb.Api().runs("MDM_100k_collect_test")
    existing_names = [run.name for run in runs]

    #for seed in range(len(noframeskip_v4_no_ram_envs)):
    for i in ['a3c', 'FuN']:
        for seed in range(len(noframeskip_v4_no_ram_envs)):
            for iters in range(len(seeds_)):
                env_name_ = noframeskip_v4_no_ram_envs[seed]

                args.model_name = i

                run_name = f"{env_name_[:-14]}_{args.model_name}_iter{iters}"
                # load wandb runs from the project using wandb API


                if run_name in existing_names:
                    print("Project with the same name already exists.")
                    # Handle the case where the project name already exists
                else:
                    # Continue with the rest of the code
                    # proceed to the next step

                    wandb.init(project="MDM_100k_collect_test",
                            config=args.__dict__
                            )
                    args.seed = seeds_[iters]
                    args.env_name = env_name_
                    wandb.run.name = f"{env_name_[:-14]}_{args.model_name}_iter{iters}"

                    experiment(args)
                    wandb.finish()


if __name__ == '__main__':
    main(args)
