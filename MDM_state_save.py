from logger import Logger
from MDM import MDM, mp_loss
from utils import make_envs, take_action, init_obj
from storage import Storage
import wandb
import numpy as np
from PIL import Image
import pickle

import argparse
import torch
import gc

parser = argparse.ArgumentParser(description='MDM')

# EXPERIMENT RELATED PARAMS
parser.add_argument('--run-name', type=str, default='MDM_25',
                    help='run name for the logger.')
parser.add_argument('--seed', type=int, default=0,
                    help='reproducibility seed.')

# GENERIC RL/MODEL PARAMETERS
parser.add_argument('--dynamic', type=int, default=0,
                    help='dynamic_neural_network or not')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--env-name', type=str, default='FrostbiteNoFrameskip-v4',
                    help='gym environment name')
parser.add_argument('--num-workers', type=int, default=2,
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


# SPECIFIC FEUDALNET PARAMETERS
parser.add_argument('--gamma-5', type=float, default=0.999,
                    help="discount factor worker")
parser.add_argument('--gamma-4', type=float, default=0.999,
                    help="discount factor supervisor")
parser.add_argument('--gamma-3', type=float, default=0.999,
                    help="discount factor manager")
parser.add_argument('--gamma-2', type=float, default=0.999,
                    help="discount factor worker")
parser.add_argument('--gamma-1', type=float, default=0.99,
                    help="discount factor supervisor")
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Intrinsic reward coefficient in [0, 1]')
parser.add_argument('--eps', type=float, default=float(1e-7),
                    help='Random Gausian goal for exploration')
parser.add_argument('--hidden-dim-Hierarchies', type=int, default=[16, 256, 256, 256, 256],
                    help='Hidden dim (d)')
parser.add_argument('--time_horizon_Hierarchies', type=int, default=[1, 10, 15, 20, 25],
                    help=' horizon (c_s)')

parser.add_argument('--lambda-policy-im', type=float, default=0.1)
parser.add_argument('--hierarchy-eps',type=float, default=1e-10)

args = parser.parse_args()

class MaxTracker:
    def __init__(self):
        #intrinsic reward가 이전에 비해 크게 뛴 곳이 좋은 곳이지 않을까?
        self.max_total_intrinsic = -1000
        self.max_state = 0

    def update(self, total_intrinsic, x):
        if self.max_total_intrinsic < total_intrinsic:
            self.max_total_intrinsic = total_intrinsic
            self.max_state = x

    def reset(self):
        self.max_total_intrinsic = -1000
        self.max_state = 0

    def show_result(self):
        return self.max_state

def experiment(args):

    # logger = Logger(args.run_name, args)
    logger = Logger(args.env_name, 'MDM_64', args)
    cuda_is_available = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda_is_available else "cpu")
    args.device = device

    torch.manual_seed(args.seed)
    if cuda_is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    envs = make_envs(args.env_name, args.num_workers)
    tracker = MaxTracker()
    MDMS = MDM(
        num_workers=args.num_workers,
        input_dim=envs.observation_space.shape,
        hidden_dim_Hierarchies = args.hidden_dim_Hierarchies,
        time_horizon_Hierarchies=args.time_horizon_Hierarchies,
        n_actions=envs.single_action_space.n,
        dynamic=0,
        device=device,
        args=args)

    path = 'FrostbiteNoFrameskip-v4_MDM_steps=100006400 (1).pt'
    MDMS.load_state_dict(torch.load(path)['model'], strict=False)
    #MDMS.eval()

    # In orther to avoid gradient exploding, we apply gradient clipping.
    #optimizer = torch.optim.RMSprop(MDMS.parameters(), lr=args.lr, alpha=0.99, eps=1e-5)

    goals_5, states_total, goals_4, goals_3, goals_2, masks = MDMS.init_obj()

    states = []
    actions = []
    steps = []
    total_intrinsics = []

    x = envs.reset()
    step = 0
    train_eps = float(args.hierarchy_eps)
    while step < args.max_steps:
        # Detaching LSTMs and goals_m
        MDMS.repackage_hidden()
        goals_5 = [g.detach() for g in goals_5]
        goals_4 = [g.detach() for g in goals_4]
        goals_3 = [g.detach() for g in goals_3]
        goals_2 = [g.detach() for g in goals_2]

        storage = Storage(size=args.num_steps,
                          keys=['r_i', 'v_5', 'v_4', 'v_3', 'v_2', 'v_1', 'ret_5', 'ret_4', 'ret_3', 'ret_2', 'ret_1',
                                'logp', 'entropy', 'state_goal_5_cos', 'state_goal_4_cos', 'state_goal_3_cos', 'state_goal_2_cos',
                                'hierarchy_selected' 'mask'])

        torch.manual_seed(step)
        epi_step = 0
        tracker.reset()
        for _ in range(args.num_steps):
            epi_step += 1

            action_dist, goals_5, states_total, value_5, goals_4, value_4, goals_3, value_3, goals_2, value_2, value_1, hierarchies_selected, train_eps \
                = MDMS(x, goals_5, states_total, goals_4, goals_3, goals_2, masks[-1], step, train_eps)
            hierarchies_selected = hierarchies_selected.to('cpu')

            # Take a step, log the info, get the next state
            action, logp, entropy = take_action(action_dist.to(args.device))
            x, reward, done, info = envs.step(action)

            logger.log_episode(info, step)

            mask = torch.FloatTensor(1 - done).unsqueeze(-1).to(args.device)
            masks.pop(0)
            masks.append(mask)
            reward_tensor = torch.FloatTensor(reward).unsqueeze(-1).to('cpu')
            Intrinsic_reward_tensor = MDMS.intrinsic_reward(states_total, goals_2, masks).to('cpu')

            state_goal_5_cos = MDMS.state_goal_cosine(states_total, goals_5, masks, 5).to('cpu')
            state_goal_4_cos = MDMS.state_goal_cosine(states_total, goals_4, masks, 4).to('cpu')
            state_goal_3_cos = MDMS.state_goal_cosine(states_total, goals_3, masks, 3).to('cpu')
            state_goal_2_cos = MDMS.state_goal_cosine(states_total, goals_2, masks, 2).to('cpu')

            total_intrinsic_ = state_goal_5_cos + state_goal_4_cos + state_goal_3_cos + state_goal_2_cos
            total_intrinsic = state_goal_2_cos


            worker_num = 3
            #가장 작은 total_intrinsic reward를 알아내고 그에 따라 state를 저장하는 코드
            #tracker.update(total_intrinsic[worker_num], x[worker_num])

            add_ = {'r': torch.FloatTensor(reward).unsqueeze(-1).to('cpu'),
                'r_i': MDMS.intrinsic_reward(states_total, goals_2, masks).to('cpu'),
                'logp': logp.unsqueeze(-1).to('cpu'),
                'entropy': entropy.unsqueeze(-1).to('cpu'),
                'hierarchy_selected': hierarchies_selected.to('cpu'),
                'hierarchy_drop_reward':(MDMS.hierarchy_drop_reward(reward_tensor + (total_intrinsic * args.lambda_policy_im), hierarchies_selected)).to('cpu'),
                'm': mask.to('cpu'),
                'v_5': value_5.to('cpu'),
                'v_4': value_4.to('cpu'),
                'v_3': value_3.to('cpu'),
                'v_2': value_2.to('cpu'),
                'v_1': value_1.to('cpu'),
                'state_goal_5_cos' : state_goal_5_cos,
                'state_goal_4_cos' : state_goal_4_cos,
                'state_goal_3_cos': state_goal_3_cos,
                'state_goal_2_cos': state_goal_2_cos}

            #for _i in range(len(done)):
            '''if (done[worker_num]) and (reward[worker_num] != 0):
                img_ = tracker.show_result()
                img_ = Image.fromarray((img_ * 255).astype(np.uint8))
                img_.save(f'./result_images/result_max_state_{epi_step}_{reward[0]}.png', 'png')'''

            states.append(x)
            actions.append(action)
            steps.append(step)
            total_intrinsics.append(total_intrinsic)

            data = {'states': states, 'actions': actions, 'steps': steps, 'intrinsic_reward': total_intrinsics}

            storage.add(add_)

            step += args.num_workers

        #with torch.no_grad():
        #    _, _, _, next_v_5, _, next_v_4, _, next_v_3, _, next_v_2, next_v_1, _, _ = MDMS(x, goals_5, states_total, goals_4, goals_3, goals_2, masks[-1], step, train_eps=0, save = False)


            #next_v_5 = next_v_5.detach()
            #next_v_4 = next_v_4.detach()
            #next_v_3 = next_v_3.detach()
            #next_v_2 = next_v_2.detach()
            #next_v_1 = next_v_1.detach()

        #optimizer.zero_grad()
        #loss, loss_dict = mp_loss(storage, next_v_5, next_v_4, next_v_3, next_v_2, next_v_1, args)
        #wandb.log(loss_dict)
        #loss.backward()
        #torch.nn.utils.clip_grad_value_(MDMS.parameters(), clip_value=args.grad_clip)
        #optimizer.step()
        #logger.log_scalars(loss_dict, step)
    with open('state_action_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    envs.close()
    #torch.save({
    #    'model': MDMS.state_dict(),
    #    'args': args,
    #    'processor_mean': MDMS.preprocessor.rms.mean,
    #    'optim': optimizer.state_dict()},
    #    f'models_new_testing/{args.env_name}_{args.run_name}_steps={step}.pt')


def main(args):
    import gym
    #all_envs = gym.envs.registry.all()
    #noframeskip_v4_no_ram_envs = [env.id for env in all_envs if
    #                              ((env.id.endswith('NoFrameskip-v4')) and ('-ram' not in env.id) and ('Defender' not in env.id))]

    run_name = args.run_name
    #for seed in range(len(noframeskip_v4_no_ram_envs)):
    #for seed in range(len(noframeskip_v4_no_ram_envs)):
        #env_name_ = noframeskip_v4_no_ram_envs[seed]
    wandb.init(project="just_testing",
                config=args.__dict__)
        #args.seed = seed
        #args.env_name = env_name_
        #wandb.run.name = f"{run_name}_{env_name_[:-14]}"

    wandb.run.name = f"{run_name}"
    experiment(args)
    wandb.finish()


if __name__ == '__main__':
    main(args)
