import os 
import argparse
import numpy as np
import datetime
import copy
from envs import EnvWithGoal
from envs.create_maze_env import create_maze_env
from hiro.hiro_utils import Subgoal 
from hiro.utils import Logger, _is_update, record_experience_to_csv, listdirs
from hiro.models import HiroAgent, TD3Agent
import torch
import wandb
import warnings

# warnings.filterwarnings(action='ignore')

def run_evaluation(args, env, agent):
    agent.load(args.load_episode)

    rewards, success_rate = agent.evaluate_policy(env, args.eval_episodes, args.render, args.save_video, args.sleep)
    
    print('mean:{mean:.2f}, \
            std:{std:.2f}, \
            median:{median:.2f}, \
            success:{success:.2f}'.format(
                mean=np.mean(rewards), 
                std=np.std(rewards), 
                median=np.median(rewards), 
                success=success_rate))

class Trainer():
    def __init__(self, args, env, agent, experiment_name):
        self.args = args
        self.env = env
        self.agent = agent 
        log_path = os.path.join(args.log_path, experiment_name)
        self.logger = Logger(log_path=log_path)

    def train(self):
        global_step = 0

        for e in np.arange(self.args.num_episode)+1:
            obs = self.env.reset()
            fg = torch.Tensor(obs['desired_goal']).to(device)
            s = torch.Tensor(obs['observation']).to(device)
            done = False

            step = 0
            episode_reward = 0

            actor_loss_low = 0
            critic_loss_low = 0
            actor_loss_high = {2: 0, 3: 0, 4: 0, 5: 0}
            critic_loss_high = {2: 0, 3: 0, 4: 0, 5: 0}
            td_error_low = 0
            td_error_high = {2: 0, 3: 0, 4: 0, 5: 0}

            self.agent.set_final_goal(fg)

            while not done:
                # Take action
                a, r, n_s, done = self.agent.step(s, self.env, step, global_step, explore=True)
                # Append
                self.agent.append(step, s, a, n_s, r, done)
                # Train
                losses, td_errors = self.agent.train(global_step)
                # Log
                self.log(global_step, [losses, td_errors])
                # Updates
                s = n_s
                episode_reward += r
                step += 1
                global_step += 1
                self.agent.end_step()
                
                # wandb log
                _to_log = {'step/training/reward': r, 'step/training/episode': e, 'step/training/step': step}
                # add to _to_log for key in losses.keys(), td_errors.keys()
                for k, v in losses.items():
                    _to_log['step/training/%s'%(k)] = v.item()
                    if 'actor_loss' in k:
                        if 'high' in k:
                            actor_loss_high[int(k[-1])] += v.item()
                            if int(k[-1]) == 2:
                                # remove the ['step/training/%s'%(k)] and rename it
                                _to_log.pop('step/training/%s'%(k), None)
                                _to_log['step/training/actor_loss_high'] = v.item()
                        else:
                            actor_loss_low += v.item()
                            # remove the ['step/training/%s'%(k)] and rename it
                            _to_log.pop('step/training/%s'%(k), None)
                            _to_log['step/training/actor_loss_low'] = v.item()
                    if 'critic_loss' in k:
                        if 'high' in k:
                            critic_loss_high[int(k[-1])] += v.item()
                            if int(k[-1]) == 2:
                                # remove the ['step/training/%s'%(k)] and rename it
                                _to_log.pop('step/training/%s'%(k), None)
                                _to_log['step/training/critic_loss_high'] = v.item()
                        else:
                            critic_loss_low += v.item()
                            # remove the ['step/training/%s'%(k)] and rename it
                            _to_log.pop('step/training/%s'%(k), None)
                            _to_log['step/training/critic_loss_low'] = v.item()
                
                for k, v in td_errors.items():
                    _to_log['step/training/%s'%(k)] = v.item()
                    if 'low' in k:
                        td_error_low += v.item()
                        _to_log.pop('step/training/%s'%(k), None)
                        _to_log['step/training/td_error_low'] = td_error_low
                        
                    if 'high' in k:
                        td_error_high[int(k[-1])] += v.item()
                        if int(k[-1]) == 2:
                            # remove the ['step/training/%s'%(k)] and rename it
                            _to_log.pop('step/training/%s'%(k), None)
                            _to_log['step/training/td_error_high'] = v.item()


                wandb.log(_to_log, step=global_step) 
                
                # print detaield results for debugging if losses and td_errors are not empty dict
                # check if the dictionary is empty
                if len(losses)==0 and len(td_errors)==0:
                    # for every N steps, print the episode number, step number, and reward
                    if ((step % 100) == 0) or done:
                        print('[EP:{episode:05d}], step:{step:05d}, reward:{reward:.2f}'.format(
                            episode=e,
                            step=step,
                            reward=r
                        ))
            self.agent.end_episode(e, self.logger)
            self.logger.write('reward/Reward', episode_reward, e)
            wandb.log({
                "episode/training/actor_loss_low": actor_loss_low,
                "episode/training/critic_loss_low": critic_loss_low,
                "episode/training/actor_loss_high": actor_loss_high[2],
                "episode/training/critic_loss_high": critic_loss_high[2],
                "episode/training/actor_loss_high3": actor_loss_high[3],
                "episode/training/critic_loss_high3": critic_loss_high[3],
                "episode/training/actor_loss_high4": actor_loss_high[4],
                "episode/training/critic_loss_high4": critic_loss_high[4],
                "episode/training/actor_loss_high5": actor_loss_high[5],
                "episode/training/critic_loss_high5": critic_loss_high[5],
                "episode/training/td_error_low": td_error_low,
                "episode/training/td_error_high": td_error_high[2],
                "episode/training/td_error_high3": td_error_high[3],
                "episode/training/td_error_high4": td_error_high[4],
                "episode/training/td_error_high5": td_error_high[5],
                "episode/training/reward": episode_reward,
                "episode/training/episode": e,
                "episode/training/step": step
            }, step=global_step)
            self.evaluate(e)

    def log(self, global_step, data):
        losses, td_errors = data[0], data[1]

        # Logs
        if global_step >= self.args.start_training_steps and _is_update(global_step, args.writer_freq):
            for k, v in losses.items():
                self.logger.write('loss/%s'%(k), v, global_step)
            
            for k, v in td_errors.items():
                self.logger.write('td_error/%s'%(k), v, global_step)
    
    def evaluate(self, e):
        # Print
        if _is_update(e, args.print_freq):
            agent = copy.deepcopy(self.agent)
            # agent = self.agent
            rewards, success_rate = agent.evaluate_policy(self.env)
            #rewards, success_rate = self.agent.evaluate_policy(self.env)
            self.logger.write('Success Rate', success_rate, e)

            print('episode:{episode:05d}, mean:{mean:.2f}, std:{std:.2f}, median:{median:.2f}, success:{success:.2f}'.format(
                    episode=e,
                    mean=np.mean(rewards),
                    std=np.std(rewards),
                    median=np.median(rewards),
                    success=success_rate))
            wandb.log({"test/mean_reward": np.mean(rewards),
                        "test/std_reward": np.std(rewards),
                        "test/median_reward": np.median(rewards),
                       "test/success_rate": success_rate
                       }, step=e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Across All
    parser.add_argument('--train', type=int , default = 1)
    parser.add_argument('--eval', type=int, default = 0)
    parser.add_argument('--render', type=int, default = 0)  
    parser.add_argument('--save_video', type=int, default = 0)
    parser.add_argument('--sleep', type=float, default=-1)
    parser.add_argument('--eval_episodes', type=float, default=5, help='Unit = Episode')
    parser.add_argument('--env', default='AntMaze', type=str)
    parser.add_argument('--td3', type=int, default = 0)
    # Training
    parser.add_argument('--num_episode', default=25000, type=int)
    parser.add_argument('--start_training_steps', default=500, type=int, help='Unit = Global Step')
    parser.add_argument('--writer_freq', default=25, type=int, help='Unit = Global Step')
    # Training (Model Saving)
    parser.add_argument('--subgoal_dim', default=15, type=int)
    parser.add_argument('--load_episode', default=-1, type=int)
    parser.add_argument('--model_save_freq', default=2000, type=int, help='Unit = Episodes')
    parser.add_argument('--print_freq', default=100, type=int, help='Unit = Episode')
    parser.add_argument('--exp_name', default=None, type=str)
    # Model
    parser.add_argument('--model_path', default='model', type=str)
    parser.add_argument('--log_path', default='log', type=str)
    parser.add_argument('--policy_freq_low', default=2, type=int)
    parser.add_argument('--policy_freq_high', default=2, type=int)
    # Replay Buffer
    parser.add_argument('--buffer_size', default=200000, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--buffer_freq', default=10, type=int) #maybe this can be used as dilation(This provides temporal abstraction,since high-level decisions via µhi are made only every c steps)
    parser.add_argument('--train_freq', default=10, type=int)
    parser.add_argument('--reward_scaling', default=0.1, type=float)
    parser.add_argument('--time_horizon', default=[5, 10, 15, 20, 25], type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="MDM_hiro_dk",
               name = "HIRO_MDM",
               config=args.__dict__
               )

    # integer to boolean
    args.train = bool(args.train)
    args.eval = bool(args.eval)
    args.render = bool(args.render)
    args.save_video = bool(args.save_video)


    # Select or Generate a name for this experiment
    if args.exp_name:
        experiment_name = args.exp_name
    else:
        if args.eval:
            # choose most updated experiment for evaluation
            dirs_str = listdirs(args.model_path)
            dirs = np.array(list(map(int, dirs_str)))
            experiment_name = dirs_str[np.argmax(dirs)]
        else:
            experiment_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(experiment_name)

    # Environment and its attributes
    env = EnvWithGoal(create_maze_env(args.env), args.env)
    goal_dim = 2
    state_dim = env.state_dim
    action_dim = env.action_dim
    scale = env.action_space.high * np.ones(action_dim)

    # Spawn an agent
    if args.td3:
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            scale=scale,
            model_save_freq=args.model_save_freq,
            model_path=os.path.join(args.model_path, experiment_name),
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            start_training_steps=args.start_training_steps
            )
    else:
        agent = HiroAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            subgoal_dim=args.subgoal_dim,
            scale_low=scale,
            start_training_steps=args.start_training_steps,
            model_path=os.path.join(args.model_path, experiment_name),
            model_save_freq=args.model_save_freq,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            buffer_freq=args.buffer_freq,
            train_freq=args.train_freq,
            reward_scaling=args.reward_scaling,
            policy_freq_high=args.policy_freq_high,
            policy_freq_low=args.policy_freq_low,
            time_horizon = args.time_horizon
            )

    # Run training or evaluation
    if args.train:
        # Record this experiment with arguments to a CSV file
        record_experience_to_csv(args, experiment_name)
        # Start training
        trainer = Trainer(args, env, agent, experiment_name)
        trainer.train()
    if args.eval:
        run_evaluation(args, env, agent)

    wandb.finish()