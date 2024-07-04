##################################################
# @copyright Kandai Watanabe
# @email kandai.wata@gmail.com
# @institute University of Colorado Boulder
#
# NN Models for HIRO
# (Data-Efficient Hierarchical Reinforcement Learning)
# Parameters can be find in the original paper
import os
import copy
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_tensor
from hiro.hiro_utils import LowReplayBuffer, HighReplayBuffer, ReplayBuffer, Subgoal, Hierarchy5_buffer, Hierarchy4_buffer, Hierarchy3_buffer, Hierarchy2_buffer
from hiro.utils import _is_update
#from dilated_lstm import DilatedLSTM
import random
import gym

class DilatedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, radius=10):
        super().__init__()
        self.radius = radius
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(input_size, hidden_size).to(device)
        self.index = torch.arange(0, radius * hidden_size, radius)
        self.dilation = 0

    def forward(self, state, hidden):
        """At each time step only the corresponding part of the state is updated
        and the output is pooled across the previous c out- puts."""
        d_idx = self.dilation_idx
        hx, cx = hidden
        state = state.type(torch.float32)

        #hx[:, d_idx], cx[:, d_idx] = self.rnn(state, (hx[:, d_idx], cx[:, d_idx]))
        hx[d_idx], cx[d_idx] = self.rnn(state.to(device), (hx[d_idx].to(device), cx[d_idx].to(device)))
        detached_hx = hx[self.masked_idx(d_idx)].detach()
        detached_hx = detached_hx.view(1, self.hidden_size, self.radius-1)
        detached_hx = detached_hx.sum(-1)

        y = (hx[d_idx] + detached_hx) / self.radius
        return y, (hx, cx)

    def masked_idx(self, dilated_idx):
        """Because we do not want to have gradients flowing through all
        parameters but only at the dilation index, this function creates a
        'negated' version of dilated_index, everything EXCEPT these indices."""
        masked_idx = torch.arange(1, self.radius * self.hidden_size + 1)
        masked_idx[dilated_idx] = 0
        masked_idx = masked_idx.nonzero()
        masked_idx = masked_idx - 1
        return masked_idx

    @property
    def dilation_idx(self):
        """Keep track at which dilation we currently we are."""
        dilation_idx = self.dilation + self.index
        self.dilation = (self.dilation + 1) % self.radius
        return dilation_idx


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, scale=None):
        super(TD3Actor, self).__init__()
        if scale is None:
            scale = torch.ones(state_dim)
        else:
            scale = get_tensor(scale)
        self.scale = nn.Parameter(scale.clone().detach().float(), requires_grad=False)

        self.l1 = nn.Linear(state_dim + goal_dim, 300).to(device)
        self.l2 = nn.Linear(300, 300).to(device)
        self.l3 = nn.Linear(300, action_dim).to(device)

    def forward(self, state, goal):
        a = F.relu(self.l1(torch.cat([state, goal], 1).type(torch.float32)))
        a = F.relu(self.l2(a))
        return self.scale * torch.tanh(self.l3(a))

class TD3Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(TD3Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + goal_dim + action_dim, 300).to(device)
        self.l2 = nn.Linear(300, 300).to(device)
        self.l3 = nn.Linear(300, 1).to(device)
        # Q2
        self.l4 = nn.Linear(state_dim + goal_dim + action_dim, 300).to(device)
        self.l5 = nn.Linear(300, 300).to(device)
        self.l6 = nn.Linear(300, 1).to(device)

    def forward(self, state, goal, action):
        sa = torch.cat([state, goal, action], 1).type(torch.float32)

        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q

class TD3Controller(object):
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim,
            scale,
            model_path,
            actor_lr=0.0001,
            critic_lr=0.001,
            expl_noise=0.1,
            policy_noise=0.2,
            noise_clip=0.5,
            gamma=0.99,
            policy_freq=2,
            tau=0.005):
        self.name = 'td3'
        self.scale = scale
        self.model_path = model_path

        # parameters
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_freq = policy_freq
        self.tau = tau

        self.actor = TD3Actor(state_dim, goal_dim, action_dim, scale=scale).to(device)
        self.actor_target = TD3Actor(state_dim, goal_dim, action_dim, scale=scale).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic1 = TD3Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic2 = TD3Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic1_target = TD3Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic2_target = TD3Critic(state_dim, goal_dim, action_dim).to(device)

        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self._initialize_target_networks()

        self._initialized = False
        self.total_it = 0

        self.linear = nn.Linear(15, 15, bias=False).to(device)

    def _initialize_target_networks(self):
        self._update_target_network(self.critic1_target, self.critic1, 1.0)
        self._update_target_network(self.critic2_target, self.critic2, 1.0)
        self._update_target_network(self.actor_target, self.actor, 1.0)
        self._initialized = True

    def _update_target_network(self, target, origin, tau):
        for target_param, origin_param in zip(target.parameters(), origin.parameters()):
            target_param.data.copy_(tau * origin_param.data + (1.0 - tau) * target_param.data)

    def save(self, episode):
        # create episode directory. (e.g. model/2000)
        model_path = os.path.join(self.model_path, str(episode))
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # save file (e.g. model/2000/high_actor.h)
        torch.save(
            self.actor.state_dict(), 
            os.path.join(model_path, self.name+"_actor.h5")
        )
        torch.save(
            self.critic1.state_dict(), 
            os.path.join(model_path, self.name+"_critic1.h5")
        )
        torch.save(
            self.critic2.state_dict(), 
            os.path.join(model_path, self.name+"_critic2.h5")
        )

    def load(self, episode):
        # episode is -1, then read most updated
        if episode<0:
            episode_list = map(int, os.listdir(self.model_path))
            episode = max(episode_list)

        model_path = os.path.join(self.model_path, str(episode)) 

        self.actor.load_state_dict(torch.load(
            os.path.join(model_path, self.name+"_actor.h5"))
        )
        self.critic1.load_state_dict(torch.load(
            os.path.join(model_path, self.name+"_critic1.h5"))
        )
        self.critic2.load_state_dict(torch.load(
            os.path.join(model_path, self.name+"_critic2.h5"))
        )

    def _train(self, states, goals, actions, rewards, n_states, n_goals, not_done, hierarchy_num):
        self.total_it += 1
        with torch.no_grad():
            noise = (
                torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            n_actions = self.actor_target(n_states, n_goals) + noise
            n_actions = torch.min(n_actions,  self.actor.scale)
            n_actions = torch.max(n_actions, -self.actor.scale)

            target_Q1 = self.critic1_target(n_states, n_goals, n_actions)
            target_Q2 = self.critic2_target(n_states, n_goals, n_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q_detached = (rewards + not_done * self.gamma * target_Q).detach()

        current_Q1 = self.critic1(states, goals, actions)
        current_Q2 = self.critic2(states, goals, actions)

        critic1_loss = F.smooth_l1_loss(current_Q1, target_Q_detached)
        critic2_loss = F.smooth_l1_loss(current_Q2, target_Q_detached)
        critic_loss = critic1_loss + critic2_loss

        td_error = (target_Q_detached - current_Q1).mean().cpu().data.numpy()

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            a = self.actor(states, goals)
            Q1 = self.critic1(states, goals, a)
            actor_loss = -Q1.mean() # multiply by neg becuz gradient ascent

            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            self._update_target_network(self.critic1_target, self.critic1, self.tau)
            self._update_target_network(self.critic2_target, self.critic2, self.tau)
            self._update_target_network(self.actor_target, self.actor, self.tau)

            return {'actor_loss_' + hierarchy_num +self.name: actor_loss, 'critic_loss_'+ hierarchy_num + self.name: critic_loss}, \
                    {'td_error_' + hierarchy_num + self.name: td_error}

        return {'critic_loss_'+ hierarchy_num+self.name: critic_loss}, \
                    {'td_error_'+ hierarchy_num+self.name: td_error}

    def train(self, replay_buffer, iterations=1):
        states, goals, actions, n_states, rewards, not_done = replay_buffer.sample()
        return self._train(states, goals, actions, rewards, n_states, goals, not_done)

    def policy(self, state, goal, to_numpy=True):
        #state = get_tensor(state)
        #goal = get_tensor(goal)
        action = self.actor(state, goal)

        #if to_numpy:
        #    return action.cpu().data.numpy().squeeze()

        return action.squeeze()

    def back(self, goal_now, goal_prev, hierarchy_drop, drop =True):
        if drop:
            result = (self.linear(torch.tensor(goal_prev, dtype=torch.float32)).to(device) * hierarchy_drop.to(device)) + goal_now.to(device)
        else:
            result = (self.linear(torch.tensor(goal_prev, dtype=torch.float32)).to(device)) + goal_now.to(device)
        return result

    def policy_with_noise(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)
        action = self.actor(state, goal)

        action = action + self._sample_exploration_noise(action)
        action = torch.min(action,  self.actor.scale)
        action = torch.max(action, -self.actor.scale)

        if to_numpy:
            return action.cpu().data.numpy().squeeze()

        return action.squeeze()

    def _sample_exploration_noise(self, actions):
        mean = torch.zeros(actions.size()).to(device)
        var = torch.ones(actions.size()).to(device)
        #expl_noise = self.expl_noise - (self.expl_noise/1200) * (self.total_it//10000)
        return torch.normal(mean, self.expl_noise*var)



class Hierarchy5(TD3Controller):
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        scale,
        model_path,
        actor_lr=0.0001,
        critic_lr=0.001,
        expl_noise=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005):
        super(Hierarchy5, self).__init__(
            state_dim, goal_dim, action_dim, scale, model_path,
            actor_lr, critic_lr, expl_noise, policy_noise,
            noise_clip, gamma, policy_freq, tau
        )
        self.name = 'high'
        self.action_dim = action_dim

    def off_policy_corrections(self, low_con, batch_size, sgoals, states, actions, candidate_goals=8):
        first_s = [s[0] for s in states] # First x
        last_s = [s[-1] for s in states] # Last x

        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        diff_goal = (torch.stack(last_s, dim=0) -
                     torch.stack(first_s, dim=0))[:, np.newaxis, :self.action_dim]

        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals
        original_goal = torch.tensor(sgoals)[:, np.newaxis, :]
        #random_goals = np.random.normal(loc=diff_goal, scale=.5*self.scale[None, None, :],
        #                                size=(batch_size, candidate_goals, original_goal.shape[-1]))
        scale_tensor = 0.5 * torch.tensor(self.scale[None, None, :]).to(device)
        random_goals = torch.normal(mean=diff_goal, std=scale_tensor.expand(batch_size, candidate_goals, -1))

        self.scale = torch.tensor(self.scale).to(device)
        random_goals = random_goals.clip(-self.scale, self.scale)

        # Shape: (batch_size, 10, subgoal_dim)
        candidates = torch.concatenate([original_goal, diff_goal, random_goals], axis=1)
        #states = np.array(states)[:, :-1, :]
        actions = torch.tensor(actions)
        seq_len = len(states[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = actions[0][0].shape
        obs_dim = states[0][0].shape
        ncands = candidates.shape[1]

        true_actions = actions.reshape((new_batch_sz,) + action_dim)
        observations = states.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = torch.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            subgoal = candidates[:,c]
            candidate = (subgoal + states[:, 0, :self.action_dim])[:, None] - states[:, :, :self.action_dim]
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = low_con.policy(observations, candidate)

        difference = (policy_actions.to(device) - true_actions.to(device))
        difference = torch.where(difference != -torch.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).permute(1, 0, 2, 3)

        logprob = -0.5*torch.sum(torch.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = torch.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]

    def train(self, replay_buffer, low_con):
        if not self._initialized:
            self._initialize_target_networks()

        states, goals, actions, n_states, rewards, not_done, states_arr, actions_arr = replay_buffer.sample()

        actions = self.off_policy_corrections(
            low_con,
            replay_buffer.batch_size,
            actions,
            states_arr,
            actions_arr)

        #actions = get_tensor(actions)
        return self._train(states, goals, actions, rewards, n_states, goals, not_done, "5_")

class Hierarchy4(TD3Controller):
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        scale,
        model_path,
        actor_lr=0.0001,
        critic_lr=0.001,
        expl_noise=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005):
        super(Hierarchy4, self).__init__(
            state_dim, goal_dim, action_dim, scale, model_path,
            actor_lr, critic_lr, expl_noise, policy_noise,
            noise_clip, gamma, policy_freq, tau
        )
        self.name = 'high'
        self.action_dim = action_dim

    def off_policy_corrections(self, low_con, batch_size, sgoals, states, actions, candidate_goals=8):
        first_s = [s[0] for s in states] # First x
        last_s = [s[-1] for s in states] # Last x

        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        diff_goal = (torch.stack(last_s, dim=0) -
                     torch.stack(first_s, dim=0))[:, np.newaxis, :self.action_dim]

        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals
        original_goal = torch.tensor(sgoals)[:, np.newaxis, :]
        #random_goals = np.random.normal(loc=diff_goal, scale=.5*self.scale[None, None, :],
        #                                size=(batch_size, candidate_goals, original_goal.shape[-1]))
        scale_tensor = 0.5 * torch.tensor(self.scale[None, None, :]).to(device)
        random_goals = torch.normal(mean=diff_goal, std=scale_tensor.expand(batch_size, candidate_goals, -1))

        self.scale = torch.tensor(self.scale).to(device)
        random_goals = random_goals.clip(-self.scale, self.scale)

        # Shape: (batch_size, 10, subgoal_dim)
        candidates = torch.concatenate([original_goal, diff_goal, random_goals], axis=1)
        #states = np.array(states)[:, :-1, :]
        actions = torch.tensor(actions)
        seq_len = len(states[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = actions[0][0].shape
        obs_dim = states[0][0].shape
        ncands = candidates.shape[1]

        true_actions = actions.reshape((new_batch_sz,) + action_dim)
        observations = states.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = torch.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            subgoal = candidates[:,c]
            candidate = (subgoal + states[:, 0, :self.action_dim])[:, None] - states[:, :, :self.action_dim]
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = low_con.policy(observations, candidate)

        difference = (policy_actions.to(device) - true_actions.to(device))
        difference = torch.where(difference != -torch.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).permute(1, 0, 2, 3)

        logprob = -0.5*torch.sum(torch.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = torch.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]

    def train(self, replay_buffer, low_con):
        if not self._initialized:
            self._initialize_target_networks()

        states, goals, actions, n_states, rewards, not_done, states_arr, actions_arr = replay_buffer.sample()

        actions = self.off_policy_corrections(
            low_con,
            replay_buffer.batch_size,
            actions,
            states_arr,
            actions_arr)

        #actions = get_tensor(actions)
        return self._train(states, goals, actions, rewards, n_states, goals, not_done, "4_")

class Hierarchy3(TD3Controller):
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        scale,
        model_path,
        actor_lr=0.0001,
        critic_lr=0.001,
        expl_noise=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005):
        super(Hierarchy3, self).__init__(
            state_dim, goal_dim, action_dim, scale, model_path,
            actor_lr, critic_lr, expl_noise, policy_noise,
            noise_clip, gamma, policy_freq, tau
        )
        self.name = 'high'
        self.action_dim = action_dim

    def off_policy_corrections(self, low_con, batch_size, sgoals, states, actions, candidate_goals=8):
        first_s = [s[0] for s in states] # First x
        last_s = [s[-1] for s in states] # Last x

        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        diff_goal = (torch.stack(last_s, dim=0) -
                     torch.stack(first_s, dim=0))[:, np.newaxis, :self.action_dim]

        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals
        original_goal = torch.tensor(sgoals)[:, np.newaxis, :]
        #random_goals = np.random.normal(loc=diff_goal, scale=.5*self.scale[None, None, :],
        #                                size=(batch_size, candidate_goals, original_goal.shape[-1]))
        scale_tensor = 0.5 * torch.tensor(self.scale[None, None, :]).to(device)
        random_goals = torch.normal(mean=diff_goal, std=scale_tensor.expand(batch_size, candidate_goals, -1))

        self.scale = torch.tensor(self.scale).to(device)
        random_goals = random_goals.clip(-self.scale, self.scale)

        # Shape: (batch_size, 10, subgoal_dim)
        candidates = torch.concatenate([original_goal, diff_goal, random_goals], axis=1)
        #states = np.array(states)[:, :-1, :]
        actions = torch.tensor(actions)
        seq_len = len(states[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = actions[0][0].shape
        obs_dim = states[0][0].shape
        ncands = candidates.shape[1]

        true_actions = actions.reshape((new_batch_sz,) + action_dim)
        observations = states.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = torch.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            subgoal = candidates[:,c]
            candidate = (subgoal + states[:, 0, :self.action_dim])[:, None] - states[:, :, :self.action_dim]
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = low_con.policy(observations, candidate)

        difference = (policy_actions.to(device) - true_actions.to(device))
        difference = torch.where(difference != -torch.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).permute(1, 0, 2, 3)

        logprob = -0.5*torch.sum(torch.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = torch.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]

    def train(self, replay_buffer, low_con):
        if not self._initialized:
            self._initialize_target_networks()

        states, goals, actions, n_states, rewards, not_done, states_arr, actions_arr = replay_buffer.sample()

        actions = self.off_policy_corrections(
            low_con,
            replay_buffer.batch_size,
            actions,
            states_arr,
            actions_arr)

        #actions = get_tensor(actions)
        return self._train(states, goals, actions, rewards, n_states, goals, not_done, "3_")


class Hierarchy2(TD3Controller):
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        scale,
        model_path,
        actor_lr=0.0001,
        critic_lr=0.001,
        expl_noise=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005):
        super(Hierarchy2, self).__init__(
            state_dim, goal_dim, action_dim, scale, model_path,
            actor_lr, critic_lr, expl_noise, policy_noise,
            noise_clip, gamma, policy_freq, tau
        )
        self.name = 'high'
        self.action_dim = action_dim

    def off_policy_corrections(self, low_con, batch_size, sgoals, states, actions, candidate_goals=8):
        first_s = [s[0] for s in states] # First x
        last_s = [s[-1] for s in states] # Last x

        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        diff_goal = (torch.stack(last_s, dim=0) -
                     torch.stack(first_s, dim=0))[:, np.newaxis, :self.action_dim]

        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals
        original_goal = torch.tensor(sgoals)[:, np.newaxis, :]
        #random_goals = np.random.normal(loc=diff_goal, scale=.5*self.scale[None, None, :],
        #                                size=(batch_size, candidate_goals, original_goal.shape[-1]))
        scale_tensor = 0.5 * torch.tensor(self.scale[None, None, :]).to(device)
        random_goals = torch.normal(mean=diff_goal, std=scale_tensor.expand(batch_size, candidate_goals, -1))

        self.scale = torch.tensor(self.scale).to(device)
        random_goals = random_goals.clip(-self.scale, self.scale)

        # Shape: (batch_size, 10, subgoal_dim)
        candidates = torch.concatenate([original_goal, diff_goal, random_goals], axis=1)
        #states = np.array(states)[:, :-1, :]
        actions = torch.tensor(actions)
        seq_len = len(states[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = actions[0][0].shape
        obs_dim = states[0][0].shape
        ncands = candidates.shape[1]

        true_actions = actions.reshape((new_batch_sz,) + action_dim)
        observations = states.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = torch.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            subgoal = candidates[:,c]
            candidate = (subgoal + states[:, 0, :self.action_dim])[:, None] - states[:, :, :self.action_dim]
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = low_con.policy(observations, candidate)

        difference = (policy_actions.to(device) - true_actions.to(device))
        difference = torch.where(difference != -torch.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).permute(1, 0, 2, 3)

        logprob = -0.5*torch.sum(torch.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = torch.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]

    def train(self, replay_buffer, low_con):
        if not self._initialized:
            self._initialize_target_networks()

        states, goals, actions, n_states, rewards, not_done, states_arr, actions_arr = replay_buffer.sample()

        actions = self.off_policy_corrections(
            low_con,
            replay_buffer.batch_size,
            actions,
            states_arr,
            actions_arr)

        #actions = get_tensor(actions)
        return self._train(states, goals, actions, rewards, n_states, goals, not_done, "2_")

class LowerController(TD3Controller):
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        scale,
        model_path,
        actor_lr=0.0001,
        critic_lr=0.001,
        expl_noise=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005):
        super(LowerController, self).__init__(
            state_dim, goal_dim, action_dim, scale, model_path,
            actor_lr, critic_lr, expl_noise, policy_noise,
            noise_clip, gamma, policy_freq, tau
        )
        self.name = 'low'

    def train(self, replay_buffer):
        if not self._initialized:
            self._initialize_target_networks()

        states, sgoals, actions, n_states, n_sgoals, rewards, not_done = replay_buffer.sample()

        return self._train(states, sgoals, actions, rewards, n_states, n_sgoals, not_done, "1_")

class Policy_Network(nn.Module):
    def __init__(self, d, time_horizon, num_workers):
        super().__init__()
        self.Mrnn = DilatedLSTM(76, 3, time_horizon)   # 76을 다른 변수들의 연산으로 만들어 놓기(지금은 그냥 디버깅해 값을 확인 후 채운 것)
        self.num_workers = num_workers
    def forward(self, z, goal_5_norm, goal_4_norm, goal_3_norm, hierarchies_selected, time_horizon, hidden, mask, step):
        goal_x_info = torch.cat([goal_5_norm.detach(),goal_4_norm.detach(),goal_3_norm.detach(), z.detach()])
        hidden = (mask * hidden[0], mask * hidden[1])
        hidden = (hidden[0].squeeze(), hidden[1].squeeze())

        policy_network_result, hidden = self.Mrnn(goal_x_info, hidden)
        policy_network_result = (policy_network_result - policy_network_result.detach().min(1, keepdim=True)[0]) / \
                                (policy_network_result.detach().max(1, keepdim=True)[0] - policy_network_result.detach().min(1, keepdim=True)[0])
        #policy_network_result = policy_network_result.round()
        return policy_network_result.type(torch.int), hidden

    def hierarchy_drop_reward(self, reward, hierarchy_selected):
        #drop_reward = (reward - (hierarchy_selected.sum(dim=1).reshape(self.num_workers, 1))) / (reward+1)
        drop_reward = reward
        return drop_reward

def init_hidden(n_workers, h_dim, device, grad=False):
    return (torch.zeros(n_workers, h_dim, requires_grad=grad).to(device),
            torch.zeros(n_workers, h_dim, requires_grad=grad).to(device))

class Agent():
    def __init__(self):
        pass

    def set_final_goal(self, fg):
        self.fg = fg

    def step(self, s, env, step, global_step=0, explore=False):
        raise NotImplementedError

    def append(self, step, s, a, n_s, r, d):
        raise NotImplementedError

    def train(self, global_step):
        raise NotImplementedError

    def end_step(self):
        raise NotImplementedError

    def end_episode(self, episode, logger=None):
        raise NotImplementedError
    
    def evaluate_policy(self, env, eval_episodes=10, render=False, save_video=False, sleep=-1):
        if save_video:
            from OpenGL import GL
            env = gym.wrappers.Monitor(env, directory='video',
                                    write_upon_reset=True, force=True, resume=True, mode='evaluation')
            render = False

        success = 0
        rewards = []
        env.evaluate = True
        for e in range(eval_episodes):
            obs = env.reset()
            fg = obs['desired_goal']
            s = obs['observation']
            done = False
            reward_episode_sum = 0
            step = 0
            
            self.set_final_goal(fg)

            while not done:
                if render:
                    env.render()
                if sleep>0:
                    time.sleep(sleep)

                a, r, n_s, done = self.step(s, env, step)
                reward_episode_sum += r
                
                s = n_s
                step += 1
                self.end_step()
            else:
                error = torch.sqrt(torch.sum(torch.square(torch.tensor(fg)-s[:2])))
                print('Goal, Curr: (%02.2f, %02.2f, %02.2f, %02.2f)     Error:%.2f'%(fg[0], fg[1], s[0], s[1], error))
                rewards.append(reward_episode_sum)
                success += 1 if error <=5 else 0
                self.end_episode(e)

        env.evaluate = False
        return np.array(rewards), success/eval_episodes

class TD3Agent(Agent):
    def __init__(
        self,
        state_dim,
        action_dim,
        goal_dim,
        scale,
        model_path,
        model_save_freq,
        buffer_size,
        batch_size,
        start_training_steps):

        self.con = TD3Controller(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            scale=scale,
            model_path=model_path
            )

        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size
            )
        self.model_save_freq = model_save_freq
        self.start_training_steps = start_training_steps

    def step(self, s, env, step, global_step=0, explore=False):
        if explore:
            if global_step < self.start_training_steps:
                a = env.action_space.sample()
            else:
                a = self._choose_action_with_noise(s)
        else:
            a = self._choose_action(s)
        
        obs, r, done, _ = env.step(a)
        n_s = obs['observation']

        return torch.Tensor(a), torch.Tensor(r), torch.Tensor(n_s), torch.Tensor(done)

    def append(self, step, s, a, n_s, r, d):
        self.replay_buffer.append(s, self.fg, a, n_s, r, d)

    def train(self, global_step):
        return self.con.train(self.replay_buffer)

    def _choose_action(self, s):
        return self.con.policy(s, self.fg)

    def _choose_action_with_noise(self, s):
        return self.con.policy_with_noise(s, self.fg)

    def end_step(self):
        pass

    def end_episode(self, episode, logger=None):
        if logger:
            if _is_update(episode, self.model_save_freq):
                self.save(episode=episode)

    def save(self, episode):
        self.con.save(episode)

    def load(self, episode):
        self.con.load(episode)


class Goal_Normalizer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
    def forward(self, goal_5, goal_4, goal_3, goal_2):
        minimum = min(goal_5.detach().min(), goal_4.detach().min(), goal_3.detach().min(), goal_2.detach().min())
        maximum = max(goal_5.detach().max(), goal_4.detach().max(), goal_3.detach().max(), goal_2.detach().max())
        goal_5_norm = (goal_5 - minimum) / (maximum - minimum)
        goal_4_norm = (goal_4 - minimum) / (maximum - minimum)
        goal_3_norm = (goal_3 - minimum) / (maximum - minimum)
        goal_2_norm = (goal_2 - minimum) / (maximum - minimum)
        return goal_5_norm, goal_4_norm, goal_3_norm, goal_2_norm


class HiroAgent(Agent):
    def __init__(
        self,
        state_dim,
        action_dim,
        goal_dim,
        subgoal_dim,
        scale_low,
        start_training_steps,
        model_save_freq,
        model_path,
        buffer_size,
        batch_size,
        buffer_freq,
        train_freq,
        reward_scaling,
        policy_freq_high,
        policy_freq_low,
        time_horizon):

        self.subgoal = Subgoal(subgoal_dim)
        scale_high = self.subgoal.action_space.high * np.ones(subgoal_dim)

        self.model_save_freq = model_save_freq
        #self.policy_network = Policy_Network(state_dim, 300, 1)
        self.policy_network = Policy_Network(goal_dim, 300, 1)

        self.Hierarchy5 = Hierarchy5(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            scale=scale_high,
            model_path=model_path,
            policy_freq=policy_freq_high
            )

        self.Hierarchy4 = Hierarchy4(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            scale=scale_high,
            model_path=model_path,
            policy_freq=policy_freq_high
            )

        self.Hierarchy3 = Hierarchy3(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            scale=scale_high,
            model_path=model_path,
            policy_freq=policy_freq_high
            )

        self.Hierarchy3 = Hierarchy3(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            scale=scale_high,
            model_path=model_path,
            policy_freq=policy_freq_high
            )

        self.Hierarchy2 = Hierarchy2(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            scale=scale_high,
            model_path=model_path,
            policy_freq=policy_freq_high
            )

        self.low_con = LowerController(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            scale=scale_low,
            model_path=model_path,
            policy_freq=policy_freq_low
            )

        self.replay_buffer_low = LowReplayBuffer(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size
            )

        self.replay_buffer_high = HighReplayBuffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            subgoal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            freq=buffer_freq
            )

        self.Hierarchy5_buffer = Hierarchy5_buffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            subgoal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            freq=buffer_freq
            )

        self.Hierarchy4_buffer = Hierarchy4_buffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            subgoal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            freq=buffer_freq
            )

        self.Hierarchy3_buffer = Hierarchy3_buffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            subgoal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            freq=buffer_freq
            )

        self.Hierarchy2_buffer = Hierarchy2_buffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            subgoal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            freq=buffer_freq
            )

        self.buffer_freq = buffer_freq
        self.train_freq = train_freq
        self.reward_scaling = reward_scaling
        self.episode_subreward = 0
        self.sr = 0

        self.buf5 = [None, None, None, 0, None, None, [], []]
        self.buf4 = [None, None, None, 0, None, None, [], []]
        self.buf3 = [None, None, None, 0, None, None, [], []]
        self.buf2 = [None, None, None, 0, None, None, [], []]
        #self.add_to_buffer()
        self.fg = np.array([0,0])
        self.sg5 = torch.Tensor(self.subgoal.action_space.sample())
        self.sg4 = torch.Tensor(self.subgoal.action_space.sample())
        self.sg3 = torch.Tensor(self.subgoal.action_space.sample())
        self.sg2 = torch.Tensor(self.subgoal.action_space.sample())
        self.sg = torch.Tensor(self.subgoal.action_space.sample())
        self.hierarchy_drop = np.ones((1, 3))

        self.start_training_steps = start_training_steps
        self.hierarchies_selected = torch.ones_like(torch.empty(1, 3))
        self.time_horizon = time_horizon
        self.hidden_policy_network = init_hidden(1, 300 * 4 * state_dim,
                                    device=device, grad=True)
        self.masks = [torch.ones(1, 1).to(device) for _ in range(2 * self.time_horizon[4] + 1)]
        self.Goal_Normalizer = Goal_Normalizer(15)

        #self.linear4 = nn.Linear(goal_dim, goal_dim, bias=False)
        #self.linear3 = nn.Linear(goal_dim, goal_dim, bias=False)
        #self.linear2 = nn.Linear(goal_dim, goal_dim, bias=False)

    def step(self, s, env, step, global_step=0, explore=False):
        '''
        if ((step % 300) == 0):
            self.hierarchies_selected, hidden_policy_network = self.policy_network(s, self.sg5, self.sg4, self.sg3, self.hierarchies_selected, self.time_horizon,
                                                                                   self.hidden_policy_network, self.masks[-1], step)
            if (1e-7 > torch.rand(1)[0]):
                self.hierarchies_selected[:, 0] = random.randrange(0,2)
                self.hierarchies_selected[:, 1] = random.randrange(0,2)
                self.hierarchies_selected[:, 2] = random.randrange(0,2)'''

        ## Lower Level Controller
        if explore:
            # Take random action for start_training_steps
            if global_step < self.start_training_steps:
                a = env.action_space.sample()
            else:
                a = self._choose_action_with_noise(s, self.sg)
        else:
            a = self._choose_action(s, self.sg)

        # Take action
        obs, r, done, _ = env.step(a)
        n_s = obs['observation']




        if explore:   # start_training_steps를 잘 조절할 필요가 있어보임. 학습이 안될지도...
            if global_step < self.start_training_steps:
                n_sg5 = torch.Tensor(self.subgoal.action_space.sample())
            else:
                n_sg5 = self._choose_subgoal_with_noise_hierarchy5(step, s, self.sg5, n_s)
        else:
            n_sg5 = self._choose_subgoal_hierarchy5(step, s, self.sg5, n_s)
            
        if explore:
            if global_step < self.start_training_steps:
                n_sg4 = torch.Tensor(self.subgoal.action_space.sample())
            else:
                n_sg4 = self._choose_subgoal_with_noise_hierarchy4(step, s, self.sg4, n_s)
        else:
            n_sg4 = self._choose_subgoal_hierarchy4(step, s, self.sg4, n_s)

        if explore:
            if global_step < self.start_training_steps:
                n_sg3 = torch.Tensor(self.subgoal.action_space.sample())
            else:
                n_sg3 = self._choose_subgoal_with_noise_hierarchy3(step, s, self.sg3, n_s)
        else:
            n_sg3 = self._choose_subgoal_hierarchy3(step, s, self.sg3, n_s)

        if explore:
            if global_step < self.start_training_steps:
                n_sg2 = torch.Tensor(self.subgoal.action_space.sample())
            else:
                n_sg2 = self._choose_subgoal_with_noise2(step, s, self.sg2, n_s)
        else:
            n_sg2 = self._choose_subgoal_hierarchy2(step, s, self.sg2, n_s)

        if ((step % 300) == 0):
            self.hierarchy_drop, hidden_policy_network = self.policy_network(torch.Tensor(obs['observation']), n_sg5, n_sg4, n_sg3, self.hierarchies_selected, self.time_horizon,
                                                      self.hidden_policy_network, self.masks[-1], step)
            if (1e-7 > torch.rand(1)[0]):
                self.hierarchy_drop[:, 0] = random.randrange(0,2)
                self.hierarchy_drop[:, 1] = random.randrange(0,2)
                self.hierarchy_drop[:, 2] = random.randrange(0,2)

        #self.hierarchy_drop = self.policy_network(obs['observation'], n_sg5, n_sg4, n_sg3, self.hierarchies_selected, self.time_horizon, self.hidden_policy_network, self.masks[-1], step)
        n_sg5_n, n_sg4_n, n_sg3_n, n_sg2_n = self.Goal_Normalizer(torch.tensor(n_sg5), n_sg4, n_sg3, n_sg2)
        zero_vec = np.zeros_like(n_sg5_n)
        zero_vec = torch.tensor(zero_vec, dtype=torch.float32).to(device)
        goal = self.back_hierarchy5(step, n_sg5_n, zero_vec, self.hierarchy_drop[0,0])
        goal = self.back_hierarchy4(step, n_sg4_n, goal, self.hierarchy_drop[0,1])
        goal = self.back_hierarchy3(step, n_sg3_n, goal, self.hierarchy_drop[0,2])
        goal = self.back_hierarchy2(step, n_sg2_n, goal, self.hierarchy_drop)
        n_sg = goal

        self.n_sg = n_sg

        return torch.Tensor(a), torch.tensor(r), torch.Tensor(n_s), torch.tensor(done)

    def add_to_buffer(self, step, s, a, n_s, r, d, buff, freq, replay_buffer):

            if  ((step != 0) & ((step % freq) == 0)):
                buff[4] = s
                buff[5] = float(d)
                replay_buffer.append(
                    state=buff[0],
                    goal=buff[1],
                    action=buff[2],
                    n_state=buff[4],
                    reward=buff[3],
                    done=buff[5],
                    state_arr=np.array(buff[6]),
                    action_arr=np.array(buff[7])
                )
            buff = [s, self.fg, self.sg, 0, None, None, [], []]

            buff[3] += self.reward_scaling * r
            buff[6].append(s)
            buff[7].append(a)
            return buff

    def append(self, step, s, a, n_s, r, d):
        self.sr = self.low_reward(s, self.sg, n_s)

        # Low Replay Buffer
        self.replay_buffer_low.append(
            s, self.sg, a, n_s, self.n_sg, self.sr, float(d))

        self.buf5 = self.add_to_buffer(step, s, a, n_s, r, d, self.buf5, freq = self.time_horizon[-1], replay_buffer = self.Hierarchy5_buffer)
        self.buf4 = self.add_to_buffer(step, s, a, n_s, r, d, self.buf4, freq = self.time_horizon[-2], replay_buffer = self.Hierarchy4_buffer)
        self.buf3 = self.add_to_buffer(step, s, a, n_s, r, d, self.buf3, freq = self.time_horizon[-3], replay_buffer = self.Hierarchy3_buffer)
        self.buf2 = self.add_to_buffer(step, s, a, n_s, r, d, self.buf2, freq = self.time_horizon[-4], replay_buffer = self.Hierarchy2_buffer)


    def train(self, global_step):
        losses = {}
        td_errors = {}

        if global_step >= self.start_training_steps:
            loss, td_error = self.low_con.train(self.replay_buffer_low)
            losses.update(loss)
            td_errors.update(td_error)

        if global_step % self.time_horizon[4] == 0:
            loss, td_error = self.Hierarchy5.train(self.Hierarchy5_buffer, self.low_con)
            losses.update(loss)
            td_errors.update(td_error)

        if global_step % self.time_horizon[3] == 0:
            loss, td_error = self.Hierarchy4.train(self.Hierarchy4_buffer, self.low_con)
            losses.update(loss)
            td_errors.update(td_error)

        if global_step % self.time_horizon[2] == 0:
            loss, td_error = self.Hierarchy3.train(self.Hierarchy3_buffer, self.low_con)
            losses.update(loss)
            td_errors.update(td_error)

        if global_step % self.time_horizon[1] == 0:
            loss, td_error = self.Hierarchy2.train(self.Hierarchy2_buffer, self.low_con)
            losses.update(loss)
            td_errors.update(td_error)

        return losses, td_errors

    def _choose_action_with_noise(self, s, sg):
        return self.low_con.policy_with_noise(s, sg)

    def _choose_subgoal_with_noise_hierarchy5(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0: # Should be zero
            sg = self.Hierarchy5.policy_with_noise(s, self.fg)
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        sg = torch.tensor(sg)
        return sg

    def _choose_subgoal_with_noise_hierarchy4(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0: # Should be zero
            sg = self.Hierarchy4.policy_with_noise(s, self.fg)
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        sg = torch.tensor(sg)
        return sg

    def _choose_subgoal_with_noise_hierarchy3(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0: # Should be zero
            sg = self.Hierarchy3.policy_with_noise(s, self.fg)
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        sg = torch.tensor(sg)
        return sg

    def _choose_subgoal_with_noise2(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0: # Should be zero
            sg = self.Hierarchy2.policy_with_noise(s, self.fg)
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        sg = torch.tensor(sg)
        return sg


    def _choose_action(self, s, sg):
        return self.low_con.policy(s, sg)

    def _choose_subgoal_hierarchy5(self, step, s, sg, n_s):
        if step % self.time_horizon[4]== 0:
            sg = self.Hierarchy5.policy(s, self.fg) # 이 부분을 수정
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return torch.tensor(sg)

    def _choose_subgoal_hierarchy4(self, step, s, sg, n_s):
        if step % self.time_horizon[3] == 0:
            sg = self.Hierarchy4.policy(s, self.fg) # 이 부분을 수정
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return torch.tensor(sg)

    def _choose_subgoal_hierarchy3(self, step, s, sg, n_s):
        if step % self.time_horizon[2] == 0:
            sg = self.Hierarchy3.policy(s, self.fg) # 이 부분을 수정
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return torch.tensor(sg)

    def _choose_subgoal_hierarchy2(self, step, s, sg, n_s):
        if step % self.time_horizon[1] == 0:
            sg = self.Hierarchy2.policy(s, self.fg) # 이 부분을 수정
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return torch.tensor(sg)

    def back_hierarchy5(self, step, goal_now, goal_prev, hierarchy_drop):
        sg = self.Hierarchy5.back(goal_now, goal_prev.detach(), hierarchy_drop, drop=True) # 여기서 goal_prev는 영벡터
        return sg
    def back_hierarchy4(self, step, goal_now, goal_prev, hierarchy_drop):
        sg = self.Hierarchy4.back(goal_now, goal_prev.detach(), hierarchy_drop, drop=True) # 이 부분을 수정
        return sg
    def back_hierarchy3(self, step, goal_now, goal_prev, hierarchy_drop):
        sg = self.Hierarchy3.back(goal_now, goal_prev.detach(), hierarchy_drop, drop=True) # 이 부분을 수정
        return sg
    def back_hierarchy2(self, step, goal_now, goal_prev, hierarchy_drop):
        sg = self.Hierarchy2.back(goal_now, goal_prev.detach(), hierarchy_drop, drop=False) # 이 부분을 수정
        return sg

    def subgoal_transition(self, s, sg, n_s):
        return s[:sg.shape[0]] + sg - n_s[:sg.shape[0]]

    def low_reward(self, s, sg, n_s):
        abs_s = torch.Tensor(s[:sg.shape[0]]).to(device) + torch.Tensor(sg).to(device)
        return -torch.sqrt(torch.sum(((abs_s - torch.Tensor(n_s[:sg.shape[0]]).to(device))**2)))

    def end_step(self):
        self.episode_subreward += self.sr
        self.sg = self.n_sg

    def end_episode(self, episode, logger=None):
        if logger: 
            # log
            logger.write('reward/Intrinsic Reward', self.episode_subreward, episode)

            # Save Model
            if _is_update(episode, self.model_save_freq):
                self.save(episode=episode)

        self.episode_subreward = 0
        self.sr = 0
        self.buf = [None, None, None, 0, None, None, [], []]

    def save(self, episode):
        self.low_con.save(episode)
        #self.high_con.save(episode)

    def load(self, episode):
        self.low_con.load(episode)
        #self.high_con.load(episode)
