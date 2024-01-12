import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np

import torch.nn.functional as F
import torch.nn as nn

from copy import deepcopy
import math

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

    def forward(self, state):
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim = 0):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        return prob

class Critic(nn.Module):
    def __init__(self, state_dim,net_width):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.relu(self.C1(state))
        v = torch.relu(self.C2(v))
        v = self.C3(v)
        return v


class PPO_agent(nn.Module):
    """
    PPO implementation for Discrete action spaces. PPO implementation is based on the code: https://github.com/XinJingHao/PPO-Discrete-Pytorch/tree/main
    """
    def __init__(self, state_dim, action_dim, lr, device, entropy_coef, entropy_coef_decay, epochs, batch_size, clip_rate, adv_normalization, gamma, lambd, l2_reg):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.device = device
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_rate = clip_rate
        self.adv_normalization = adv_normalization
        self.gamma = gamma
        self.lambd = lambd
        self.l2_reg = l2_reg

        self.net_width = 16 # TODO: set this properly

        '''Build Actor and Critic'''
        self.actor = Actor(self.state_dim, self.action_dim, self.net_width).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic = Critic(self.state_dim, self.net_width).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)


    def choose_action(self, state, deterministic=False):
        assert len(state.shape) == 1, 'pass state with no batch dimension'
        s = torch.from_numpy(state).float().to(self.device).unsqueeze(0) # add batch dim

        with torch.no_grad():
            pi = self.actor.pi(s, softmax_dim=-1)[0]
            if deterministic:
                a = torch.argmax(pi).item()
                return a, None
            else:
                m = Categorical(pi)
                a = m.sample().item()
                pi_a = pi[a].item()
                return a, pi_a
    


    def train(self, state_buffer, action_buffer, log_prob_action_buffer, reward_buffer, done_buffer, next_state_buffer):
        self.entropy_coef *= self.entropy_coef_decay #exploring decay
        '''Prepare PyTorch data'''
        s = torch.tensor(state_buffer, dtype=torch.float32).to(self.device) # [num_steps, state_dim]
        a = torch.tensor(action_buffer, dtype=torch.int64).unsqueeze(1).to(self.device) # [num_steps, 1]
        r = torch.tensor(reward_buffer, dtype=torch.float32).unsqueeze(1).to(self.device) # [num_steps, 1]
        s_next = torch.tensor(next_state_buffer, dtype=torch.float32).to(self.device) # [num_steps, state_dim]
        old_prob_a = torch.tensor(log_prob_action_buffer, dtype=torch.float32).unsqueeze(1).to(self.device) # [num_steps, 1]
        done = torch.tensor(done_buffer, dtype=torch.bool).unsqueeze(1).to(self.device) # [num_steps, 1]
        dw = torch.tensor(done_buffer, dtype=torch.bool).unsqueeze(1).to(self.device) # [num_steps, 1]

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s) # [num_steps, 1]
            vs_ = self.critic(s_next) # [num_steps, 1]

            '''dw(dead and win) for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (~dw) - vs # [num_steps, 1]
            deltas = deltas.cpu().flatten().numpy() # [num_steps,]
            adv = [0]

            '''done for GAE'''
            for dlt, done in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~done)
                adv.append(advantage)
            adv.reverse()
            adv = deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.device) # [num_steps, 1]
            td_target = adv + vs # [num_steps, 1]
            if self.adv_normalization:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  #sometimes helps

        """PPO update"""
        #Slice long trajectopy into short trajectory and perform mini-batch PPO update
        optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))

        actor_epoch_losses = []
        critic_epoch_losses = []
        for _ in range(self.epochs):
            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0]) # [num_steps,]
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device) # [num_steps]
            s, a, td_target, adv, old_prob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone() # [num_steps, state_dim], [num_states, 1], [num_states, 1], [num_steps, 1], [num_steps, 1]

            '''mini-batch PPO update'''
            actor_batch_losses = []
            critic_batch_losses = []
            for i in range(optim_iter_num):
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0])) # generate indices for the current batch

                '''actor update'''
                prob = self.actor.pi(s[index], softmax_dim=1) # [B, action_dim]
                entropy = Categorical(prob).entropy().sum(0, keepdim=True) # [1]
                prob_a = prob.gather(1, a[index])
                ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_a[index]))  # a/b == exp(log(a)-log(b)) # [B, 1]

                surr1 = ratio * adv[index] # [B, 1]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index] # [B, 1]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy # [B, 1]

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                '''critic update'''
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()

                # log losses
                actor_batch_losses.append(a_loss.mean().detach().cpu().item())
                critic_batch_losses.append(c_loss.detach().cpu().item())
        
            # log avg epoch losses
            actor_epoch_losses.append(np.mean(actor_batch_losses))
            critic_epoch_losses.append(np.mean(critic_batch_losses))

        return actor_epoch_losses, critic_epoch_losses