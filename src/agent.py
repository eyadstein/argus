"""
The DQN agent for Argus: replay buffer + action selection + learning update.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.uint8),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.uint8),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, n_actions, device="cpu", lr=1e-4, gamma=0.99,
                 buffer_capacity=100_000, use_double_dqn=True):
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.use_double_dqn = use_double_dqn

        self.online_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0)
            q_values = self.online_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train_step(self, batch_size):
        if len(self.buffer) < batch_size:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states_t = torch.tensor(states, dtype=torch.uint8, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.uint8, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.online_net(states_t)
        q_value = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_double_dqn:
                next_q_online = self.online_net(next_states_t)
                best_next_action = next_q_online.argmax(dim=1)
                next_q_target = self.target_net(next_states_t)
                next_q_value = next_q_target.gather(1, best_next_action.unsqueeze(1)).squeeze(1)
            else:
                next_q_target = self.target_net(next_states_t)
                next_q_value = next_q_target.max(dim=1)[0]

            target = rewards_t + self.gamma * next_q_value * (1 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item(), q_value.mean().item()
