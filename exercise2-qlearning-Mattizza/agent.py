import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

import sys

class ReplayMemoryDataset(Dataset):
    def __init__(self, replay_memory):
        self.replay_memory = replay_memory

    def __len__(self):
        return len(self.replay_memory)

    def __getitem__(self, idx):
        return self.replay_memory[idx]

# NOTE: added one additional layer
class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.fc1 = torch.nn.Linear(state_space, 12)
        self.fc2 = torch.nn.Linear(12, 12)
        self.fc3 = torch.nn.Linear(12, action_space)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 1e-1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class Agent(object):
    def __init__(self, policy, target_agent=None):
        self.train_device = "mps"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
        # FIXME
        self.batch_size = 32
        self.gamma = 0.98
        # self.gamma = 1
        self.observations = []
        self.actions = []
        self.rewards = []
        self.target_agent = target_agent
        
        # Freeze gradients of target agent at initialization
        if self.target_agent is None:
            for param in self.policy.parameters():
                param.requires_grad = False

    def train(self, replay_memory, epochs):
        dataset = ReplayMemoryDataset(replay_memory)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for e in range(epochs):
            loss_list = []
            for batch in dataloader:
                self.optimizer.zero_grad()

                states = batch[:, 0:4].float().to(self.train_device)
                new_states = batch[:, 4:8].float().to(self.train_device)
                done = batch[:, 8].float().to(self.train_device)
                rewards = batch[:, 9].float().to(self.train_device)
                actions = batch[:, 10].long().to(self.train_device)
                
                with torch.no_grad():
                    target_train_obs = torch.max(self.target_agent.policy.forward(new_states), dim=1).values
                    TD_targets = rewards + (1 - done) * self.gamma * target_train_obs

                # Compute the loss
                q_values = self.policy(states)
                q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                loss = F.mse_loss(q_values, TD_targets)

                loss.backward()
                for param in self.policy.parameters():
                    param.grad.data.clamp_(-1.5, 1.5)
                self.optimizer.step()
                loss_list.append(loss.item())

            loss_list = []

    def get_action(self, observation, evaluation=False, epsilon=0.0):
        x = torch.from_numpy(observation).float().to(self.train_device)
        aprob = self.policy.forward(x)
        if evaluation:
            if torch.rand(1).item() < epsilon:
                action = torch.randint(0, self.policy.action_space, (1,)).item()
            else:
                action = torch.argmax(aprob).item()
        else:
            dist = torch.distributions.Categorical(aprob)
            action = dist.sample().item()
        return action, aprob
