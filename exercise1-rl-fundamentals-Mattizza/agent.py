import torch
import torch.nn.functional as F
from utils import discount_rewards

import sys


# NOTE: added one additional layer
class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, reward_type):
        super().__init__()
        self.state_space = state_space + 2
        self.action_space = action_space
        self.reward_type = reward_type
        self.fc1 = torch.nn.Linear(state_space + 2, 12)
        if self.reward_type is None:
            self.fc2 = torch.nn.Linear(12, 12)
        self.fc3 = torch.nn.Linear(12, action_space)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 1e-1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.reward_type is None :
            x = self.fc1(x)
            x = F.sigmoid(x)
            x = self.fc2(x)
            x = F.sigmoid(x)
            x = self.fc3(x)
        else:
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc3(x)
        return F.softmax(x, dim=-1)


class Agent(object):
    def __init__(self, policy):
        self.train_device = "mps"  # "cuda" if torch.cuda.is_available() else "mps" or "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
        self.batch_size = 1
        self.gamma = 0.98
        # self.gamma = 1
        self.observations = []
        self.actions = []
        self.rewards = []

    def episode_finished(self, episode_number):
        all_actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        self.observations, self.actions, self.rewards = [], [], []
        discounted_rewards = discount_rewards(all_rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        # element-wise multiplication, such that for each action we have -ln(outputnetwork)*disc_reward_normalized
        weighted_probs = all_actions * discounted_rewards 

        # print('all_actions:', all_actions)
        # print('discounted rewards:', discounted_rewards)
        # print('weighted probs:', weighted_probs)
        # sys.exit()

        # You want to perform gradient descent on the average loss, so to decrease the overall mean loss
        # => less probability for actions that led to below average rewards,
        # and more probability for actions that led to above average rewards.
        loss = torch.mean(weighted_probs)
        loss.backward()

        if (episode_number+1) % self.batch_size == 0:
            self.update_policy()

    def update_policy(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)
        aprob = self.policy.forward(x)
        if evaluation:
            action = torch.argmax(aprob).item()
        else:
            dist = torch.distributions.Categorical(aprob)
            action = dist.sample().item()
        return action, aprob

    def store_outcome(self, observation, action_output, action_taken, reward):
        dist = torch.distributions.Categorical(action_output)
        action_taken = torch.Tensor([action_taken]).to(self.train_device)
        log_action_prob = -dist.log_prob(action_taken) # -ln(networkoutput)

        self.observations.append(observation)
        self.actions.append(log_action_prob)
        self.rewards.append(torch.Tensor([reward]))
