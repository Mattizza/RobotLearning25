import gym
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
from agent import Agent, Policy
import random
import seaborn as sns
import pandas as pd
import argparse
import pickle as pkl
import sys
import torch
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument("--epsilon", "--e", type=float, default=0.2,
                    help="Epsilon value for epsilon-greedy policy. If not GLIE, it will be constant, "
                    "otherwise it represents the last epsilon value computed through a linear decrease.")
parser.add_argument("--glie", "--g", action='store_true',
                    help="Whether to use GLIE schedule for epsilon.")
parser.add_argument("--episodes_glie", "--tg", type=int, default=20_000,
                    help="Number of episodes to reach the final epsilon value in GLIE.")
parser.add_argument("--episodes", "--ep", type=int, default=20_000,
                    help="Number of episodes to train the agent.")
parser.add_argument("--buffer_size", "--bs", type=int, default=1_000,
                    help="Size of the replay buffer.")
parser.add_argument("--epochs", "--epc", type=int, default=1,
                    help="Number of epochs to train the agent.")
parser.add_argument("--max_timesteps", "--mt", type=int, default=200,
                    help="Maximum number of timesteps per episode.")
parser.add_argument("--reset", "--r", type=int, default=10,
                    help="Number of episodes before changing the target policy.")
parser.add_argument("--counter_sampling", "--cs", type=int, default=100,
                    help="Number of episodes before sampling the replay buffer.")
parser.add_argument("--big_replay", "--br", type=int, default=10_000,
                    help="Size of the big replay buffer.")
args = parser.parse_args()

np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

# Whether to perform training or use the stored .npy file
MODE = 'TRAINING' # TRAINING, TEST

# episodes = args.episodes
episodes = args.episodes
test_episodes = 100
num_of_actions = 2  # 2 discrete actions for Cartpole

gamma = 0.98
alpha = 0.1
epsilon = args.epsilon
b = int(args.epsilon * args.episodes_glie / (1 - args.epsilon))
epochs = args.epochs

def get_space_dim(space):
    t = type(space)
    if t is gym.spaces.Discrete:
        return space.n
    elif t is gym.spaces.Box:
        return space.shape[0]
    else:
        raise TypeError("Unknown space type:", t)
    
def custom_reward(state, old_state, done, timestep):
    x, x_dot, theta, theta_dot = state
    x_old, x_dot_old, theta_old, theta_dot_old = old_state
    if not done:
        return 1 + (-0.5 if (np.abs(theta) > np.abs(theta_old)) else 0.5) + (-0.5 if (np.abs(theta_dot) > np.abs(theta_dot_old)) else 0.5) + (-0.5 if (np.abs(x_dot) > np.abs(x_dot_old)) else 0.5)
    elif (done) and (timestep < 499):
        return -2
    else:
        return 2
    
observation_space_dim = get_space_dim(env.observation_space)
action_space_dim = get_space_dim(env.action_space)

target_policy = Policy(observation_space_dim, action_space_dim)
target_agent = Agent(target_policy)

policy = Policy(observation_space_dim, action_space_dim)
agent = Agent(policy, target_agent)

if MODE == 'TEST':
    q_grid = np.load('q_values.npy')

# Training loop
buffer_size = args.buffer_size
ep_lengths, epl_avg, reward_episode, epsilon_episode = [], [], [], []
replay_memory = np.zeros((buffer_size, 11))
big_replay = np.zeros((args.big_replay, 11))
counter = 0
counter_train = 0
counter_sampling = 0
counter_big_replay = 0
env._max_episode_steps = args.max_timesteps

for ep in range(episodes):
    test = ep > episodes
    cum_reward = 0

    if (counter_train % args.reset == 0) and (ep > 0):
        print('Changing target policy.')
        target_agent = deepcopy(agent)
        for param in target_agent.policy.parameters():
            param.requires_grad = False
        agent.target_agent = target_agent
        counter_train = 0
    counter_train += 1

    if MODE == 'TEST':
        test = True

    state, done, steps = env.reset(), False, 0

    if args.glie:
        epsilon = b / (b + ep)
    else:
        pass

    while not done:
        if counter_sampling == args.counter_sampling:
            # Perform training
            non_zero_indices = np.where(~np.all(big_replay == 0, axis=1))[0]
            if len(non_zero_indices) >= buffer_size:
                sampled_indices = np.random.choice(non_zero_indices, buffer_size, replace=False)
            else:
                sampled_indices = non_zero_indices
            replay_memory = big_replay[sampled_indices]

            agent.train(replay_memory, epochs)
            counter_sampling = 0

        action, aprob = agent.get_action(state, evaluation=True, epsilon=epsilon)
        if action not in [0, 1]:
            action = 0
        new_state, reward, done, _ = env.step(action)
        
        if not test:
            reward = custom_reward(new_state, state, done, steps)
            big_replay[counter_big_replay % args.big_replay] = np.hstack((state, new_state, done, reward, action))
        else:
            env.render()
            pass

        state = new_state
        steps += 1
        cum_reward += reward
        counter += 1
        counter_sampling += 1
        counter_big_replay += 1
    reward_episode.append(cum_reward)
    epsilon_episode.append(epsilon)

    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 1 == 0:
        print(f"Episode {ep}, episode reward: {reward_episode[-1]}, episode length: {ep_lengths[-1]}")
        print('Epsilon:', epsilon)

mode = 'glie' if args.glie else 'constant'
with open(f'dum.pkl', 'wb') as f:
    pkl.dump(pd.DataFrame({'reward' : reward_episode,
                           'ep_lengths' : ep_lengths}), f)

if MODE == 'TEST':
    sys.exit()