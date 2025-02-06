import gym
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
import random
import seaborn as sns
import pandas as pd
import argparse
import pickle as pkl
import sys

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
parser.add_argument("--discretization", "--d", type=int, default=16,
                    help="Discretization of the state space.")
parser.add_argument("--q_init", "--q", type=float, default=0,
                    help="Initial value for Q-values.")
parser.add_argument("--max_timesteps", "--mt", type=int, default=200,
                    help="Maximum number of timesteps per episode.")
args = parser.parse_args()

np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

# Whether to perform training or use the stored .npy file
MODE = 'TRAINING' # TRAINING, TEST

episodes = args.episodes
test_episodes = 100
num_of_actions = 2  # 2 discrete actions for Cartpole

# Reasonable values for Cartpole discretization
discr = args.discretization
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# Parameters
gamma = 0.98
alpha = 0.1
epsilon = args.epsilon
env._max_episode_steps = args.max_timesteps

# constant_eps = 0.2
# TODO: choose b so that with GLIE we get an epsilon of 0.1 after 20'000 episodes
# b is dynamically updated depending on the desired number of episodes. All experiments
# have been run with 40,000 episodes just to have more data a a better convergence.
b = int(args.epsilon * args.episodes_glie / (1 - args.epsilon))

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

# Initialize Q values
q_grid = np.ones((discr, discr, discr, discr, num_of_actions)) * args.q_init

if MODE == 'TEST':
    q_grid = np.load('q_values.npy')

def find_nearest(array, value):
    return np.argmin(np.abs(array - value))

def get_cell_index(state):
    """Returns discrete state from continuous state"""
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av


def get_action(state, q_values, greedy=False):
    x, v, th, av = get_cell_index(state)

    if greedy: # TEST -> greedy policy
        # TODO: greedy w.r.t. q_grid
        best_action_estimated = np.argmax(q_values[x, v, th, av, :]) 

        return best_action_estimated

    else: # TRAINING -> epsilon-greedy policy
        if np.random.rand() < epsilon:
            # Random action
            # TODO: choose random action with equal probability among all actions
            action_chosen = env.action_space.sample()
            return action_chosen
        else:
            # Greedy action
            # TODO: greedy w.r.t. q_grid
            best_action_estimated = np.argmax(q_values[x, v, th, av, :])
            return best_action_estimated


def update_q_value(old_state, action, new_state, reward, done, q_array):
    old_cell_index = get_cell_index(old_state)
    new_cell_index = get_cell_index(new_state)
    pi_value = np.max(q_array[new_cell_index[0], new_cell_index[1], new_cell_index[2], new_cell_index[3], :])
    mu_value = q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action]

    # Target value used for updating our current Q-function estimate at Q(old_state, action)
    if done is True:
        target_value = reward  # HINT: if the episode is finished, there is not next_state. Hence, the target value is simply the current reward.
    else:
        # TODO
        target_value = reward + gamma * pi_value 

    # Update Q value
    q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] = q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] + alpha * (target_value - mu_value)
    return


# Training loop
ep_lengths, epl_avg, reward_episode, epsilon_episode = [], [], [], []
for ep in range(episodes+test_episodes):
    test = ep > episodes
    cum_reward = 0

    if MODE == 'TEST':
        test = True

    state, done, steps = env.reset(), False, 0

    if args.glie:
        epsilon = b / (b + ep)
    else:
        pass

    while not done:
        action = get_action(state, q_grid, greedy=test)
        if action not in [0, 1]:
            action = 0
        new_state, reward, done, _ = env.step(action)
        if not test:
            update_q_value(state, action, new_state, reward, done, q_grid)
        else:
            # env.render()
            pass

        state = new_state
        steps += 1
        cum_reward += reward
    reward_episode.append(cum_reward)
    epsilon_episode.append(epsilon)

    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))
        print('Epsilon:', epsilon)

mode = 'glie' if args.glie else 'constant'
with open(f'dum.pkl', 'wb') as f:
    pkl.dump(pd.DataFrame({'reward' : reward_episode,
                           'epsilon' : epsilon_episode}), f)

if MODE == 'TEST':
    sys.exit()

# Save the Q-value array
np.save(f"dum.npy", q_grid)