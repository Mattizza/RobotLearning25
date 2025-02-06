"""
    Robot Learning
    Exercise 1

    Reinforcement Learning 

    Polito A-Y 2024-2025
"""
import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import sys
import time
from agent import Agent, Policy
from utils import get_space_dim

import sys


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--render_training", action='store_true',
                        help="Render each frame during training. Will be slower")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument("--reward_type", default=None,
                        help="Reward to use. Default is the one encouraging the sliding behaviour, otherwise by passing "
                        "a value you will encourage the agent to stay close to it")
    parser.add_argument("--random_policy", action='store_true', help="Applying a random policy training")
    parser.add_argument('--path', type=str, default='params',
                        help='Add path to store the weights learned at training time')
    parser.add_argument('--max_timesteps', type=int, default=None,
                        help='Maximum number of timesteps over which training or testing')
    return parser.parse_args(args)


# Policy training function
def train(agent, env, train_episodes, reward_type=None, early_stop=True, render=False,
          silent=False, train_run_id=0, random_policy=False):
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []
    max_reward_sum = -1
    if reward_type is None:
        name = 'slide'
    elif reward_type == 'default':
        name = 'default'
    else:
        name = 'affine'

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state (it's a random initial state with small values)
        observation = env.reset()
        observation = np.hstack((observation, np.zeros(2)))
        checkpoints = np.array([-1.8, 1.8])
        visited_checkpoints = 0
        bonus = False
        
        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)

            if random_policy:
                # Task 1.1
                """
                Sample a random action from the action space
                """
                # TODO
                # Done!
                action = env.action_space.sample()
                
            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            # note that after env._max_episode_steps the episode is over, if we stay alive that long
            observation, reward, done, info = env.step(action)

            if (np.abs(observation[0] - checkpoints[visited_checkpoints]) <= 0.05):
                visited_checkpoints = 1 if observation[0] <= 1.7 else 0
                bonus = True

            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            # TODO
            # Done!
            tmp = np.zeros(2)
            if reward_type is None:
                tmp[visited_checkpoints] = 1
                observation = np.hstack((observation, tmp))
                reward = new_reward(observation, 'slide', checkpoints=checkpoints, visited_checkpoints=visited_checkpoints, bonus=bonus)
            elif reward_type == 'default':
                observation = np.hstack((observation, np.zeros(2)))
            else:
                observation = np.hstack((observation, np.zeros(2)))
                reward = new_reward(observation, 'affine', reward_type)

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Draw the frame, if desired
            if render:
                env.render()

            # Store total episode reward
            reward_sum += reward
            timesteps += 1
            bonus = False

        if not silent:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))
        
        if reward_sum > max_reward_sum:
            model_file = f'{args.path}/CartPole-v0_params_{name}_early_{args.max_timesteps}.ai'
            torch.save(agent.policy.state_dict(), model_file)
            print("Model saved to", model_file)
            max_reward_sum = reward_sum

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # If we managed to stay alive for 15 full episodes, assume it's learned
        # (in the default setting)
        if early_stop and np.mean(timestep_history[-15:]) == env._max_episode_steps:
            if not silent:
                print("Looks like it's learned. Finishing up early")
            break

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

    # Store the data in a Pandas dataframe for easy visualization
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         "reward": reward_history,
                         "mean_reward": average_reward_history})
    return data


# Function to test a trained policy
def test(agent, env, episodes, reward_type=None, render=False):
    test_reward, test_len, timesteps = 0, 0, 0

    episodes = 100
    print('Num testing episodes:', episodes)
    
    # Uncomment for statistics
    # observation_data = pd.DataFrame(columns=['x', 'x_dot', 'theta', 'theta_dot', 'episode'])

    for ep in range(episodes):
        done = False
        observation = env.reset()
        observation = np.hstack((observation, np.zeros(2)))
        checkpoints = np.array([-1.8, 1.8])
        visited_checkpoints = 0
        bonus = False
        position, velocity, angle, angular_velocity, reward_list, action_list = [], [], [], [], [], []
        
        while not done:
        # Task 1.2
            """
            Test on 500 timesteps
            """
            # TODO
            # Done!

            action, _ = agent.get_action(observation, evaluation=True)  # Similar to the training loop above -
                                                                        # get the action, act on the environment, save total reward
                                                                        # (evaluation=True makes the agent always return what it thinks to be
                                                                        # the best action - there is no exploration at this point)
            observation, reward, done, info = env.step(action)
            position.append(observation[0])
            velocity.append(observation[1])
            angle.append(observation[2])
            angular_velocity.append(observation[3])
            action_list.append(action)

            if np.abs(observation[0] - checkpoints[visited_checkpoints]) <= 0.05:
                visited_checkpoints = 1 if observation[0] <= 1.7 else 0
                bonus = True

            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            # TODO
            # Done!
            tmp = np.zeros(2)
            if reward_type is None:
                tmp[visited_checkpoints] = 1
                observation = np.hstack((observation, tmp))
                reward = new_reward(observation, 'slide', checkpoints=checkpoints, visited_checkpoints=visited_checkpoints, bonus=bonus)
            elif reward_type == 'default':
                observation = np.hstack((observation, np.zeros(2)))
            else:
                observation = np.hstack((observation, np.zeros(2)))
                reward = new_reward(observation, 'affine', reward_type)
            
            reward_list.append(reward)
            if render:
                env.render()
            test_reward += reward
            test_len += 1
            timesteps += 1
            bonus = False

    #     observation_data = pd.concat([observation_data, pd.DataFrame({'x': position, 'x_dot': velocity, 
    #                                                                   'theta': angle, 'theta_dot': angular_velocity, 'episode': ep, 'reward': reward_list, 'action': action_list})])

    # with open(f'dum_{time.time()}.pkl', 'wb') as f:
    #     pkl.dump(observation_data, f)
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)


def new_reward(state, reward_type, goal_position=0, checkpoints=None, visited_checkpoints=None, bonus=None):
    # Task 3.1
    """
        Use a different reward, overwriting the original one
    """
    # TODO
    # Done!
    if reward_type == 'affine':
        goal_position = float(goal_position)
        reward = (1 + np.abs(goal_position)) - np.abs(state[0] - goal_position) + (5 if np.abs(state[0] - goal_position) <= 0.05 else 0)
    else:
        reward = 4 - np.abs(checkpoints[visited_checkpoints] - state[0]) + (1 if np.abs(state[1]) >= 0.5 else 0) + (500 if bonus else 0) - (100 if np.abs(state[0]) >= 2.2 else 0)
    return reward

# The main function
def main(args):
    # Create a Gym environment with the argument CartPole-v0 (already embedded in)
    env = gym.make(args.env)

    # Task 1.2
    """
    # For CartPole-v0 - change the maximum episode length
    """
    # TODO
    # Done!
    if args.max_timesteps is None:
            env._max_episode_steps = np.inf
    else:
        env._max_episode_steps = args.max_timesteps

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim, args.reward_type)
    agent = Agent(policy)

    # Print some stuff
    print("Environment:", args.env)
    print("Training device:", agent.train_device)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        # Train
        training_history = train(agent, env, args.train_episodes, args.reward_type, False, args.render_training, random_policy=args.random_policy)
    
        # Save the model
        if args.random_policy:
            model_file = f'{args.path}/{args.env}_params_random_{args.max_timesteps}.ai'
        else:
            model_file = f'{args.path}/{args.env}_params_{args.max_timesteps}.ai'
        torch.save(policy.state_dict(), model_file)
        print("Model saved to", model_file)

        # Plot rewards
        sns.lineplot(x="episode", y="reward", data=training_history, color='blue', label='Reward')
        sns.lineplot(x="episode", y="mean_reward", data=training_history, color='orange', label='100-episode average')
        plt.legend()
        plt.title("Reward history (%s)" % args.env)
        plt.show()
        print("Training finished.")
        with open(f'dum_{time.time()}.pkl', 'wb') as f:
                    pkl.dump(training_history, f)
    else:
        # Test
        print("Loading model from", args.test, "...")
        state_dict = torch.load(args.test)
        policy.load_state_dict(state_dict)
        print("Testing...")
        test(agent, env, args.train_episodes, args.reward_type, args.render_test)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

