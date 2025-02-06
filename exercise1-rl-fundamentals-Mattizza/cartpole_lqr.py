'''
    Robot Learning
    Exercise 1

    Linear Quadratic Regulator

    Polito A-Y 2024-2025
'''
import gym
import numpy as np
from scipy import linalg     # get riccati solver
import argparse
import matplotlib.pyplot as plt
import sys
from utils import get_space_dim, set_seed
import pdb 
import time
import pickle as pkl

# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Environment to use')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--time_sleep', action='store_true',
                        help='Add timer for visualizing rendering with a slower frame rate')
    parser.add_argument('--mode', type=str, default='control',
                        help="Type of test ['control', 'multiple_R']")
    parser.add_argument('--path', type=str, default='data',
                        help='Add path to store the sequence of states')
    return parser.parse_args(args)

def linerized_cartpole_system(mp, mk, lp, g=9.81):
    mt=mp+mk
    # state matrix
    # a1 = 0
    a1 = (g*mp)/((mk + mp)*(mp/(mk + mp) - 4/3))
    a2 = - g /(lp*((mp/mt)-4/3))
    
    A = np.array([[0, 1, 0,  0],
                  [0, 0, a1, 0],
                  [0, 0, 0,  1],
                  [0, 0, a2, 0]])

    # input matrix
    # b1 = 1/mt
    b1 = -(mp/(mt*(mp/mt - 4/3)) - 1)/mt
    b2 = 1 / (lp*mt*((mp/mt)-4/3))
    B = np.array([[0], [b1], [0], [b2]])
    
    return A, B

def optimal_controller(A, B, R_value=1):
    R = R_value*np.eye(1, dtype=int)  # choose R (weight for input)
    Q = 5*np.eye(4, dtype=int)        # choose Q (weight for state)
   # solve ricatti equation
    P = linalg.solve_continuous_are(A, B, Q, R)

    # calculate optimal controller gain
    K = np.dot(np.linalg.inv(R),
            np.dot(B.T, P))
    return K

def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left

def multiple_R(env, mp, mk, l, g, time_sleep=False, terminate=True):
    '''
    Vary the value of R within the range [0.01, 0.1, 10, 100] and plot the forces 
    '''
    #TODO: 
    # Done\
    
    R_values = [0.01, 0.1, 1, 10, 100]
    for R in R_values:
        print(f'R = {R}')
        control(env, mp, mk, l, g, R, time_sleep, terminate)
    return

def control(env, mp, mk, l, g, R=None, time_sleep=False, terminate=True):
    '''
    Control using LQR
    '''
    #TODO: plot the states of the system ...
    # Done!
    set_seed(args.seed)    # seed for reproducibility
    env.env.seed(args.seed)

    obs = env.reset()    # Reset the environment for a new episode

    state_seq = np.zeros((400, 4))
    forces_seq = np.zeros((400, 7))
    
    if R is None:
        A, B = linerized_cartpole_system(mp, mk, l, g)
        K = optimal_controller(A, B)    # Re-compute the optimal controller for the current R value
    else:
        A, B = linerized_cartpole_system(mp, mk, l, g)
        K = optimal_controller(A, B, R)

    for i in range(1000):

        env.render()
        if time_sleep:
            time.sleep(.1)
        
        # get force direction (action) and force value (force)
        action, force = apply_state_controller(K, obs)
        
        # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
        abs_force = abs(float(np.clip(force, -10, 10)))
        
        # change magnitute of the applied force in CartPole
        env.env.force_mag = abs_force

        # apply action
        obs, reward, done, _ = env.step(action)

        # store state for plotting
        state_seq[i] = obs
        forces_seq[i, :4] = obs
        forces_seq[i, 4] = force
        forces_seq[i, 5] = abs_force
        forces_seq[i, 6] = R
        
        if (terminate and done) and (i > 398):
            print(f'Terminated after {i+1} iterations.')
            if R is None:
                with open(args.path + '/state_seq.pkl', 'wb') as f:
                    pkl.dump(state_seq, f)
            else:
                with open(args.path + f'/state_force_seq_R_{R}.pkl', 'wb') as f:
                    pkl.dump(forces_seq, f)  
            break

# The main function
def main(args):
    # Create a Gym environment with the argument CartPole-v0 (already embedded in)
    env = gym.make(args.env)

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Print some stuff
    print('Environment:', args.env)
    print('Observation space dimensions:', observation_space_dim)
    print('Action space dimensions:', action_space_dim)

    set_seed(args.seed)    # seed for reproducibility
    env.env.seed(args.seed)
     
    mp, mk, l, g = env.masspole, env.masscart, env.length, env.gravity

    if args.mode == 'control':
        control(env, mp, mk, l, g, None, args.time_sleep, terminate=True)
    elif args.mode == 'multiple_R':
        multiple_R(env, mp, mk, l, g, args.time_sleep, terminate=True)

    env.close()

# Entry point of the script
if __name__ == '__main__':
    args = parse_args()
    main(args)

