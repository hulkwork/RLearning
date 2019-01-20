import os
dir_name = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0,os.path.join(dir_name, "../"))
from src.deep_q_learning import DeepQLearningPixel
import argparse
import gym

parser = argparse.ArgumentParser(description='Test simple deep q learning.')
parser.add_argument('--game', dest='game', type=str, default="Alien-ram-v0", help='Define your game')

args = parser.parse_args()


# Parameters
observetime = 5000  # Number of timesteps we will be acting on the game and observing results
epsilon = 0.7  # Probability of doing a random move
gamma = 0.7  # Discounted future reward. How much we care about steps further in time
mb_size = 60  # Learning mini batch size
env = gym.make(args.game)  # Choose game (any in the gym should work)
env.reset()
input_shape = env.render('rgb_array').shape
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
q = DeepQLearningPixel(state_size, action_size, input_shape, memory_size=50)
q.exploration(env)
q.play(env)
