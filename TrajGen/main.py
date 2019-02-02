'''
main.py

@yashbonde - 02.02.2019
'''

# importing the dependencies
import argparse
import tensorflow as tf

# custom
from model.ppo_learner import PPOLearnerNetwork
from utils.config import Config
from utils import get_gpu_fraction

# parse args
parser = argparse.ArgumentParser()
# environemnt
parser.add_argument('--env-name', type = str, default = 'Breakout-v4', help = 'Name of gym environment')
parser.add_argument('--action-request', type = int, default = 4, help = 'The number of action to be repeated')
parser.add_argument('--is-train', type = bool, default = True, help = 'Whether model is in training mode')

args = parser.parse_args()

tf.random.set_random_seed(args.random_seed)

if __name__ == '__main__':

	# making config
	config = Config(args)

	# making network
	ppo_learner = PPOLearnerNetwork(config)