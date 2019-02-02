'''
main.py

@yashbonde - 01.02.2019
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

# etc
parser.add_argument('--use-gpu', type = bool, default = False, help = 'Whether to use gpu or not')
parser.add_argument('--gpu-fraction', type = str, default = '1/1', help = 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
parser.add_argument('--display', type = bool, default = False, help = 'Whether to display the game screening or not')
parser.add_argument('--is_train', type = bool, default = True, help = 'Whether to do training or testing')
parser.add_argument('--random-seed', type = int, default = 123, help = 'Value of the random seed')

args = parser.parse_args()

tf.random.set_random_seed(args.random_seed)

if __name__ == '__main__':

	# making config
	config = Config(args)

	# making network
	ppo_learner = PPOLearnerNetwork(config)