'''
main.py

@yashbonde - 02.02.2019
'''

# importing the dependencies
import argparse
import tensorflow as tf

# custom
from model import LearnerAgent
from utils.config import Config
from utils import get_gpu_fraction

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', type = str, default = 'Breakout-v4', help = 'Name of gym environment')
parser.add_argument('--action-request', type = int, default = 4, help = 'The number of action to be repeated')
parser.add_argument('--is-train', type = bool, default = True, help = 'Whether model is in training mode')
parser.add_argument('--random-seed', type = int, default = 123, help = 'Seed for random numbers')
parser.add_argument('--save-folder', type = str, default = './experiments', help = 'Folder to save files')
args = parser.parse_args()

tf.random.set_random_seed(args.random_seed)

if __name__ == '__main__':

	# making config
	args.save_folder = args.save_folder + '/' + self.env_name + '/'
	config = Config(args)

	# making env
	env = gym.make(args.env_name)

	# making network
	traj_generator = PPOLearnerNetwork(config, env)
	traj_generator.init_network()
	traj_generator.train()