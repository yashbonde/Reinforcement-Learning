'''
model.py

@yashbonde - 02.02.2019
'''

# dependencies
import tensorflow as tf
from tqdm import tqdm
from tf.contrib import layers

# custom
from agent_base import AgentBase
from state_processor import StateProcessor
from history import History
from param_copy import ModelParametersCopy

class LearnerAgent(AgentBase):
	def __init__(self, env, config):
		self.config = config

		self.env = env

		self.ACTION_SPACE = self.env.action_space.n
		self.OBSERVATION_SPACE = self.env.observation_space.shape

		self.stateProcessor = StateProcessor
		self.history = History

		self.sess = None

	def make_copier(self):
		'''
        As per the paper we need to update the model after every few training iterations.
        Args:
            estimator1: estimator (Network) to copy values from
            estimator2: estimator (Network) to copy values to
        '''
		e1_params = [t for t in tf.trainable_variables() if t.name.startswith('prediction_network')]
        e1_params = sorted(e1_params, key = lambda v:v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith('target_network')]
        e2_params = sorted(e2_params, key = lambda v:v.name)
		
		self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)

	def fill_history_buffer():
		print('[*] Populating Experience Buffer')
		# first reset
		screen = self.env.reset()
		obs_old = self.stateProcessor.process(screen, self.sess)
		for _ in range(self.config.buffer_size):
			action = env.action_space.sample()
			obs_new, reward, done, _ = env.step(action)
			self.history.add([obs_old, action, reward, obs_new, done])

			if done:
				obs_old = env.reset()
			else:
				obs_old = obs_new


	def build_network(self):

		# defining two different networks

		self.screen_placeholder = tf.placeholder(tf.uint8, [self.batch_size, 84, 84, 4], name = sceen)

			screen = tf.to_float(self.screen_placeholder)/255.0 # convert to float and normalise

		with tf.variable_scope('prediction_network'):
			out = layers.conv2d(screen, num_outputs = 32, kerel_size = [8, 8], stride = 4)
			out = layers.conv2d(out, num_outputs = 64, kerel_size = [4, 4], stride = 2)
			out = layers.conv2d(out, num_outputs = 64, kerel_size = [3, 3], stride = 1)
			out = layers.flatten(out)
			out = layers.fully_connected(out, 512)
			self.pred_out = layers.fully_connected(out, self.ACTION_SPACE, activation = tf.nn.softmax)

		with tf.variable_scope('target_network'):
			out = layers.conv2d(screen, num_outputs = 32, kerel_size = [8, 8], stride = 4)
			out = layers.conv2d(out, num_outputs = 64, kerel_size = [4, 4], stride = 2)
			out = layers.conv2d(out, num_outputs = 64, kerel_size = [3, 3], stride = 1)
			out = layers.flatten(out)
			out = layers.fully_connected(out, 512)
			self.target_out = layers.fully_connected(out, self.ACTION_SPACE, activation = tf.nn.softmax)

		# losses and training
		self.loss = 
		

	def init_net(self):
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
		self.make_copier()

	def train(self):
		self.fill_history_buffer()

		self.play_number = 0


		for ep in tdqm(range(config.max_ep)):
			 done = False
			 screen = env.reset()
			 obs = self.state_processor.process(screen, self.sess)

			 while not done:
			 	self.previous_action = self.predict(obs)
			 	if self.play_number % config.action_repeat == 0:
			 		# run the model here
			 		obs, reward, done, _ = env.step(self.previous_action)
			 	else:
			 		# repeat previous action
			 		_, reward, done, _ = env.step(self.previous_action)

			 	####################

			 	q_value = None
			 	q_t_plus_1_value = None

			 	# take training action here

			 	####################

		 		if done:
		 			# update the play number so that it is multiple of config.action_repeat, this way when
		 			# new game starts we will take an action
		 			self.play_number += self.play_number + (config.action_repeat - self.play_number % config.action_repeat)
		 			break