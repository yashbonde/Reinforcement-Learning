# importing the dependencies
import numpy as np
import tensorflow as tf
import gym 

# defining Experience Replay Memory Class
class ExperienceReplayMemory(object):
	"""
	ExperienceReplayMemory: The memory which contains past occurrences.
		This is same as the one given in the paper.
	Args:
		mem_size: size of memory (N)
		batch_size: size of minibatch
	"""
	def __init__(self, mem_size, batch_size):
		self.mem_size = mem_size
		self.batch_size = batch_size

		self.memory = list()

	def remember(self, states, done):
		"""
		Function to store the new states to the memory
		Args:
			states: a tuple <s,a,r,s'>
			done: whether the game ended or not
		"""

		self.memory.append([states, done])
		if len(self.memory) > self.mem_size:
			del self.memory[0]

	def get_batch(self):
		"""
		Function to get a minibatch of size self.batch_size
		"""
		batch = np.arange(len(self.mem_size))
		batch = batch[:self.batch_size]

		return np.array(self.memory)[batch]

# Deep Q-network	
class DQN(object):
	"""
	DQN: Deep Q-Network class. As we will be using OpenAI Gym we do not need
		as many arguments as given in the paper.
	Args:
		env: environment used
		output_size: output size of the network
		sess = tensorflow session
		num_episodes: number of episodes of game run
		num_epochs: number of epochs to train for each episode
		mem_size: size of the memory (N)
		batch_size: size of minibatch
		intial_epsilon: initial value of epsilon
		final_epsilon: final value of epsilon
		final_exploration_frame: steps over which the epsilon will be reduced
		replay_start_size: number of steps while random action policy is
			implemented just to gather experience
		discount_factor: discount factor gamma
		learning_rate: learning rate of model
		gradient_momentum: gradient momentum for RMSProp Optimizer
		network_update_frequency: frequency (measured in terms of parameter
			updates) with which target network is updated (C)
		disp_step: display results after these many interations
		test_cycles: how many times to test the model
		render_test: to render the environment while training or not
		render_true: to render the environment while testing or not
	"""
	def __init__(self, env, output_size, sess, num_episodes, num_epochs,
			mem_size = 1000000, batch_size = 32, intial_epsilon = 1,
			final_epsilon = 0.1, final_exploration_frame = 1000000,
			replay_start_size = 50000, discount_factor = 0.99,
			learning_rate = 0.025, gradient_momentum = 0.95,
			network_update_frequency = 1e4, disp_step = 100,
			test_cycles = 500, render_train = False, render_test = False):
		# need to be defined to run the network
		self.env = env
		self.output_size = output_size
		self.sess = sess
		self.num_episodes = num_episodes
		self.num_epochs = num_epochs

		# values pre-defined
		self.mem_size = mem_size
		self.batch_size = batch_size
		self.intial_epsilon = intial_epsilon
		self.final_epsilon = final_epsilon
		self.final_exploration_frame = final_exploration_frame
		self.replay_start_size = replay_start_size
		self.discount_factor = discount_factor
		self.learning_rate = learning_rate
		self.gradient_momentum = gradient_momentum
		self.network_update_frequency = network_update_frequency
		self.disp_step = disp_step
		self.test_cycles = test_cycles
		self.render_train = render_train
		self.render_test = render_test

		# initialize the memory
		self.exp_memory = ExperienceReplayMemory(mem_size, batch_size)

		# making epsilon list
		el = np.linspace(intial_epsilon, final_epsilon, min(final_exploration_frame, num_epochs))
		if num_epochs > final_exploration_frame:
			el = np.hstack([el, np.ones(num_epochs - final_exploration_frame, dtype = np.float32) * final_epsilon])
		self.epsilon_list = el

		# model params
		self.input_size = env.reset().shape[0]
		self.hidden_1 = 128
		self.hidden_2 = 32

		# build the models
		self._build_network()


	def _build_network(self):
		# we are using linear feedforward network for now
		with tf.variable_scope("action_value_network"):
			# placeholders for IO to network
			self.va_input = tf.placeholder(tf.float32, [self.batch_size, self.input_size])
			self.va_target = tf.placeholder(tf.float32, [self.batch_size, self.output_size]) # y
			with tf.variable_scope("layer_1"):
				self.W1_av = tf.Variable(tf.truncated_normal([self.input_size, self.hidden_1]), name = 'W1_av')
				self.b1_av = tf.Variable(tf.truncated_normal([self.hidden_1]), name = 'b1_av')
				y1 = tf.nn.relu(tf.matmul(self.va_input, self.W1_av) + self.b1_av)

			with tf.variable_scope("layer_2"):
				self.W2_av = tf.Variable(tf.truncated_normal([self.hidden_1, self.hidden_2]), name = 'W2_av')
				self.b2_av = tf.Variable(tf.truncated_normal([self.hidden_2]), name = 'b2_av')
				y2 = tf.nn.relu(tf.matmul(y1, self.W2_av) + self.b2_av)

			with tf.variable_scope("layer_3"):
				self.W3_av = tf.Variable(tf.truncated_normal([self.hidden_2, self.output_size]), name = 'W3_av')
				self.b3_av = tf.Variable(tf.truncated_normal([self.output_size]), name = 'b3_av')
				self.va_out = tf.nn.softmax(tf.matmul(y2, self.W3_av) + self.b3_av) # Q()

		with tf.variable_scope("target_value_network"):
			# placeholders for IO to network
			self.tv_input = tf.placeholder(tf.float32, [self.batch_size, self.input_size])
			with tf.variable_scope("layer_1"):
				self.W1_tv = tf.Variable(tf.truncated_normal([self.input_size, self.hidden_1]), name = 'W1_tv')
				self.b1_tv = tf.Variable(tf.truncated_normal([self.hidden_1]), name = 'b1_tv')
				y1_tv = tf.nn.relu(tf.matmul(self.tv_input, self.W1_tv) + self.b1_tv)

			with tf.variable_scope("layer_2"):
				self.W2_tv = tf.Variable(tf.truncated_normal([self.hidden_1, self.hidden_2]), name = 'W2_tv')
				self.b2_tv = tf.Variable(tf.truncated_normal([self.hidden_2]), name = 'b2_tv')
				y2_tv = tf.nn.relu(tf.matmul(y1_tv, self.W2_tv) + self.b2_tv)

			with tf.variable_scope("layer_3"):
				self.W3_tv = tf.Variable(tf.truncated_normal([self.hidden_2, self.output_size]), name = 'W3_tv')
				self.b3_tv = tf.Variable(tf.truncated_normal([self.output_size]), name = 'b3_tv')
				self.tv_out = tf.nn.softmax(tf.matmul(y2_tv, self.W3_tv) + self.b3_tv)

		# making the training function
		self.loss = tf.reduce_mean(tf.pow(self.va_target - self.va_out, 2))
		# though we can use the input for further arguments, we are not going to use it
		self.train_step = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate,
						momentum = self.gradient_momentum).minimize(self.loss)

	def update_networks(self):
		"""
		Update the network, last step in DQN Algorithm
		"""
		# layer 1 update
		self.W1_tv = tf.assign(self.W1_tv, self.W1_av)
		self.b1_tv = tf.assign(self.b1_tv, self.b1_av)

		# layer 2 update
		self.W2_tv = tf.assign(self.W2_tv, self.W2_av)
		self.b2_tv = tf.assign(self.b2_tv, self.b2_av)

		# layer 3 update
		self.W3_tv = tf.assign(self.W3_tv, self.W3_av)
		self.b3_tv = tf.assign(self.b3_tv, self.b3_av)

	def train(self):
		# initialize the model
		self.sess.run(tf.global_variables_initializer())

		for episode in range(self.num_episodes):
			# reset the environment for every 
			obs = self.env.reset()

			# first replay_start_size is just going to be to gather experience
			for _ in range(self.replay_start_size):
				# run one step to see if not done
				action = self.env.action_space.sample()
				obs_new, reward, done, _ = self.env.step(action) 
				self.exp_memory.remember((obs, action, reward, obs_new), done)
				obs = obs_new

				while not done:
					# repeat while the game is not over
					action = self.env.action_space.sample()
					obs_new, reward, done, _ = self.env.step(action) 
					self.exp_memory.remember((obs, action, reward, obs_new), done)
					obs = obs_new

			print("Now running episode number {0}...".format(episode))
			for epoch in range(self.num_epochs):
				# if we decide to render the game
				if self.render_train:
					self.env.render()

				# epsilon greedy policy
				if np.random.random() < self.epsilon_list[epoch]:
					action = self.env.action_space.sample()
				else:
					feed_dict_epsilon_greedy = {self.va_input: obs}
					va_out = self.sess.run(self.va_out, feed_dict = feed_dict_epsilon_greedy)
					action = self.sess.run(tf.argmax(va_out))

				# execute action in emulator and get new values
				obs_new, reward, done, _ = self.env.step(action)
				self.exp_memory.remember((obs, action, reward, obs_new), done)
				obs = obs_new

				# sample random minibatch from memory
				batch = self.exp_memory.get_batch()
				obs_mb, action_mb, reward_mb, obs_new_mb, done_mb = [],[],[],[], []
				for i,b in enumerate(batch):
					obs_mb.append(b[0][0])
					action_mb.append(b[0][1])
					reward_mb.append(b[0][2])
					obs_new_mb.append(b[0][3])
					done_mb.append(b[1])
				y = np.zeros([self.batch_size, 3])

				obs_mb = np.array(obs_mb)
				action_mb = np.array(action_mb)
				reward_mb = np.array(reward_mb)
				obs_new_mb = np.array(obs_new_mb)
				done_mb = np.array(done_mb)

				"""
				print("obs_mb.shape:", obs_mb.shape)
				print("action_mb.shape:", action_mb.shape)
				print("reward_mb.shape:",reward_mb.shape)
				print("obs_new_mb.shape:", obs_new_mb.shape)
				print("done_mb.shape:", done_mb.shape)
				"""

				# filling y
				for i,d in enumerate(done_mb):
					if d:
						y[i][action_mb[i]] = reward_mb[i]
					else:
						feed_dict_target_val = {self.tv_input:obs_new_mb}
						target_val = self.sess.run(self.tv_out, feed_dict = feed_dict_target_val)[i]
						target_val *= self.discount_factor
						corr_ind = self.sess.run(tf.argmax(target_val))
						# print(target_val)
						y[i] = target_val
						y[i][corr_ind] += reward_mb[i]

				# perform gradient descent
				feed_dict_train_step = {self.va_input: obs_mb, self.va_target: y}
				l, _ = self.sess.run([self.loss, self.train_step], feed_dict = feed_dict_train_step)

				# update the network
				if epoch % self.network_update_frequency == 0:
					self.update_networks()

				# display the results
				if (epoch) % self.disp_step == 0:
					print("Epoch {0}, loss {1}".format(epoch, l))


	def test(self):
		"""
		Function to test the model
		"""
		test_rewards = []
		obs = self.env.reset()
		done_test = False
		for t in range(self.test_cycles):
			# to render or not to render
			if self.render_test:
				env.render()

			cycle_rewards = 0
			while not done_test:
				feed_dict_test = {self.va_input: obs}
				action_test = self.sess.run(self.va_out, feed_dict = feed_dict_test)
				action_test = self.sess.run(tf.argmax(action_test))
				obs_test, r_test, done_test,_ = env.step(action_test)
				cycle_rewards += r_test

			test_rewards.append(cycle_rewards)

		return test_rewards


# main function 
def main():
	mem_size = 50000
	num_episodes = 100
	num_epochs = 10000
	disp_step = 1000
	env = gym.make("MountainCar-v0")

	sess = tf.Session()

	dqn = DQN(env = env, output_size = 3, sess = sess, num_episodes = num_episodes,
		num_epochs = num_epochs, disp_step = disp_step)

	dqn.train()

if __name__ == '__main__':
	main()