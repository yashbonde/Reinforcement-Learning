from keras.models import Sequential
from keras.layers import Dense, Flatten
from six.moves import xrange
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import gym
env = gym.make('CartPole-v0')

print('[*]Observation Space(High):',env.observation_space.high)
print('[*]Action Space Size:',env.action_space.n) # --> Discrete means that there are only 2 possible values

# Declaring some functions
def make_observations(env, D, observe_time, epsilon):
	observation = env.reset() # taking the first observation
	obs = np.expand_dims(observation, axis = 0)
	state = np.stack((obs, obs), axis = 1)
	done = False
	for t in xrange(observe_time):
		if np.random.rand() <= epsilon:
			# If any random value for the event is less than epsilon, make a random movement
			action = np.random.randint(0, env.action_space.n, size = [1])[0]
		else:
			# otherwise make the model predict the Q-value which will be used to predict direction of movement
			Q = model.predict(state) # passing through the neural network
			action = np.argmax(Q) # maximum value will be chosen as the action
		# Now that we have collected (s(t), a(t)) we can use it to get new (s(t+1), r(t)) pair
		# then we can use the (s(t+1), r(t)) to predict a(t+1)
		observation_new, reward, done, info = env.step(action) # get the new (s, r, d, i) pair
		obs_new = np.expand_dims(observation_new, axis = 0)
		# Bascially overlay the older state with the newer state
		state_new = np.append(np.expand_dims(obs_new, axis = 0), state[:, :1, :], axis = 1)
		D.append((state, action, reward, state_new, done)) # Remeber everything
		state = state_new # update the state
		if done:
			env.reset()
			obs = np.expand_dims(observation, axis = 0)
			state = np.stack((obs, obs), axis = 1)
	return D, state

'''
def example_learning(model, minibatch):
	state = minibatch[0]
	action = minibatch[1]
	reward = minibatch[2]
	state_new = minibatch[3]
	done = minibatch[4]
	# Bellman equation for Q-function
	inputs = np.expand_dims(state, axis = 0)
	targets = model.predict(state)
	Q_sa = model.predict(state_new)
	if done:
		targets = reward
	else:
		targets[action] = reward + gamma*np.max(Q_sa)
	# Train network to output Q-function
	model.fit(inputs, targets)
	return model
'''

def batch_learning(model, env, D, state, mb_size):
	minibatch = random.sample(D, mb_size)
	inputs = np.zeros((mb_size, ) + state.shape[1:])
	targets = np.zeros((mb_size, env.action_space.n))
	for i in xrange(0, mb_size):
		state = minibatch[i][0]
		action = minibatch[i][1]
		reward = minibatch[i][2]
		state_new = minibatch[i][3]
		done = minibatch[i][4]
		# Bellman equation for Q-function
		inputs[i:i+1] = np.expand_dims(state, axis = 0)
		targets[i] = model.predict(state)
		Q_sa = model.predict(state_new)
		if done:
			targets[i,action] = reward
		else:
			targets[i, action] = reward + gamma*np.max(Q_sa)
		# Train network to output Q-function
		model.train_on_batch(inputs, targets)
	return model

# STEP ONE: MAKE THE MODEL
model = Sequential()
model.add(Dense(30, input_shape = (2, ) + env.observation_space.shape, activation = 'relu'))
model.add(Flatten())
model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(env.action_space.n, activation = 'softmax'))
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
print('[*]Model Compliled')

# STEP TWO: TRAINING NETWORK
# Params
D = deque() # we use this as is later helps in making the minibatches easily
observe_time = 50000 # Number of time steps we will be acting on the game and observing the results
epsilon = 0.7 # probability of doing a random movement
gamma = 0.9 # Discounted future reward, how much we care about the steps in the future
mb_size = 50 # size of minibatch

# Observations
D, state = make_observations(env, D, observe_time, epsilon)
print('[*]Observation Finished')
# learning
model = batch_learning(model, env, D, state, mb_size)
print('[*]Learning Complete')

# STEP THREE: PLAYING THE GAME
num_games = 1000
rewards = []
# Playing the game
count = 0
batch_run = deque()
for game_n in xrange(num_games):
	observation = env.reset()
	obs = np.expand_dims(observation, axis = 0)
	state = np.stack((obs, obs), axis = 1)
	done = False
	tot_reward = 0.0
	while not done:
		count += 1
		if np.random.rand() <= epsilon:
			action = np.random.randint(0, env.action_space.n, size = [1])[0]
		else:
			# use neural network to determine the action
			Q = model.predict(state)
			action = np.argmax(Q)
		observation_new, reward, done, info = env.step(action)
		obs_new = np.expand_dims(observation_new, axis = 0)
		state_new = np.append(np.expand_dims(obs_new, axis = 0), state[:, :1, :], axis = 1)
		batch_run.append((state, action, reward, state_new, done)) # new example
		if count == mb_size:
			model = batch_learning(model, env, batch_run, state, mb_size)
		tot_reward += reward
		state = state_new # update the state
	rewards.append(tot_reward)
	if game_n % 100 == 0:
		print('[*]Game Ended! Total Reward:', tot_reward)

print('[*]Maximum Reward Achieved:', max(rewards))
'''
x_ = [i for i in range(num_games)]
plt.plot(x_, rewards, '-')
plt.show()
'''