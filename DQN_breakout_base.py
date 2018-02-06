# importing dependencies

# for environment
import gym
from gym.wrappers import Monitor

# machine learning
import tensorflow as tf

# interacting with OS/System
import os
import sys
import psutil

# misc tools
import itertools # for iterations
import random # random
import numpy as np # matrix math
import plotting # ??
from collections import deque, namedtuple # ?? 

if "../" not in sys.path:
    sys.path.append("../")
    
env = gym.make("Breakout-v0")

'''
Example of running a model, where
env.render() --> gives us a visual representation of the model
env.step(action) --> makes an action towards that step, returs a tuple of
    observation: the state after making the step
    reward: the reward that is obtained on taking that step
    done: whether the environment is over or not
    info: ??
'''
for i in range(10):
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close

# Getting the features of the model
observation_space = env.observation_space # --> Box(210, 160, 3)
action_space = env.action_space # --> Discrete(4)
# thus valid actions
VALID_ACTIONS = [0, 1, 2, 3]

# Since we are using the Breakout environment, we get an image rather than just observation values
# so we need to process the images that we are getting
class ProcessState():
    def __init__(self):
        # Build the tensorflow graph for image processing
        with tf.variable_scope("Process_State"):
            self.input_state = tf.placeholder(tf.uint8, [210, 160, 3])
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(self.output, [84, 84],
                                                 method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)
            
    def process(self, sess, state):
        '''
        Args:
            sess: tensorflow session
            state: a [210, 160, 3] RGB image
        Returns:
            a processed [84, 84, 1] greyscale image
        '''
        return sess.run(self.output, {self.input_state: state})

# The class that has the neural network
class DQN():
    '''
    This is used for both the Estimator and Target Network
    '''
    def __init__(self, scope = 'estimator', summaries_dir = None):
        self.scope = scope
        self.summaries_writer = None
        with tf.variable_scope(scope):
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, 'summaries_{}'.format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)
                
    def _build_model(self):
        '''
        Build the tensorflow graph
        '''
        # defining the placeholders
        # input placeholder
        self.X_pl = tf.placeholder(tf.uint8, [None, 84, 84, 4], name = 'X')
        # target value placeholder
        self.y_pl = tf.placeholder(tf.float32, [None], name = 'y')
        # integer of which action was taken
        self.actions_pl = tf.placeholder(tf.int32, [None], name = 'actions')
        
        # convert the input placeholder to float value
        X = tf.to_float(self.X_pl) / 255.0
        # get batch size
        batch_size = tf.shape(self.X_pl)[0]
        
        # three convolution layers
        conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn = tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn = tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn = tf.nn.relu)
        
        # fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))
        
        # get predictions for chosen actions only
        # what we are doing here is just another way to get predcitions, we could have
        # argmax
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
        
        # define loss
        self.loss = tf.reduce_mean(tf.squared_difference(self.y_pl, self.action_predictions))
        self.train_step = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.loss)
        
        # summaries for tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_mean(self.predictions))
        ])
        
    def predict(self, sess, s):
        '''
        s: state of shape [84, 84, 1]
        '''
        return sess.run(self.predictions, {self.X_pl: s})
    
    def update(self, sess, s, a, y):
        '''
        s: state of shape [84, 84, 1]
        a: chosen actions of shape [batch_size]
        y: target of shape [batch_size]
        '''
        feed_dict = {self.X_pl: s, self.y_pl: y, self.a_pl: a}
        summaries, global_step, _, loss = sess.run([self.summaries,
                                                   tf.contrib.framework.get_global_step(),
                                                   self.train_step,
                                                   self.loss],
                                                  feed_dict = feed_dict)
        if self.summaries_writer:
            self.summaries_writer.add_summary(summaries, global_step)
        return loss

# class to copy the model parameters
class ModelParametersCopy():
    def __init__(self, estimator1, estimator2):
        '''
        As per the paper we need to update the model after every few training iterations.
        Args:
            estimator1: estimator (Network) to copy values from
            estimator2: estimator (Network) to copy values to
        '''
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key = lambda v:v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key = lambda v:v.name)
        
        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)
        
    def make(self, sess):
        return sess.run(update_ops)

# epsilon-greedy policy
def make_epsilon_greedy_policy(estimator, nA):
    '''
        epsilon-greedy policy function.
        Args:
            estimator: Network
            nA: number of actions
            sess: tensorflow session
            observation: observation space
        '''
    def get_value(sess, observation, epsilon):
        A = np.ones(nA, dtype = np.float32) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += 1.0 - epsilon
        return A
    
    return get_value

def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size = 500000,
                    replay_memory_init_size = 50000,
                    update_target_estimator_every = 10000,
                    discount_factor = 0.99,
                    epsilon_start = 1.0,
                    epsilon_end = 0.0,
                    epsilon_decay_steps = 500000,
                    batch_size = 32,
                    record_video_every = 50):
    '''
    The function that will train our network to play Breakout
    Args:
        sess: tensorflow session
        env: game environment
        q_estimator: q-value network
        target_estimator: target network
        state_processor: ProcessState class which will process the states
        num_episodes: number of episodes to run for
        experiment_dir: Directory to save tensorflow summaries in
        replay_memory_size: size of the replay memory
        replay_memory_init_size: number of random experiences to sample when initializing the reply memory.
        update_target_estimator_every_: after these iterations upadate the network
        discount_factor: discount factor
        epsilon_start: starting value of the epsilon
        epsilon_end: ending value of epsilon
        epsilon_decay_steps: number of steps over which to decay the epsilon value
        batch_size: size of batch during training
        record_video_every: record video after these episodes
    Returns:
        a tuple with two numpy arrays one for episode_lengths and other for episode_rewards
    '''
    
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
    
    # replay memory
    replay_memory = []
    
    # make model copier
    estimator_copy = ModelParametersCopy(q_estimator, target_estimator)
    
    # keep statistics of the important stuff
    stats = plotting.EpisodeStats(
        episode_lengths = np.zeros(num_episodes),
        episode_rewards = np.zeros(num_episodes))
    
    # for 'system/' summaries, usefull to chec if current process looks healthy
    current_process = psutil.Process()
    
    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")
    
    # create directories if not exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)
        
    # saver 
    saver = tf.train.Saver()
    
    # get to the current time step
    total_t = sess.run(tf.contrib.framework.get_global_step())
    
    # epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    
    # policy
    policy = make_epsilon_greedy_policy(q_estimator, len(VALID_ACTIONS))
    
    # populate the replay memory with initial experience
    print("Populating replay memory...")
    
    # start new game
    state = env.reset()
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis = 2)
    
    # run the loop to gte values
    for i in range(replay_memory_init_size):
        # get action probabilities
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps - 1)])
        # select action randomly
        action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
        # perform one step
        next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
        # process the state
        next_state = state_processor.process(sess, next_state)
        # ??
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis = 2)
        # add the state to replay memory
        replay_memory.append(Transition(state, action, reward, next_state, done))
        # if done reset
        if done:
            state = env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis = 2)
        else:
            state = next_state
            
    # record videos
    env = Monitor(env, directory = monitor_path,
                  video_callable = lambda count: count % record_video_every == 0,
                  resume = True)
    
    # run the model for num_episodes
    for i_episode in range(num_episodes):
        
        # save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)
        
        # reset the environment
        state = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis = 2)
        loss = None
        
        # one step in the environment
        for t in itertools.count():
            
            # epsilon at the current time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            
            # update the target network if required
            if total_t % update_target_estimator_every == 0:
                estimator_copy.make(sess)
                print("[!]Copying model parameters...\n")
                
            # print out which step we're on, helps in debugging
            print('[!]Step {} ({}) @ Episode {}/{}, loss:'.format(
                t, total_t, i_episode + 1, num_episodes, loss), end = " ")
            sys.stdout.flush()
            
            # take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            # process the next state
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis = 2)
            
            # if replay memory is full delete the first element
            if len(replay_memory) == replay_memory_size:
                del replay_memory[0]
                
            # Save the transition to replay memory
            replay_memory.append(Transition(state, actio, reward, next_state, done))
            
            # update the statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # sample a minibatch from replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
            
            # calculate q_values and targets
            q_values_next = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.argmax(q_values_next, axis = 1)
            
            # perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)
            
            # if done
            if done:
                break
                
            # else update the state and go on
            state = next_state
            total_t += 1
        
        # add summaries to the tensorboard
        episode_summary = tf.Summary()
        
        # add values
        episode_summary.value.add(simple_value = epsilon, tag = 'episode/epsilon')
        episode_summary.value.add(simple_value = stats.episode_rewards[i_episode], tag = 'episode/reward')
        episode_summary.value.add(simple_value = state.episode_lengths[i_episode], tag = 'episode/length')
        episode_summary.value.add(simple_value = current_process.cpu_percent(), tag = 'system/cpu_usage_percent')
        episode_summary.value.add(simple_value = current_process.memory_percent(memtype ='vms'), tag = 'system/v_memory_usage_percent')
        q_estimator.summary_writer.add_summary(episode_summary, i_episode)
        q_estimator.summary_write.flush()
        
        # yield
        yield total_t, plotting.EpisodeStats(
            episode_lengths = stats.episode_lengths[:i_episode + 1],
            episode_rewards = stats.episode_rewards[:i_episode + 1])
        
    return stats

# reset to default
tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

# create a global step variable
global_step = tf.Variable(0, name = 'global_step', trainable = False)

# create estimators (networks)
q_estimator = DQN(scope = 'q_estimator', summaries_dir = experiment_dir)
target_estimator = DQN(scope = 'target_q')

# state processor
state_processor = ProcessState()

# RUN THE MODEL
# Run it!
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t, stats in deep_q_learning(sess,
                                    env,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    state_processor=state_processor,
                                    experiment_dir=experiment_dir,
                                    num_episodes=10000,
                                    replay_memory_size=500000,
                                    replay_memory_init_size=50000,
                                    update_target_estimator_every=10000,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=500000,
                                    discount_factor=0.99,
                                    batch_size=32):

        print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))
