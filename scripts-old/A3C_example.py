# importing the dependencies
import numpy as np # matrix math
import tensorflow as tf # ML
import gym # RL environment
import os # OS interaction
import multiprocessing # parallel computation
import a3c_util as util # utility functions
from threading import Thread # for making threads
from random import choice # random choice
from time import sleep, time # time package

#make the actor crtic network
class AC_Network():
    def __init__(self, observation_space, action_space, scope, trainer):
        print('[!]New Actor Critic network starting, scope:',scope)
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, 160, 160, 3],dtype=tf.float32)
            self.conv1 = tf.contrib.layers.conv2d(activation_fn=tf.nn.elu,
                inputs=self.inputs, num_outputs=16,
                kernel_size=[8,8], stride=[4,4],
                padding='VALID')
            self.conv2 = tf.contrib.layers.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv1,
                num_outputs=32,
                kernel_size=[4,4],
                stride=[2,2],
                padding='VALID')
            hidden = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.conv2),
                256, activation_fn=tf.nn.elu)
            
            #Recurrent network for temporal dependencies
            # define the LSTM cell
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)
            # initial conditions
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            # placeholders for initial conditions
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.inputs)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            # one operation
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])
            
            #Output layers for policy and value estimations
            self.policy = tf.contrib.layers.fully_connected(rnn_out,action_space,
                activation_fn=tf.nn.softmax)
            self.value = tf.contrib.layers.fully_connected(rnn_out,1,
                activation_fn=None)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                # we calcute the GAE and all outside and push it here for processing
                # placeholder for actions
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                # convert the actions to one hot
                self.actions_onehot = tf.one_hot(self.actions,action_space,dtype=tf.float32)
                # placeholder for target values
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                # placeholder for advantages
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                # main outputs that will be used
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

class Worker(object):
    def __init__(self, game, name, trainer, model_path, global_episodes):
        print('[!]New Worker starting, name:', name)
        self.name = "worker_" + str(name) # name of the worker
        self.number = name # number of the worker
        self.model_path = model_path # path of the model file
        self.trainer = trainer # optimizer
        self.global_episodes = global_episodes # global episodes
        self.increment = self.global_episodes.assign_add(1) # when one iteration s
        self.episode_rewards = [] # list of rewards that we get in the episode
        self.episode_lengths = [] # length of the episodes we are running
        self.episode_mean_values = [] # mean reward values for the episodes
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number)) # summary

        # parameters for the game
        input_dim = game.observation_space.shape
        action_space = game.action_space.n

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(input_dim, action_space, self.name,trainer)
        self.update_local_ops = util.update_target_graph('global',self.name)        
        
        # actions will be 
        self.actions = np.identity(action_space, dtype=bool).tolist()
        # convert to internal environment
        self.env = game

        # flags
        self.initial_boot = True
        
    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = np.stack(rollout[:,0], axis = 0)
        actions = rollout[:,1]
        rewards = rollout[:,2]
        # next_observations = rollout[:,3]
        values = rollout[:,5]
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = util.discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = util.discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to be periodically saved
        if self.initial_boot:
            print('[!]observations_shape:', observations.shape)
            print('[!]observations[0]_shape:', observations[0].shape)
            print('[!]observations[1]_shape:', observations[1].shape)
            print('[!]observations_type:', type(observations))
            print('[!]observations_0_type:', type(observations[0]))
            self.initial_boot = False

        # make the fee dictionary
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:observations,
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:self.batch_rnn_state[0],
            self.local_AC.state_in[1]:self.batch_rnn_state[1]}

        # get the losses and norms
        value_loss, policy_loss, entropy_loss, grad_norms, var_norms, self.batch_rnn_state, _ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss, self.local_AC.entropy, self.local_AC.grad_norms,
            self.local_AC.var_norms, self.local_AC.state_out, self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return value_loss/len(rollout), policy_loss/len(rollout), entropy_loss/len(rollout), grad_norms, var_norms
        
    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                done = False
                
                observation = self.env.reset()
                episode_frames.append(observation)
                observation = util.process_frame(observation)
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                while not done:
                    # a_dist: distribution of actions
                    # v: value
                    # Take an action using probabilities from policy network output.
                    a_dist, value, rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                        feed_dict={self.local_AC.inputs:[observation],
                        self.local_AC.state_in[0]:rnn_state[0],
                        self.local_AC.state_in[1]:rnn_state[1]})
                    action = np.random.choice(a_dist[0], p = a_dist[0])
                    action = np.argmax(a_dist == action)

                    # take a step and receive value
                    observation_new, reward, done, _ = self.env.step(action) 

                    if not done:
                        episode_frames.append(observation_new)
                        observation_new = util.process_frame(observation_new)

                    else:
                        observation_new = observation
                        
                    episode_buffer.append([observation, action, reward, observation_new, done, value[0,0]])
                    episode_values.append(value[0,0])

                    episode_reward += reward
                    observation = observation_new                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and not done and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.inputs:[observation],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                        value_loss, policy_loss, entropy_loss, grad_norms, var_norms = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if done:
                        break
                                            
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    value_loss, policy_loss, entropy_loss,grad_norms, var_norms = self.train(episode_buffer, sess, gamma, 0.0)
                                
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(images,'./frames/image'+str(episode_count)+'.gif',
                            duration=len(images)*time_per_step,true_image=True,salience=False)
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(value_loss))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(policy_loss) )
                    summary.value.add(tag='Losses/Entropy', simple_value=float(entropy_loss))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(grad_norms) )
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(var_norms))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1


def main():
    print('[!]Starting the Shit!')
    max_episode_length = 10000
    gamma = 0.99
    load_model = False
    model_path = './model'
    frame_path = './frames'

    tf.reset_default_graph()

    # save the model
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)

    # define the game
    print('[!]Making the game')
    game_name = 'Breakout-v0'
    game = gym.make(game_name)
    input_dim = game.observation_space.shape
    action_space = game.action_space.n
    print('[!]Input_dim:',input_dim)
    print('[!]Action Space:',action_space)

    # trainable stuff
    with tf.device('/cpu:0'):
        print('[!]Inside TF CPU with')
        global_episodes = tf.Variable(0, dtype = tf.int32, name = 'global_episodes', trainable = False)
        # we don't need trainer for 'global'
        master_network = AC_Network(observation_space = input_dim, action_space = action_space, scope = 'global', trainer = None)
        trainer = tf.train.AdamOptimizer()
        num_workers = multiprocessing.cpu_count()
        workers = []
        # create worker class
        for i in range(num_workers):
            workers.append(Worker(game = game, name = i, trainer = trainer, model_path = model_path, global_episodes = global_episodes))
        saver = tf.train.Saver(max_to_keep = 5)
    
    with tf.Session() as sess:
        print('[!]inside the sess')
        coord = tf.train.Coordinator()
        if load_model:
            print('Loading Model ...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        
        # this is the asynchronous stuff
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
            t = Thread(target = (worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)

if __name__ == '__main__':
    main()
