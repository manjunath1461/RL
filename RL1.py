import tensorflow as tf
print(tf.__version__)
import warnings
warnings.filterwarnings('ignore')
import gym
import multiprocessing
import threading
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
env = gym.make('MountainCarContinuous-v0')
state_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]
action_bound = [env.action_space.low, env.action_space.high]
num_workers = multiprocessing.cpu_count() 
num_episodes = 2000 
num_timesteps = 200 
global_net_scope = 'Global_Net'
update_global = 10
gamma = 0.90 
beta = 0.01 
log_dir = 'logs'
class ActorCritic(object):
     def __init__(self, scope, sess, globalAC=None):
        self.sess=sess
        self.actor_optimizer = tf.train.RMSPropOptimizer(0.0001, name='RMSPropA')
        self.critic_optimizer = tf.train.RMSPropOptimizer(0.001, name='RMSPropC')
        if scope == global_net_scope:
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, state_shape], 'state')
                self.actor_params, self.critic_params = self.build_network(scope)[-2:]
        else:
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, state_shape], 'state')
                self.action_dist = tf.placeholder(tf.float32, [None, action_shape], 'action')
                self.target_value = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                mean, variance, self.value, self.actor_params, self.critic_params = self.build_network(scope)
                td_error = tf.subtract(self.target_value, self.value, name='TD_error')
                with tf.name_scope('critic_loss'):
                    self.critic_loss = tf.reduce_mean(tf.square(td_error))           
                with tf.name_scope('wrap_action'):
                    mean, variance = mean * action_bound[1], variance + 1e-4
                normal_dist = tf.distributions.Normal(mean, variance)
                with tf.name_scope('actor_loss'):
                    log_prob = normal_dist.log_prob(self.action_dist)
                    entropy_pi = normal_dist.entropy()
                    self.loss = log_prob * td_error + (beta * entropy_pi)
                    self.actor_loss = tf.reduce_mean(-self.loss)
                with tf.name_scope('select_action'):
                    self.action = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), 
                                                   action_bound[0], action_bound[1])
                with tf.name_scope('local_grad'):
                    self.actor_grads = tf.gradients(self.actor_loss, self.actor_params)
                    self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)
            with tf.name_scope('sync'):
                with tf.name_scope('push'):
                    self.update_actor_params = self.actor_optimizer.apply_gradients(zip(self.actor_grads,
                                                                                        globalAC.actor_params))
                    self.update_critic_params = self.critic_optimizer.apply_gradients(zip(self.critic_grads, 
                                                                                          globalAC.critic_params))
                with tf.name_scope('pull'):
                    self.pull_actor_params = [l_p.assign(g_p) for l_p, g_p in zip(self.actor_params, 
                                                                                  globalAC.actor_params)]
                    self.pull_critic_params = [l_p.assign(g_p) for l_p, g_p in zip(self.critic_params, 
                                                                                   globalAC.critic_params)]
     def build_network(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.state, 200, tf.nn.relu, kernel_initializer=w_init, name='la')
            mean = tf.layers.dense(l_a, action_shape, tf.nn.tanh,kernel_initializer=w_init, name='mean')
            variance = tf.layers.dense(l_a, action_shape, tf.nn.softplus, kernel_initializer=w_init, name='variance')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.state, 100, tf.nn.relu, kernel_initializer=w_init, name='lc')
            value = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='value')        
        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mean, variance, value, actor_params, critic_params
     def update_global(self, feed_dict):
        self.sess.run([self.update_actor_params, self.update_critic_params], feed_dict)
     def pull_from_global(self):
        self.sess.run([self.pull_actor_params, self.pull_critic_params])
     def select_action(self, state):   
        state = state[np.newaxis, :]
        return self.sess.run(self.action, {self.state: state})[0]

class Worker(object):
    def __init__(self, name, globalAC, sess):
        self.env = gym.make('MountainCarContinuous-v0').unwrapped
        self.name = name
        self.AC = ActorCritic(name, sess, globalAC)
        self.sess=sess
    def work(self):
        global global_rewards, global_episodes
        total_step = 1
        batch_states, batch_actions, batch_rewards = [], [], []
        while not coord.should_stop() and global_episodes < num_episodes:
            state = self.env.reset()
            Return = 0
            for t in range(num_timesteps):
                if self.name == 'W_0':
                    self.env.render()
                action = self.AC.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                done = True if t == num_timesteps - 1 else False
                Return += reward
                batch_states.append(state)
                batch_actions.append(action)
                batch_rewards.append((reward+8)/8)
                if total_step % update_global == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = self.sess.run(self.AC.value, {self.AC.state: next_state[np.newaxis, :]})[0, 0]
                    batch_target_value = []
                    for reward in batch_rewards[::-1]:
                        v_s_ = reward + gamma * v_s_
                        batch_target_value.append(v_s_)
                    batch_target_value.reverse()
                    batch_states, batch_actions, batch_target_value = np.vstack(batch_states), np.vstack(batch_actions), np.vstack(batch_target_value)
                    feed_dict = {
                                 self.AC.state: batch_states,
                                 self.AC.action_dist: batch_actions,
                                 self.AC.target_value: batch_target_value,
                                 }
                    self.AC.update_global(feed_dict)
                    batch_states, batch_actions, batch_rewards = [], [], []
                    self.AC.pull_from_global()
                state = next_state
                total_step += 1
                if done:
                    if len(global_rewards) < 5:
                        global_rewards.append(Return)
                    else:
                        global_rewards.append(Return)
                        global_rewards[-1] =(np.mean(global_rewards[-5:]))
                    
                    global_episodes += 1
                    break
global_rewards = []
global_episodes = 0
sess = tf.Session()
with tf.device("/cpu:0"):
    global_agent = ActorCritic(global_net_scope,sess)
    worker_agents = []
    for i in range(num_workers):
        i_name = 'W_%i' % i
        worker_agents.append(Worker(i_name, global_agent,sess))
coord = tf.train.Coordinator()
sess.run(tf.global_variables_initializer())
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
tf.summary.FileWriter(log_dir, sess.graph)
worker_threads = []
for worker in worker_agents:
    job = lambda: worker.work()
    thread = threading.Thread(target=job)
    thread.start()
    worker_threads.append(thread)
coord.join(worker_threads)
