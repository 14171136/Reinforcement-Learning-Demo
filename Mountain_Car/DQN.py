# -*- coding: utf-8 -*-
# @Time    : 2020/8/16 17:18
# @Author  : Syao
# @FileName: DQN.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(2020)
tf.set_random_seed(2020)

class DeepQNetwork:
    def __init__(self,n_actions,n_features,lr,replace_target_iter,e_greedy,
                 gamma,memory_size,batchsize=32,
                 build_graph=True,e_greedy_increment=None,):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.gamma = gamma
        self.epsilon_max = e_greedy
        self.memory_size = memory_size
        self.batchsize = batchsize
        self.replace_target_iter = replace_target_iter
        self.learning_step = 0
        self.memory = np.zeros((self.memory_size,self.n_features*2+2))
        self.build_graph = build_graph
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess = tf.Session()
        self.replace_target_op = [tf.assign(t,e) for t,e in zip(t_params,e_params)]
        if self.build_graph:
            tf.summary.FileWriter('logs/',self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_hitsory = []

    def _build_net(self):
        self.s = tf.placeholder(tf.float32,[None,self.n_features],name='s')
        self.q_target = tf.placeholder(tf.float32,[None,self.n_actions],name='q_target')

        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 10
            w_init = tf.random_normal_initializer(0.,0.3)
            b_init = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_init, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_init, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_init, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_init, collections=c_names)
                self.q_eval = tf.nn.relu(tf.matmul(l1,w2)+b2)

            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
            with tf.variable_scope('train'):
                self.train = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        self.s_ = tf.placeholder(tf.float32,[None,self.n_features],name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_init, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_init, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_init, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_init, collections=c_names)
                self.q_next = tf.nn.relu(tf.matmul(l1,w2)+b2)

    def store_transition(self,s,a,r,s_):
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s,[a,r],s_))

        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1

    def choose_action(self,observation):
        observation = observation[np.newaxis,:]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval,feed_dict={self.s:observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0,self.n_actions)
        return action

    def learn(self):
        if self.learning_step % self.replace_target_iter == 0 and self.learning_step>200:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size,size=self.batchsize)
        else:
            sample_index = np.random.choice(self.memory_counter,size=self.batchsize)

        batch_memory = self.memory[sample_index,:]

        q_next,q_eval = self.sess.run([self.q_next,self.q_eval],
                                      feed_dict={
                                          self.s_:batch_memory[:,-self.n_features:],
                                          self.s:batch_memory[:,:self.n_features]
                                      })
        q_target = q_eval.copy()

        batch_index = np.arange(self.batchsize,dtype=np.int32)
        eval_act_index = batch_memory[:,self.n_features].astype(int)
        reward = batch_memory[:,self.n_features+1]
        q_target[batch_index,eval_act_index] = reward + self.gamma*np.max(q_next,axis=1)

        _,self.cost = self.sess.run([self.train,self.loss],
                                    feed_dict={self.s:batch_memory[:,:self.n_features],
                                               self.q_target:q_target})
        self.cost_hitsory.append(self.cost)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learning_step += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange((len(self.cost_hitsory))),self.cost_hitsory)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()
