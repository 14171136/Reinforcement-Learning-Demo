# -*- coding: utf-8 -*-
# @Time    : 2020/8/17 21:23
# @Author  : Syao
# @FileName: CartPole.py
# @Software: PyCharm

import numpy as np
import gym
import tensorflow as tf
import random
from collections import deque

GAMMA = 0.5
LEARNING_RATE = 0.01

class Policy_Gradient:
    def __init__(self,env):
        self.time_step = 0
        self.state_num = env.observation_space.shape[0]
        self.action_num = env.action_space.n
        self.ep_obs,self.ep_as,self.ep_rs = [],[],[]
        self.creat_softmax_network()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def creat_softmax_network(self):
        w1 = tf.get_variable(shape=[self.state_num,20],dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.1),name='w1')
        b1 = tf.constant(0.01,shape=[20])
        w2 = tf.get_variable(shape=[20, self.action_num], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.1),name='w2')
        b2 = tf.constant(0.01,shape=[self.action_num])

        self.input_states = tf.placeholder(tf.float32,[None,self.state_num])
        self.actions = tf.placeholder(tf.int32,[None,])
        self.val_actions = tf.placeholder(tf.float32,[None,])

        hidden_layer = tf.nn.relu(tf.add((tf.matmul(self.input_states,w1)),b1))
        self.out = tf.add(tf.matmul(hidden_layer,w2),b2)

        self.out_prob = tf.nn.softmax(self.out)
        self.log_actions = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out,
                                                                          labels=self.actions)

        self.loss = tf.reduce_mean(self.log_actions*self.val_actions)

        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)


    def choose_action(self,observation):
        prob_weights = self.sess.run(self.out_prob, feed_dict={self.input_states: observation[np.newaxis, :]})
        action = np.random.choice(list(range(prob_weights.shape[1])), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self,s,a,r):
        self.ep_as.append(a)
        self.ep_obs.append(s)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running = 0
        for t in reversed(range(0,len(self.ep_rs))):
            running += running*GAMMA+self.ep_rs[t]
            discounted_ep_rs[t] = running

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        self.sess.run([self.train_op],feed_dict={
            self.input_states:np.vstack(self.ep_obs),
            self.actions:np.array(self.ep_as),
            self.val_actions:discounted_ep_rs
        })

        self.ep_obs,self.ep_as,self.ep_rs = [],[],[]

ENV_NAME = 'CartPole-v0'
EPISODES = 3000
STEP = 3000
TEST = 10

def main():
    env = gym.make(ENV_NAME)
    agent = Policy_Gradient(env)

    for episode in range(EPISODES):
        state = env.reset()
        for step in range(STEP):
            action = agent.choose_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.store_transition(state,action,reward)
            state = next_state
            if done:
                agent.learn()
                break

        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.choose_action(state)
                    n_state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print('Episode:{0},Evaluation Average Reward:{1}'.format(episode+1,ave_reward))

if __name__ == "__main__":
    main()
