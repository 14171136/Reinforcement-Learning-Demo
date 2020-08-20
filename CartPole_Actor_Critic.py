# -*- coding: utf-8 -*-
# @Time    : 2020/8/19 19:48
# @Author  : Syao
# @FileName: CartPole_Actor_Critic.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
import gym

class Actor:
    def __init__(self,gamma,learning_rate,num_actions,num_states):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        self.num_states = num_states
        self.build_network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_network(self):
        w1 = tf.get_variable(shape=[self.num_states,20],dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.1),name='w1')
        b1 = tf.constant(0.1,shape=[20],name='b1')
        w2 = tf.get_variable(shape=[20,self.num_actions],dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.1),name='w2')
        b2 = tf.constant(0.1,shape=[self.num_actions],name='b2')

        self.state_inputs = tf.placeholder(tf.float32,[None,self.num_states])
        self.actions_inputs = tf.placeholder(tf.int32,[None,self.num_actions])
        self.td_error = tf.placeholder(tf.float32)

        with tf.name_scope('hidden_layer1'):
            self.l1 = tf.add(tf.matmul(self.state_inputs,w1),b1)
            self.l1 = tf.nn.relu(self.l1)

        with tf.name_scope('hidden_layer2'):
            self.l2 = tf.add(tf.matmul(self.l1,w2),b2)

        self.l2_softmax = tf.nn.softmax(self.l2)

        self.log_erro = tf.nn.softmax_cross_entropy_with_logits(labels=self.actions_inputs,
                                                                logits=self.l2)

        self.loss = tf.reduce_mean(self.log_erro * self.td_error)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.loss)

    def choose_actions(self,state):
        weights = self.sess.run(self.l2_softmax,feed_dict={self.state_inputs:state[np.newaxis,:]})
        action = np.random.choice(range(weights.shape[1]),p=weights.ravel())
        return action

    def learn(self,state,action,td_error):
        state = state[np.newaxis,:]
        action_one_hot = np.zeros(self.num_actions)
        action_one_hot[action] = 1
        action = action_one_hot[np.newaxis,:]
        self.sess.run([self.train_op],feed_dict={self.state_inputs:state,
                                                 self.actions_inputs:action,
                                                 self.td_error:td_error})


class Critic:
    def __init__(self,gamma,learning_rate,num_actions,num_states):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        self.num_states = num_states
        self.build_netwoprk()
        self.train_method()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def build_netwoprk(self):
        w1q = tf.get_variable(name='w1q',shape=[self.num_states,20],initializer=tf.random_normal_initializer(0,0.1))
        b1q = tf.constant(0.1,shape=[20])
        w2q = tf.get_variable(name='w2q',shape=[20,1],initializer=tf.random_normal_initializer(0,0.1))
        b2q = tf.constant(0.1,shape=[1])

        self.input_states = tf.placeholder(tf.float32,[None,self.num_states])
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(self.input_states,w1q),b1q))
        self.out = tf.add(tf.matmul(hidden_layer,w2q),b2q)

    def train_method(self):
        self.next_value = tf.placeholder(tf.float32,[1,1])
        self.reward = tf.placeholder(tf.float32)

        with tf.name_scope('squared_TD_error'):
            self.td_error = self.reward + self.gamma*self.next_value - self.out
            self.loss = tf.square(self.td_error)
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def learn(self,state,reward,next_state):
        state,next_state = state[np.newaxis,:],next_state[np.newaxis,:]
        next_value = self.sess.run(self.out,feed_dict={self.input_states:state})
        loss,_ = self.sess.run([self.loss,self.train_op],feed_dict={self.input_states:state,
                                                                    self.next_value:next_value,
                                                                    self.reward:reward})
        return loss

def main():
    import time
    ENV_NAME= 'CartPole-v0'
    EPISODES = 3000
    STEP = 3000
    TEST = 10

    env = gym.make(ENV_NAME)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n


    actor = Actor(learning_rate=0.01,gamma=0.9,num_actions=num_actions,num_states=num_states)
    critic = Critic(gamma=0.9,learning_rate=0.01,num_actions=num_actions,num_states=num_states)

    for e in range(EPISODES):
        state = env.reset()
        for step in range(STEP):
            action = actor.choose_actions(state)
            next_state,reward,done,_ = env.step(action)

            td_error = critic.learn(state,reward,next_state)
            actor.learn(state,action,td_error)
            state = next_state

            if done:
                break
        if e % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = actor.choose_actions(state)
                    next_state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('Episode {0}: Evaluation Average Reward={1:.4f}'.format(e+1,ave_reward))


if __name__ == "__main__":
    main()
