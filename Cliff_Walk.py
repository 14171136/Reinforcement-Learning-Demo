# -*- coding: utf-8 -*-
# @Time    : 2020/8/16 15:10
# @Author  : Syao
# @FileName: Cliff_Walk_QL.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(2020)

WORLD_WIDTH,WORLD_HEIGHT = 12,6

CLIFF = [[5,i] for i in range(1,10)]

ALPHA = 0.5
GAMMA = 0.5
EPSILON = 0.1
MAX_EPISODES = 500
EPISODE_OBSERVE = MAX_EPISODES-1
UP,DOWN,LEFT,RIGHT = 0,1,2,3
ACTIONS = [UP,DOWN,LEFT,RIGHT]
START = [5,0]
END = [5,11]

REWARD_OTHER = -1
REWARD_CLIFF = -100

def get_next_state(state,action):
    i,j = state
    if action == UP:
        return [max(0,i-1),j]
    elif action == DOWN:
        return [min(WORLD_HEIGHT-1, i + 1), j]
    elif action == LEFT:
        return [i, max(j-1,0)]
    elif action == RIGHT:
        return [i, min(WORLD_WIDTH-1,j+1)]
    else:
        assert 'Invalid action'

def get_reward(state):
    if state in CLIFF:
        return REWARD_CLIFF
    else:
        return  REWARD_OTHER


def init_world():
    world = [['X']*WORLD_WIDTH for i in range(WORLD_HEIGHT)]
    for x,y in CLIFF:
        world[x][y] = 'C'
    world[START[0]][START[1]] = 'O'
    world[END[0]][END[1]] = 'E'
    return world

def print_cur_world(world):
    print('-'*WORLD_WIDTH)
    res = [''.join(world[i]) for i in range(len(world))]
    for i in res:
        print(i)
    time.sleep(2)

def episode(q_values,e):
    world = init_world()
    state = START
    rewards = 0
    steps = 0
    if np.random.binomial(1,EPSILON) == 1:
        action = np.random.choice(ACTIONS)
    else:
        values = q_values[state[0],state[1],:]
        action = np.random.choice([action_ for action_,value in enumerate(values) if value == np.max(values)])
    while state != END and state not in CLIFF:
        next_state = get_next_state(state,action)
        # if next_state == state:
        #     continue
        if e == EPISODE_OBSERVE:
            world[next_state[0]][next_state[1]] = 'O'
            print_cur_world(world)
        if np.random.binomial(1, EPSILON) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values = q_values[next_state[0], next_state[1], :]
            next_action = np.random.choice([action_ for action_, value in enumerate(values) if value == np.max(values)])
        reward = get_reward(next_state)
        rewards += reward
        max_q = np.max(q_values[next_state[0],next_state[1],:])
        q_values[state[0],state[1],action] += ALPHA*(reward+GAMMA*max_q-q_values[state[0],state[1],action])
        state = next_state
        action = next_action
        steps += 1
    return steps,rewards

def Q_learning():
    q_values = np.zeros((WORLD_HEIGHT,WORLD_WIDTH,len(ACTIONS)))
    steps_list = []
    rewards_list = []
    for i in range(MAX_EPISODES):
        print('Episode No.{}'.format(i+1))
        _step,_reward = episode(q_values,i)
        steps_list.append(_step)
        rewards_list.append(_reward)
    i = 0
    ave_reward = list(np.array(rewards_list)/np.array([13]*len(rewards_list)))
    for step,reward in zip(steps_list,rewards_list):
        print('Episode {0}--->Steps:{1},Rewards:{2}'.format(i+1,step,reward))
        i += 1
    plt.figure()
    plt.plot(list(range(len(ave_reward))),ave_reward)
    plt.show()


if __name__ == '__main__':
    Q_learning()
