import random
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)


def build_qtable(n_states,actions):
    return pd.DataFrame(np.zeros((n_states,len(actions))),columns=actions)

def choose_action(state,q_table):
    state_actions = q_table.iloc[state,:]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()

    return action_name

def get_env_feedback(state,action):
    if action == 'l':
        reward = 0
        if state == 0:
            next = state
        else:
            next = state - 1
    else:
        if state == STATES - 2:
            next = 'terminal'
            reward = 1
        else:
            next = state + 1
            reward = 0
    return next,reward

def update_env(state,episode,step):
    env_list = ['-']*(STATES-1)+['T']
    if state == 'terminal':
        steps_list.append(step)
        print('Episode {0}:total_steps={1}'.format(episode+1,step))
        time.sleep(2)
    else:
        env_list[state] = 'o'
        c = ''.join(env_list)
        print(c)
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_qtable(STATES,ACTIONS)
    for episode in range(MAX_EPISODES):
        step = 0
        state = 0
        is_terminate = False
        update_env(state,episode,step)
        while not is_terminate:
            action = choose_action(state,q_table)
            next_action,reward = get_env_feedback(state,action)
            q_predict = q_table.loc[state,action]
            if next_action != 'terminal':
                q2update = reward + GAMMA*q_table.loc[next_action,:].max()
            else:
                q2update = reward
                is_terminate = True
            q_table.loc[state,action] += ALPHA*(q2update-q_predict)
            state = next_action

            update_env(state,episode,step+1)
            step += 1
    return q_table

if __name__ == "__main__":
    ACTIONS = ['l', 'r']
    STATES = 6
    EPSILON = 0.9
    GAMMA = 0.9
    ALPHA = 0.1
    MAX_EPISODES = 15
    FRESH_TIME = 0.2
    steps_list = []
    q_table = rl()
    print(q_table)
    #print(steps_list)
    plt.plot(list(range(1,MAX_EPISODES+1)),steps_list,'o-','b')
    plt.xlabel('Episodes')
    plt.ylabel('Total Steps')
    #plt.xlim(0,MAX_EPISODES+1)
    plt.show()
