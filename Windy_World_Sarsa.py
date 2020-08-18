import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

WORLD_HEIGHT,WORLD_WIDTH = 7,10

WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0] #WIND:UP

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

EPSILON = 0.1
ALPHA = 0.5
REWARD = -1.0

START = [3,0]
END = [3,7]
ACTIONS = [ACTION_UP,ACTION_DOWN,ACTION_LEFT,ACTION_RIGHT]

def Step(state,action):
    i,j = state
    if action == ACTION_UP:
        return [max(i-1-WIND[j],0),j]
    elif action == ACTION_DOWN:
        return [max(min(i+1-WIND[j], WORLD_HEIGHT-1),0), j]
    elif action == ACTION_LEFT:
        return [max(i-WIND[j],0),max(j-1,0)]
    elif action == ACTION_RIGHT:
        return [max(i-WIND[j],0),min(j+1,WORLD_WIDTH-1)]
    else:
        assert False

def episode(q_value):
    step = 0
    state = START
    if np.random.binomial(1,EPSILON) == 1:
        action = np.random.choice(ACTIONS)
    else:
        v = q_value[state[0],state[1],:]
        action = np.random.choice([action_ for action_,value in enumerate(v) if value==np.max(v)])

    while state != END:
        next_state = Step(state,action)
        if np.random.binomial(1,EPSILON) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            v = q_value[next_state[0],next_state[1],:]
            next_action = np.random.choice([action_ for action_,value_ in enumerate(v) if value_== np.max(v)])

        q_value[state[0],state[1],action] += ALPHA*(REWARD+q_value[next_state[0],next_state[1],next_action]-q_value[state[0],state[1],action])

        state = next_state
        action = next_action
        step += 1
    return step

def Sarsa():
    q_values = np.zeros((WORLD_HEIGHT,WORLD_WIDTH,len(ACTIONS)))
    steps = []
    for _ in tqdm(range(MAX_EPISODES)):
        steps.append(episode(q_values))

    steps = np.add.accumulate(steps)
    plt.plot(steps,np.arange(1,len(steps)+1))
    plt.xlabel('Time Steps')
    plt.ylabel('Episodes')
    plt.show()

    policy = []
    for i in range(WORLD_HEIGHT):
        policy.append([])
        for j in range(WORLD_WIDTH):
            if [i,j] == END:
                policy[-1].append('End')
                continue
            best_action = np.argmax(q_values[i,j,:])
            if best_action == ACTION_UP:
                policy[-1].append('UP')
            elif best_action == ACTION_DOWN:
                policy[-1].append('DOWN')
            elif best_action == ACTION_LEFT:
                policy[-1].append('LEFT')
            elif best_action == ACTION_RIGHT:
                policy[-1].append('RIGHT')
    print('Optimal Policy is:')
    for i in policy:
        print(i)
    print('Wind Strength for each coloum:\n{}'.format([str(w) for w in WIND]))

if __name__ == "__main__":
    MAX_EPISODES = 5000
    Sarsa()
