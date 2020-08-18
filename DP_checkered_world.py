def NextState(state,action):
    next_state = state
    if (state%N==0 and action=='l') or ((state+1)%N==0 and action=='r') or (state<N and action=='u') or (state>N*(N-1)-1 and action=='d'):
        pass
    else:
        change = change_actions[action]
        next_state = state + change
    return next_state

def get_intime_reward(state):
    return 0 if  state in [0,N**2-1] else -1

def isTerminate(state):
    return state in [0,N**2-1]

def get_all_possible_next_state(state):
    res = []
    if isTerminate(state):
        return res
    for next in actions:
        next_state = NextState(state,next)
        res.append(next_state)
    return res

def update_StateValue(state):
    possible_next_states = get_all_possible_next_state(state)
    cur_reward = get_intime_reward(state)
    newValue = 0
    number = len(possible_next_states)
    for n in possible_next_states:
        newValue += 1/number * (cur_reward+gamma*values[n])
    return newValue

def Iteration():
    new_values = [0 for i in range(N**2)]
    for s in states:
        new_values[s] = update_StateValue(s)
    global values
    values = new_values
    return values

def printValue(v):
    for i in range(N**2):
        print('{0:>6.2f}'.format(v[i]), end=" ")
        if (i + 1) % N == 0:
            print("")
    print()

if __name__ == "__main__":
    N = 4

    states = [i for i in range(N ** 2)]
    actions = ['u', 'd', 'l', 'r']
    values = [0 for i in range(N ** 2)]

    change_actions = {'u': -4, 'd': 4, 'r': 1, 'l': -1}

    gamma = 1.0
    iterations = 160
    cur_iteration = 0

    while cur_iteration <= iterations:
        print("Iteration {0}".format(cur_iteration+1))
        Iteration()
        cur_iteration += 1
    printValue(values)
