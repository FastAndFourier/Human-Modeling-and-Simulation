import gym
import gym_maze
from gym_maze.envs import MazeEnvSample3x3, MazeEnvSample5x5, MazeEnvSample10x10, MazeEnvSample100x100
import sys
sys.path.append("..")
import irrationality.Grid
import irrationality.boltzman_rational
import irrationality.irrational_behavior
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time



def simulate():

    env.render()
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
    streak = 0
    reach = 0

    for e in tqdm(range(MAX_EPISODE)):

        obv = env.reset()
        state = state_to_bucket(obv)

        if e%1000==0:
            print("Episode #",e,"(",reach,")")

        for k in range(MAX_STEP):


            action = select_action(state,q_table)
            new_s, reward, done, _ = env.step(action)
            new_s = state_to_bucket(new_s)
            update_q(q_table,action,state,new_s,reward,done)
            state = new_s

            if done :
                #print("Succeed without reaching max step")
                break

            if RENDER:
                env.render()

        if done and k <= MAX_STEP:
            reach += 1
            streak += 1
        else:
            streak = 0

        if streak > MIN_STREAK:
            print(MIN_STREAK,"episode under",MAX_STEP,"!!")
            break

    return  q_table



def update_q(q,a,s,s1,r,done):

    s = state_to_bucket(s)
    s1 = state_to_bucket(s1)

    if done:
        td = r - q[s+(a,)]
        #print("IN")
    else:
        td = r + DISCOUNT*np.max(q[s1+(a,)]) - q[s+(a,)]

    q[s+(a,)] += LR*td


def select_action(state,q):
    e = np.random.rand(1)[0]
    if e < EPSILON:
        action = env.action_space.sample()
    else:
        action = int(np.argmax(q[state]))
    return action


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

def boltz_rational_noisy(env,q_table,beta):
    # Tau : temperature coefficient
    # n : number of demonstrations generated from the same start

    dic_action = ['N','E','S','W']
    obv = env.reset()
    state = state_to_bucket(obv)
    a=[]

    env.render()
    done=0
    a.append([])

    while not(done):

        actions = q_table[state]
        boltz_distribution = np.exp(actions/beta)/np.sum(np.exp(actions/beta))
        noisy_behaviour = np.random.choice(dic_action,p=boltz_distribution)
        #noisy_behaviour = dic_action[noisy_behaviour]

        print(actions,boltz_distribution)
        print(dic_action[select_action(state,q_table)],noisy_behaviour)

        new_state,reward,done,_ = env.step(noisy_behaviour)

        state=state_to_bucket(new_state)
        a.append(noisy_behaviour)
        env.render()
        time.sleep(0.2)

    a = np.array(a)





if __name__ == "__main__":

    env = MazeEnvSample3x3()
    #env = gym.make("maze-random-10x10-plus-v0")

    MAX_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAX_SIZE
    NUM_ACTIONS = env.action_space.n
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))


    LR = 0.1
    EPSILON = 0.3
    MAX_EPISODE = 40000
    MAX_STEP = 50
    DISCOUNT = 0.99
    MIN_STREAK = 500
    RENDER = False

    q_table = simulate()
    print(q_table)

    boltz_rational_noisy(env,q_table,0.1)


    state = state_to_bucket(env.reset())
    EPSILON = 0
    a = []
    env.render()

    for k in range(MAX_STEP):
        action = select_action(state,q_table)
        a.append(action)
        new_s, reward, done, _ = env.step(action)
        new_s = state_to_bucket(new_s)
        state = new_s
        env.render()
        time.sleep(1)
        if done :
            break

    print(len(a),"itérations -> ",irrationality.Grid.action2str(a))
