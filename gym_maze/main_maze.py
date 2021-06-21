import gym
import gym_maze
from gym_maze.envs import MazeEnvSample5x5, MazeEnvSample10x10, MazeEnvSample100x100
import sys
sys.path.append("..")
import irrationality.Grid
import irrationality.boltzman_rational
import irrationality.irrational_behavior
import numpy as np



def simulate():

    obv = env.reset()
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    #for e in range(MAX_EPISODE):


def select_action(state):
    e = np.random.rand(1)[0]
    if e < EPSILON:
        action = env.action_space.sample()
    else:
        action = int(np.argmax(qtable[state]))
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
            print(offset,scaling)
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

if __name__ == "__main__":

    env = MazeEnvSample10x10()

    #gym.make("maze-random-10x10-plus-v0")

    MAX_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAX_SIZE
    NUM_ACTIONS = env.action_space.n
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    print(STATE_BOUNDS)

    LR = 0.1
    EPSILON = 0.1
    MAX_EPISODE = 5000
    MAX_STEP = 200

    simulate()

    print(state_to_bucket([1,1]))

    try:
        while True:
            env.render()
    except KeyboardInterrupt:
        pass
