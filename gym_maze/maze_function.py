import gym
import gym_maze
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import time


env = gym.make("maze-sample-10x10-v0")


env.render()
#env1 = gym.make("maze-random-10x10-plus-v0")
#env = gym.make("maze2d_10x10")

MAX_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
NUM_BUCKETS = MAX_SIZE
NUM_ACTIONS = env.action_space.n
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))


LR = 0.05
EPSILON = 0.02
MAX_EPISODE = 10000
MAX_STEP = 200
DISCOUNT = 1
MIN_STREAK = MAX_EPISODE
RENDER = True
SIMULATE = False

OP_VI = 0

################# HELPER FUNCTIONS #############################################

def action2str(demo):

    #Turn action index into str
    res=[]
    for i in demo:
        if i==0:
            res.append("North")
        elif i==1:
            res.append("South")
        elif i==2:
            res.append("East")
        else :
            res.append("West")

    return res

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

def get_reward(env):

    reward_tab = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))
    #print(reward_tab[(0,0)][0])
    state = env.reset()
    state = state_to_bucket(state)

    for i in range(MAX_SIZE[0]):
        for j in range(MAX_SIZE[1]):

            state = state_to_bucket([i,j])

            for a in range(4):
                env.env.reset(state)
                new_s,r,_,_ = env.step(a)
                reward_tab[state][a] = r



    return reward_tab


def get_new_state(env):

    new_state_tab = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,),dtype=tuple)

    state = env.reset()
    state = state_to_bucket(state)

    for i in range(MAX_SIZE[0]):
        for j in range(MAX_SIZE[1]):

            state = state_to_bucket([i,j])

            for a in range(4):
                env.env.reset(state)
                new_s,r,_,_ = env.step(a)
                new_state_tab[state+(a,)] = tuple(new_s)



    return new_state_tab

def select_action_from_v(env,state,v):

    reward_table = get_reward(env)
    new_state_table = get_new_state(env)
    v_choice = []
    env.env.reset(state)


    for a in range(4):

        new_state = new_state_table[tuple(state) + (a,)]
        reward = reward_table[tuple(state)+(a,)]
        v_choice.append(reward + DISCOUNT*v[tuple(new_state)])

    action = np.argmax(v_choice)

    return action

############################ Q-LEARNING ########################################

def simulate():

    #env.render()
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
    streak = 0
    reach = 0

    for e in tqdm(range(MAX_EPISODE)):

        obv = env.env.reset(np.array([0,0]))
        env._elapsed_steps = 0

        state = state_to_bucket(obv)

        if e%1000==0:
            #print(get_epsilon(e))
            print("Episode #",e,"(",reach,")")

        for k in range(MAX_STEP):

            epsi = get_epsilon(e)
            action = select_action(state,q_table,epsi)
            new_s, reward, done, _ = env.step(action)
            new_s = state_to_bucket(new_s)
            update_q(q_table,action,state,new_s,reward,done)
            state = new_s

            if done :
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

        #time.sleep(0.1)

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


def get_epsilon(e):

    return max(EPSILON,0.1 - e*EPSILON/(MAX_EPISODE*0.60))

def select_action(state,q,e):
    e = np.random.rand(1)[0]
    epsi = get_epsilon(e)
    if e < EPSILON:
        action = env.action_space.sample()
    else:
        action = int(np.argmax(q[state]))
    return action



##################### BOLTZMANN NOISY RATIONAL #################################

def boltz_rational_noisy(env,q_table,beta):
    # Tau : temperature coefficient
    # n : number of demonstrations generated from the same start

    print(beta)
    dic_action = ['N','S','E','W']
    obv = env.reset()
    state = state_to_bucket(obv)
    print(state)
    a=[]

    env.render()
    done=0
    a.append([])

    while not(done):

        actions = q_table[state]
        b = max(actions)
        boltz_distribution = np.exp((actions-b)/beta)/np.sum(np.exp((actions-b)/beta))

        noisy_behaviour = np.random.choice(dic_action,p=boltz_distribution)

        new_state,reward,done,_ = env.step(noisy_behaviour)

        state=state_to_bucket(new_state)
        a.append(noisy_behaviour)

        if RENDER:
            env.render()
            time.sleep(0.05)

    return a


"""

    IRRATIONAL BIASES

"""

#########################  BOLTZMANN RATIONAL ##################################

def boltz_rational(env,beta):

    v_vector = np.zeros((NUM_BUCKETS[0],NUM_BUCKETS[1]))
    #v_vector = np.random.rand(NUM_BUCKETS[0],NUM_BUCKETS[1])

    end = env.observation_space.high
    v_vector[tuple(end)] = 1

    reward_table = get_reward(env)
    new_state_table = get_new_state(env)

    _=env.reset()
    env.render()
    theta=0.05
    err=2

    while err>theta:

        v_temp = np.copy(v_vector)
        err=0

        for i in range(NUM_BUCKETS[0]):
            for j in range(NUM_BUCKETS[1]):

                state = env.env.reset([i,j])

                v = v_vector[tuple(state)]
                x = []
                for a in range(4):
                    new_state = new_state_table[tuple(state) + (a,)]
                    reward = reward_table[tuple(state)+(a,)]
                    x.append(reward + DISCOUNT*v_temp[tuple(new_state)])

                #print(x)
                x = np.array(x)
                b = 0#np.max(x)
                v_vector[tuple(state)] = np.sum(x*np.exp((x-b)*beta))/np.sum(np.exp((x-b)*beta))
                #print(err,abs(v_vector[tuple(state)]-v))

                err = max(err,abs(v_vector[tuple(state)]-v))


        #print(err)
    print("VI Boltz done")

    return v_vector

def generate_traj_v(env,v):

    done=False
    obv = env.env.reset([0,0])
    s = state_to_bucket(obv)
    it=0
    action_ = []

    env.render()

    while not(done) and it<1000:

        action = select_action_from_v(env,s,v)
        new_s,reward,done,_ = env.step(int(action))

        it+=1
        action_.append(action)
        obv = new_s
        s = state_to_bucket(obv)


        if RENDER:
            env.render()
            time.sleep(0.25)
    print("Start ",env.reset(),"->",it,"iterations",action2str(action_))



def edges_and_walls_list_extractor(env):

      edges_list = []
      walls_list = []
      maze = env.env.maze_view.maze

      maze_size = MAX_SIZE[0]
      print(maze_size)
      # top line and left line
      for i in range(0,maze_size):
          walls_list.append([[0,0],[i,i+1]]) # north walls
          walls_list.append([[i,i+1],[0,0]]) # west walls

      # other matplotlib.lines
      for i in range(0,maze_size):
          for j in range(0,maze_size):
              walls_list.append([[i+1,i+1],[j,j+1]]) # south walls
              walls_list.append([[i,i+1],[j+1,j+1]]) # east walls


      for i in range(0,maze_size):
          for j in range(0,maze_size):
              maze_cell = maze.get_walls_status(maze.maze_cells[j,i])
              if maze_cell['N']==1 and [[i,i],[j,j+1]] in walls_list:
                  walls_list.remove([[i,i],[j,j+1]])
              if maze_cell['S']==1 and [[i+1,i+1],[j,j+1]] in walls_list:
                  walls_list.remove([[i+1,i+1],[j,j+1]])
              if maze_cell['E']==1 and [[i,i+1],[j+1,j+1]] in walls_list:
                  walls_list.remove([[i,i+1],[j+1,j+1]])
              if maze_cell['W']==1 and [[i,i+1],[j,j]] in walls_list:
                  walls_list.remove([[i,i+1],[j,j]])

      for i in range(0,maze_size):
          for j in range(0,maze_size):
              idx = i + j*maze_size
              if [[i,i],[j,j+1]] not in walls_list:
                  edges_list.append((idx,idx-1,1))
                  #graph.add_edge(idx,idx-1,1)
              if [[i+1,i+1],[j,j+1]] not in walls_list:
                  edges_list.append((idx,idx+1,1))
                  #graph.add_edge(idx,idx+1,1)
              if [[i,i+1],[j+1,j+1]] not in walls_list:
                  edges_list.append((idx,idx+maze_size,1))
                  #graph.add_edge(idx,idx+maze_size,1)
              if [[i,i+1],[j,j]] not in walls_list:
                  edges_list.append((idx,idx-maze_size,1))
                  #graph.add_edge(idx,idx-maze_size,1)

      return edges_list, walls_list
