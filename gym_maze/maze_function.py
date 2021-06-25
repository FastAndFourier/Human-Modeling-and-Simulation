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
MAX_EPISODE = 50000
MAX_STEP = 200
DISCOUNT = 1 #0.99
MIN_STREAK = MAX_EPISODE
RENDER = True
SIMULATE = False

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



    return np.transpose(reward_tab,axes=[1,0,2])


# def get_new_state(s,a):
#
#     if a == 'N' and s[0]!=0:

def select_action_from_v(env,state,v):

    reward_table = get_reward(env,state)
    for a in range(4):
        r = reward_table[tuple(state)+a]
        new_s,r,_,_ = env.step(a)

        #print(r,v[state_to_bucket(new_s)])
        v_choice.append(r + DISCOUNT*v[state_to_bucket(new_s)])


    if not(OP_VI):
        action = np.argmax(v_choice)


############################ Q-LEARNING ########################################

def simulate():

    env.render()
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
    streak = 0
    reach = 0

    for e in tqdm(range(MAX_EPISODE)):

        obv = env.reset()
        state = state_to_bucket(obv)

        if e%1000==0:
            print(get_epsilon(e))
            print("Episode #",e,"(",reach,")")

        for k in range(MAX_STEP):

            epsi = get_epsilon(e)
            action = select_action(state,q_table,epsi)
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

    #v_vector = np.zeros((NUM_BUCKETS[0],NUM_BUCKETS[1]))
    v_vector = np.random.rand(NUM_BUCKETS[0],NUM_BUCKETS[1])

    end = env.observation_space.high
    v_vector[state_to_bucket(end)] = 1


    _=env.reset()
    env.render()
    theta=0.2
    err=2

    while err>theta:

        v_temp = np.copy(v_vector)
        err=0

        for i in range(NUM_BUCKETS[0]):
            for j in range(NUM_BUCKETS[1]):

                obv = np.array([i,j],dtype=int)
                s = state_to_bucket(obv)

                v = v_vector[s]
                x = []
                for a in ['N','S','E','W']:
                    env.env.reset(s)
                    new_s,reward,_,_ = env.step(a)
                    #print(s,a,state_to_bucket(new_s))
                    x.append(reward + DISCOUNT*v_temp[state_to_bucket(new_s)])

                #print(x)
                x = np.array(x)
                b = np.max(x)
                v_vector[s] = np.sum(x*np.exp((x-b)*beta))/np.sum(np.exp((x-b)*beta))
                err = max(err,abs(v_vector[s]-v))


        print(err)
    print("VI Boltz done")

    return v_vector

def generate_traj_v(env,v):

    done=False
    obv = env.reset()
    s = state_to_bucket(obv)
    it=0
    action_ = []

    env.render()


    dic_action = ['N','S','E','W']
    while not(done) and it<1000:
        print("Current state",env.state)
        v_choice = []
        for a in dic_action:
            env.env.reset(s)
            new_s,r,_,_ = env.step(a)

            #print(r,v[state_to_bucket(new_s)])
            v_choice.append(r + DISCOUNT*v[state_to_bucket(new_s)])


        if not(OP_VI):
            action = np.argmax(v_choice)
        else:
            v_choice = np.array(v_choice)
            b = max(v_choice)
            distrib = np.exp((v_choice-b)/1e-1)/np.sum(np.exp((v_choice-b)/1e-1))
            action = np.random.choice([0,1,2,3],p=distrib)

        env.env.reset(s)
        new_s,reward,done,_ = env.step(dic_action[action])
        it+=1
        action_.append(action)
        obv = new_s
        s = state_to_bucket(obv)


        if RENDER:
            env.render()
            time.sleep(0.1)
    print("Start ",env.reset(),"->",it,"iterations",action2str(action_))



def edges_and_walls_list_extractor(env):

      edges_list = []
      walls_list = []
      maze = env.env.maze_view.maze

      maze_size = 10#MAX_SIZE[0]
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
