import gym
import gym_maze
from gym_maze.envs import MazeEnvSample3x3, MazeEnvSample5x5, MazeEnvSample10x10, MazeEnvSample100x100
import sys
sys.path.append("..")
#import irrationality.Grid
#import irrationality.boltzman_rational
#import irrationality.irrational_behavior
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


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
        # if done:
        #     print("new_state",new_state)
        #     break

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


###################### MAIN ####################################################

if __name__ == "__main__":

    #env = MazeEnvSample10x10()
    env = gym.make("maze-sample-10x10-v0")
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

    OP_VI = 1 # 0 = Argmax, 1 = Softmax

    if SIMULATE:
        q_table = simulate()
        path = "qtable2_10x10"
        np.save(path,q_table)
    else:
        path = "qtable1_10x10.npy"
        q_table = np.load(open(path,'rb'))


    v_vector = boltz_rational(env,1)
    print("Boltzmann value function :\n",v_vector)

    v_from_q = np.zeros((NUM_BUCKETS[0],NUM_BUCKETS[1]))
    for i in range(NUM_BUCKETS[0]):
        for j in range(NUM_BUCKETS[1]):
            state = state_to_bucket([i,j])
            v_from_q[i,j] = np.max(q_table[state])

    print("\nV(s) = Q(s,pi(s)) :\n",v_from_q)

    generate_traj_v(env,v_vector)
    generate_traj_v(env,v_from_q)


    # traj = []
    # beta = [1e-1,1e-3]
    # beta = [1e-4,1e-5]
    # for b in beta:
    #     print(b)
    # demo = boltz_rational_noisy(env,q_table,1e-4)
    # print(env.state)
    # s=env.env.reset()
    # print(s)
    # demo = boltz_rational_noisy(env,q_table,1e-5)

    #while True:

    # _=env.reset()
    # env.env.reset(np.array([0,0]))
    # demo = boltz_rational_noisy(env,q_table,1e-4)
    # traj.append(demo)
    #
    # print("Trajectory length",[len(t) for t in traj])
    # len_traj = [len(t) for t in traj]
    #
    # plt.hist(len_traj,density=True)
    # plt.show()
    # print("Min trajectory length",np.min([len(t) for t in traj]))
    # print("Max trajectory length",np.max([len(t) for t in traj]))
    # print("Mean",np.mean([len(t) for t in traj]))
    # print("Standard deviation",np.std([len(t) for t in traj]))

    # print("IN")
    # state = state_to_bucket(env.reset())
    # EPSILON = 0
    # a = []
    # env.render()
    #
    # for k in range(MAX_STEP):
    #     action = select_action(state,q_table,0)
    #     a.append(action)
    #     new_s, reward, done, _ = env.step(action)
    #     new_s = state_to_bucket(new_s)
    #     state = new_s
    #
    #     if RENDER:
    #         env.render()
    #         time.sleep(0.1)
    #     if done :
    #         break
    #
    # print(len(a),"itérations -> ",action2str(a))
