import gym
import gym_maze
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import time


# LR = 0.05
# EPSILON = 0.02
MAX_EPISODE = 10000
MAX_STEP = 500
# DISCOUNT = 1
MIN_STREAK = MAX_EPISODE
RENDER = 1 #0
SIMULATE = 0
RENDER_TIME = 0.05

REWARD_T = "human" #"env", "human"
OPE = "softmax" #"argmax", "softmax"



class MyMaze():

    def __init__(self,env_name,lr=0.05,epsi=0.02,disc=0.99):

        self.env = gym.make(env_name)
        #print(self.env)
        self.env.render()

        self.maze_size = tuple((self.env.observation_space.high + np.ones(self.env.observation_space.shape)).astype(int))[0]
        self.lr = lr
        self.epsilon = epsi
        self.discount = disc

        self.reward_table = self.get_reward()
        self.new_state_table = self.get_new_state()
        self.transition_table = self.get_transition_probalities()
        self.optimal_policy = []


        # MAX_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
        # NUM_BUCKETS = MAX_SIZE
        # NUM_ACTIONS = env.action_space.n
        # STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))


        


    ################# HELPER FUNCTIONS #############################################

    def action2str(self,demo):
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
    

    def get_reward(self):

        # Outputs an array with every possible reward (for every action) on every state

        reward_tab = np.zeros((self.maze_size,self.maze_size,4))
        state = self.env.reset()
        state = tuple(state)

        for i in range(self.maze_size):
            for j in range(self.maze_size):

                state = np.array([i,j])

                for a in range(4):
                    self.env.env.reset(state)
                    new_s,r,_,_ = self.env.step(a)
                    reward_tab[tuple(state)][a] = r

        return reward_tab


    def get_new_state(self):

        # Outputs an array with every new state reached after executing action A from a state S

        new_state_tab = np.zeros((self.maze_size,self.maze_size,4),dtype=tuple)
        state = self.env.reset()
        state = tuple(state)

        for i in range(self.maze_size):
            for j in range(self.maze_size):

                state = np.array([i,j])

                for a in range(4):
                    self.env.env.reset(state)
                    new_s,r,_,_ = self.env.step(a)
                    new_state_tab[tuple(state)+(a,)] = tuple(new_s)


        return new_state_tab

    def get_transition_probalities(self):

        # Outputs transition matrix T(s,a,s')

        transition_tab = np.zeros((self.maze_size,self.maze_size,4),dtype=tuple)

        #Deterministic case

        state = self.env.reset()
        state = tuple(state)

        # for i in range(self.maze_size):
        #     for j in range(self.maze_size):



    def set_optimal_policy(self,q_table):

        # Sets optimal policy from a given Q-table

        optimal_policy = np.zeros((self.maze_size,self.maze_size),dtype=int)
        for i in range(self.maze_size):
            for j in range(self.maze_size):

                optimal_policy[i,j] = np.argmax(q_table[tuple([i,j])])


        self.optimal_policy = optimal_policy




    def select_action_from_v(self,state,v,reward_type,operator):

        # Selects action following a policy given a Value-function v

        # -> reward_type: either "env" (reward given by GymMaze = -1/num_tile for every step) 
        #                 or "human" (+1 if action=optimal_action, -1 otherwise)

        # -> operator: selection operator, either "argmax" or "softmax" 
        #              "softmax" is mainly used for the boltzmann irrational bias which already uses a Boltzmann (or Gibbs, softmax) summary operator.
        #              Here, the softmax operator is used to turn expected rewards into a distribution over actions. 

        v_choice = []
        self.env.env.reset(np.array(state))


        for a in range(4):

            new_state = self.new_state_table[tuple(state) + (a,)]

            if reward_type=="human":
                optimal_action = self.optimal_policy[tuple(state)]

                #print(a,optimal_action)
                if a==optimal_action:
                    reward = 1
                else:
                    reward = -1

            elif reward_type=="env":
                reward = self.reward_table[tuple(state)+(a,)]

            else:
                print("Unknown reward type")


            v_choice.append(reward + self.discount*v[tuple(new_state)])


        p = np.exp(v_choice)/np.sum(np.exp(v_choice))
        h = -np.sum(p*np.log(p))

        if operator=="argmax":
            action = np.argmax(v_choice)

        elif operator=="softmax":
            v_choice = np.array(v_choice)
            b = np.max(v_choice)
            x = np.exp((v_choice-b)*1)/np.sum(np.exp((v_choice-b)*1))
            action = np.random.choice([0,1,2,3],p=x)

        

        return action,h


    def generate_traj_v(self,v):

        # Generates trajectory following a policy derived from value function v
        # If RENDER=True, the trajectory is displayed on the GymMaze graphical environment
        # Also, if RENDER=TRUE, the entropy of expected rewards distribution at each step is displayed 

        done=False
        obv = self.env.env.reset([0,0])
        s = tuple(obv)
        it=0
        action_ = []
        entropy = []

        self.env.render()

        while not(done):# and it<1000:

            action, h = self.select_action_from_v(s,v,REWARD_T,OPE)
            entropy.append(h)
            new_s,reward,done,_ = self.env.step(int(action))

            it+=1
            action_.append(action)
            obv = new_s
            s = tuple(obv)


            if RENDER:
                self.env.render()
                time.sleep(RENDER_TIME)
                
        print("Start ",self.env.reset(),"->",it,"iterations",self.action2str(action_))

        if RENDEER:
            plt.figure()
            plt.plot(entropy)
            plt.show()

        
        

        return action_


    def edges_and_walls_list_extractor(self):

          edges_list = []
          walls_list = []
          maze = self.env.env.maze_view.maze

          print("walls maze size",self.maze_size)

          
          # top line and left line
          for i in range(0,self.maze_size):
              walls_list.append([[0,0],[i,i+1]]) # north walls
              walls_list.append([[i,i+1],[0,0]]) # west walls

          # other matplotlib.lines
          for i in range(0,self.maze_size):
              for j in range(0,self.maze_size):
                  walls_list.append([[i+1,i+1],[j,j+1]]) # south walls
                  walls_list.append([[i,i+1],[j+1,j+1]]) # east walls


          for i in range(0,self.maze_size):
              for j in range(0,self.maze_size):
                  maze_cell = maze.get_walls_status(maze.maze_cells[j,i])
                  if maze_cell['N']==1 and [[i,i],[j,j+1]] in walls_list:
                      walls_list.remove([[i,i],[j,j+1]])
                  if maze_cell['S']==1 and [[i+1,i+1],[j,j+1]] in walls_list:
                      walls_list.remove([[i+1,i+1],[j,j+1]])
                  if maze_cell['E']==1 and [[i,i+1],[j+1,j+1]] in walls_list:
                      walls_list.remove([[i,i+1],[j+1,j+1]])
                  if maze_cell['W']==1 and [[i,i+1],[j,j]] in walls_list:
                      walls_list.remove([[i,i+1],[j,j]])

          for i in range(0,self.maze_size):
              for j in range(0,self.maze_size):
                  idx = i + j*self.maze_size
                  if [[i,i],[j,j+1]] not in walls_list:
                      edges_list.append((idx,idx-1,1))
                      #graph.add_edge(idx,idx-1,1)
                  if [[i+1,i+1],[j,j+1]] not in walls_list:
                      edges_list.append((idx,idx+1,1))
                      #graph.add_edge(idx,idx+1,1)
                  if [[i,i+1],[j+1,j+1]] not in walls_list:
                      edges_list.append((idx,idx+self.maze_size,1))
                      #graph.add_edge(idx,idx+maze_size,1)
                  if [[i,i+1],[j,j]] not in walls_list:
                      edges_list.append((idx,idx-self.maze_size,1))
                      #graph.add_edge(idx,idx-maze_size,1)

          return edges_list, walls_list

    
    ############################ Q-LEARNING ########################################

    def q_learning(self):

        q_table = np.zeros((self.maze_size,self.maze_size,4), dtype=float)
        streak = 0
        reach = 0

        for e in tqdm(range(MAX_EPISODE)):


            random_reset = np.random.randint(self.maze_size,size=2)
            obv = self.env.env.reset(random_reset)
            self.env._elapsed_steps = 0

            state = tuple(obv)

            if e%1000==0:
                print("Episode #",e,"(",reach,")")

            for k in range(MAX_STEP):

                epsi = self.get_epsilon(e)
                action = self.select_action(state,q_table,epsi)
                new_s, reward, done, _ = self.env.step(action)
                new_s = tuple(new_s)
                self.update_q(q_table,action,state,new_s,reward,done)
                state = new_s

                if done :
                    break

                if RENDER:
                    self.env.render()

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



    def update_q(self,q_table,action,state,new_state,reward,done):

        state = tuple(state)
        new_state = tuple(new_state)

        if done:
            td = reward - q_table[state+(action,)]
            #print("IN")
        else:
            td = reward + self.discount*np.max(q_table[new_state]) - q_table[state+(action,)]

        q_table[state+(action,)] += self.lr*td


    def get_epsilon(self,e):

        return max(self.epsilon,0.3 - e*self.epsilon/(MAX_EPISODE*0.60))

    def select_action(self,state,q,e):
        e = np.random.rand(1)[0]
        epsi = self.get_epsilon(e)
        if e < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = int(np.argmax(q[state]))
        return action



    ##################### BOLTZMANN NOISY RATIONAL #################################

    def boltz_rational_noisy(self,q_table,beta):
        # Tau : temperature coefficient
        # n : number of demonstrations generated from the same start

        dic_action = ['N','S','E','W']
        obv = self.env.env.reset([0,0])
        state = tuple(obv)
        a=[]

        self.env.render()
        done=0
        a.append([])

        while not(done):

            actions = q_table[state]
            b = max(actions)
            boltz_distribution = np.exp((actions-b)/beta)/np.sum(np.exp((actions-b)/beta))

            noisy_behaviour = np.random.choice(dic_action,p=boltz_distribution)

            new_state,reward,done,_ = self.env.step(noisy_behaviour)

            state=tuple(new_state)
            a.append(noisy_behaviour)

            if RENDER:
                self.env.render()
                time.sleep(RENDER_TIME)

        return a


    ################################################################################
    #########################  IRRATIONAL BIASES  ##################################
    ################################################################################



    #########################  BOLTZMANN RATIONAL ##################################

    def v_from_q(self,q):

        v = np.zeros((self.maze_size,self.maze_size))

        for i in range(self.maze_size):
            for j in range(self.maze_size):
                state = tuple([i,j])
                v[i,j] = np.max(q[state])

        v[tuple(self.env.observation_space.high)] = 1

        return v

    def boltz_rational(self,beta):

        v_vector = np.zeros((self.maze_size,self.maze_size))
        #v_vector = np.random.rand(self.maze_size,self.maze_size)

        end = self.env.observation_space.high

        self.env.reset()
        self.env.render()
        threshold = 1e-5
        err=2

        it=0

        while err>threshold and it<2000:

            v_temp = np.copy(v_vector)
            err=0

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    state = tuple([i,j])

                    if state==tuple(end):
                        break

                    self.env.env.reset(np.array(state))

                    v = v_temp[tuple(state)]
                    x = []
                    for a in range(4):

                        new_state = self.new_state_table[tuple(state) + (a,)]
                        optimal_action = self.optimal_policy[state]


                        if a!=optimal_action:
                            reward = -1
                        else:
                            reward = 1 #self.reward_table[state+(a,)]

                        x.append(reward + self.discount*v_vector[tuple(new_state)])

                    x = np.array(x)
                    b = np.max(x)
                    v_vector[tuple(state)] = np.sum(x*np.exp((x-b)*beta))/np.sum(np.exp((x-b)*beta))

                    
                    err = max(err,abs(v-v_vector[tuple(state)]))

            it+=1

            if it%1000==0:
                print("Iteration",it," err =",err)

        print("VI Boltz done")
        
        v_vector[tuple(end)] = 1

        return v_vector


    ####################### PROSPECT BIAS (loss aversion + scope insensitivity) ############################

    def prospect_bias(self,c):
        #v_vector = np.random.rand(env.size[1]*env.size[0])
        v_vector = np.zeros((self.maze_size,self.maze_size))
        end = self.env.observation_space.high
        v_vector[tuple(end)] = 1

        self.env.reset()
        self.env.render()

        threshold = 1e-8
        err=2

        while err>threshold:

            v_temp = np.copy(v_vector)
            err=0

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    state = tuple([i,j])

                    if state==tuple(end):
                        pass
                    else:
                        self.env.env.reset(state)
                        v = v_temp[state]
                        x = []
                        for a in range(4):

                            new_state = self.new_state_table[state + (a,)]
                            reward = self.reward_table[state+(a,)]

                            if reward > 0:
                                reward = np.log(1+abs(reward))
                            elif reward==0:
                                reward = 0
                            else:
                                reward = -c*np.log(1+abs(reward))

                            x.append(reward + self.discount*v_temp[tuple(new_state)])

                        v_vector[state] = np.max(x)#reward + self.discount*v_temp[tuple(new_state)]


                        err = max(err,abs(v-v_vector[tuple(state)]))


        print("VI prospect bias done")
        return v_vector


    ######################### MYOPIC DISCOUNT ##########################################################

    def myopic_discount(self,disc):


        #v_vector = np.random.rand(self.maze_size,self.maze_size)
        v_vector = np.zeros((self.maze_size,self.maze_size),dtype=float)
        end = self.env.observation_space.high
        #v_vector[tuple(end)] = 1

        self.env.reset()

        err=2
        threshold = 1e-8

        while err > threshold:

            
            err = 0
            v_temp = np.copy(v_vector)
            #print(v_vector)

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    

                    state = tuple([i,j])
                    
                    if (state==tuple([self.maze_size-1,self.maze_size-1])):
                        break

                    self.env.env.reset(np.array(state))
                    v = v_temp[state]
                    x = []

                    optimal_action = self.optimal_policy[state]


                    for a in range(4):

                        new_state = self.new_state_table[state+(a,)]

                        if a!=optimal_action:
                            reward = -1
                        else:
                            reward = 1
                        
                        x.append(reward + disc*v_temp[new_state])

                    #print(x)
                    v_vector[state] = np.max(x)
                    err = max(err,abs(v-v_vector[tuple(state)]))
                    
            #time.sleep(2)
            #print(v_vector)
            #err = np.max(v_temp-v_vector)
            #print(v_vector)

   
        end = self.env.observation_space.high
        v_vector[tuple(end)] = 1
        print("VI myopic discount done")
        return v_vector


    



    """
    def value_iteration_dynamic_boltzmann(self,beta):

        v_vector = np.zeros((self.maze_size,self.maze_size))
        #v_vector = np.random.rand(self.maze_size,self.maze_size)
        q_table = np.zeros((self.maze_size,self.maze_size,4), dtype=float)

        end = self.env.observation_space.high

        self.env.reset()
        self.env.render()
        threshold = 1e-5
        err=2

        it=0

        while err>threshold:

            v_temp = np.copy(v_vector)
            err=0

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    state = tuple([i,j])

                    if state==tuple(end):
                        v_vector[state] = 1
                        break

                    self.env.env.reset(state)

                    v = v_temp[tuple(state)]

                    for action in range(4):
                        new_state = self.new_state_table[tuple(state) + (action,)]
                        reward = self.reward_table[tuple(state)+(action,)]
                        q_table[state+(action,)] = reward + self.discount*v_temp[new_state]

                    x = np.array(q_table[state])
                    b = np.max(x)
                    v_vector[state] = np.sum(x*np.exp((x-b)/beta))/np.sum(np.exp((x-b)/beta))
                    
                    err = max(err,abs(v-v_vector[tuple(state)]))

                    #print(x,v_vector[state])
                    #time.sleep(2)

            it+=1

            if it%1000==0:
                print("Iteration",it," err =",err)

        print("VI Boltz done")
        
        return v_vector, q_table

    def select_action_from_v(self,state,v):

        v_choice = []
        self.env.env.reset(np.array(state))


        for a in range(4):

            new_state = self.new_state_table[tuple(state) + (a,)]
            reward = self.reward_table[tuple(state)+(a,)]
            v_choice.append(reward + self.discount*v[tuple(new_state)])


        #print(v_choice)

        # v_choice = np.array(v_choice)
        # b = np.max(v_choice)
        # x = np.exp((v_choice-b))/np.sum(np.exp((v_choice-b)))
        # #print(x)
        # action = np.random.choice([0,1,2,3],p=x)

        action = np.argmax(v_choice)

        return action

    def select_action_from_v_human(self,state,v):

        v_choice = []
        self.env.env.reset(np.array(state))

        optimal_action = self.optimal_policy[state]


        for a in range(4):

            new_state = self.new_state_table[state+(a,)]

            if a!=optimal_action:
                reward = -1
            else:
                reward = self.reward_table[state+(a,)]

            
            v_choices.append(reward + disc*v_temp[new_state])
       

        #print(v_choice)

        # v_choice = np.array(v_choice)
        # b = np.max(v_choice)
        # x = np.exp((v_choice-b))/np.sum(np.exp((v_choice-b)))
        # #print(x)
        # action = np.random.choice([0,1,2,3],p=x)

        action = np.argmax(v_choice)

        return action
    """
