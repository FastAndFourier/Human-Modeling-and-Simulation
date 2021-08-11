import gym
import gym_maze
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import time
import Graph

MAX_EPISODE = 10000
MAX_STEP = 500
MIN_STREAK = MAX_EPISODE
RENDER = 0


DETERMINISTIC = True




class MyMaze():

    def __init__(self,env_name,reward,lr=0.05,epsi=0.02,disc=0.99):

        self.env = gym.make(env_name)
        self.env.render()
        self.reward_type = reward

        self.maze_size = self.env.observation_space.high[0] + 1
        self.epsilon = epsi
        self.discount = disc
        self.lr = lr

        self.metric_graph = self.set_graph( )

        self.reward_table = [] 
        self.new_state_table = self.get_new_state()
        self.transition_table = self.get_transition_probalities()
        self.optimal_policy = self.set_optimal_policy()
        


    ################# HELPER FUNCTIONS #############################################

    def action2str(self,demo):

        # Parameter: demo, int in [0,3]
        # Return: res, str

        # Turns action index into str 

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

    def sub2lin_traj(self,traj):

        lin_traj = []
        for state in traj:
            lin_traj.append(state[0]*self.maze_size+state[1])

        return lin_traj

    ############### DISPLAY ########################################################

    def edges_and_walls_list_extractor(self):

          edges_list = []
          walls_list = []
          maze = self.env.env.maze_view.maze

          
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

    def plot_v_value(self,fig,ax,v_vector,title):

        _, walls_list = self.edges_and_walls_list_extractor()

        ax.set_xlim(-1,self.maze_size)
        ax.set_ylim(self.maze_size,-1)
        ax.set_aspect('equal')


        im = ax.imshow(np.transpose(v_vector.reshape(self.maze_size,self.maze_size)))

        for state in range(0,self.maze_size*self.maze_size):
            i=state//self.maze_size
            j=state%self.maze_size
            text = ax.text(i,j, str(v_vector[i,j])[0:4],ha="center", va="center", color="black")


        for i in walls_list:
            ax.add_line(mlines.Line2D([i[1][0]-0.5,i[1][1]-0.5],[i[0][0]-0.5,i[0][1]-0.5],color='k'))

        for i in range(0,self.maze_size):
            ax.add_line(mlines.Line2D([-0.5,-0.5],[i-0.5,i+0.5],color='k'))
            ax.add_line(mlines.Line2D([i-0.5,i+0.5],[-0.5,-0.5],color='k'))
            ax.add_line(mlines.Line2D([self.maze_size-0.5,self.maze_size-0.5],[i-0.5,i+0.5],color='k'))
            ax.add_line(mlines.Line2D([i-0.5,i+0.5],[self.maze_size-0.5,self.maze_size-0.5],color='k'))

        fig.suptitle(title)

    def plot_traj(self,fig,ax,v_vector,nb_traj,max_step,title,operator,beta):

        traj = np.zeros((self.maze_size,self.maze_size),dtype=int)
        total_length = []

        for epoch in tqdm(range(nb_traj)):
            self.env.reset()
            state = [0,0]
            traj[tuple(state)]+=1
            length = 0
            
            while (self.env.state!=self.env.observation_space.high).any() and length < max_step:
                action = self.select_action_from_v(state,v_vector,self.reward_type,operator,beta)[0]
                new_s,reward,done,_ = self.env.step(int(action))
                state = new_s
                traj[tuple(state)]+=1
                length+=1
            total_length.append(length)

        print("Mean length",int(np.array(total_length).mean()),"Standard deviation",int(np.array(total_length).std()))
        print(total_length)
        ax.set_xlim(-1,self.maze_size)
        ax.set_ylim(self.maze_size,-1)
        ax.set_aspect('equal')
        fig.suptitle(title)

        #Draw value table
        im = ax.imshow(np.transpose(traj.reshape(self.maze_size,self.maze_size)))
        for state in range(0,self.maze_size*self.maze_size):
            i=state//self.maze_size
            j=state%self.maze_size
            text = ax.text(i,j, str(traj[i,j])[0:4],ha="center", va="center", color="black")

        _, walls_list = self.edges_and_walls_list_extractor()

        for i in walls_list:
            ax.add_line(mlines.Line2D([i[1][0]-0.5,i[1][1]-0.5],[i[0][0]-0.5,i[0][1]-0.5],color='k'))

        for i in range(0,self.maze_size):
            ax.add_line(mlines.Line2D([-0.5,-0.5],[i-0.5,i+0.5],color='k'))
            ax.add_line(mlines.Line2D([i-0.5,i+0.5],[-0.5,-0.5],color='k'))
            ax.add_line(mlines.Line2D([self.maze_size-0.5,self.maze_size-0.5],[i-0.5,i+0.5],color='k'))
            ax.add_line(mlines.Line2D([i-0.5,i+0.5],[self.maze_size-0.5,self.maze_size-0.5],color='k'))



    def plot_policy(self,fig,ax,v_vector,title,operator,beta):

        _, walls_list = self.edges_and_walls_list_extractor()

        ax.set_xlim(-1,self.maze_size)
        ax.set_ylim(self.maze_size,-1)
        ax.set_aspect('equal')

        ax.scatter(0,0, marker="o", s=100,c="b")
        ax.scatter(self.maze_size-1,self.maze_size-1, marker="o", s=100,c="r")

        for i in range(self.maze_size):
            for j in range(self.maze_size):

                if ([i,j]==[self.maze_size-1,self.maze_size-1]):
                    break

                action = self.select_action_from_v([i,j],v_vector,self.reward_type,operator,beta)[0]

                if action==0:
                    ax.quiver(i,j,0,.05,color='c')#,width=0.01,headwidth=2,headlength=1)
                if action==1:
                    ax.quiver(i,j,0,-.05,color='c')#,width=0.01,headwidth=2,headlength=1)
                if action==2:
                    ax.quiver(i,j,.05,0,color='c')#,width=0.01,headwidth=2,headlength=1)
                if action==3:
                    ax.quiver(i,j,-.05,0,color='c')#,width=0.01,headwidth=2,headlength=1)

        
        for i in walls_list:
            ax.add_line(mlines.Line2D([i[1][0]-0.5,i[1][1]-0.5],[i[0][0]-0.5,i[0][1]-0.5],color='k'))

        for i in range(0,self.maze_size):
            ax.add_line(mlines.Line2D([-0.5,-0.5],[i-0.5,i+0.5],color='k'))
            ax.add_line(mlines.Line2D([i-0.5,i+0.5],[-0.5,-0.5],color='k'))
            ax.add_line(mlines.Line2D([self.maze_size-0.5,self.maze_size-0.5],[i-0.5,i+0.5],color='k'))
            ax.add_line(mlines.Line2D([i-0.5,i+0.5],[self.maze_size-0.5,self.maze_size-0.5],color='k'))

        fig.suptitle(title)


    ##################### ENVIRONMENT MODEL #########################################


    def get_transition_probalities(self):

        # Parameter: None
        # Return: transition_tab, ndarray of size (number_of_states,number_of_states,number_of_actions)

        # Outputs transition matrix T(s,a,s')
        # DETERMINISTIC = True: Every possible transition as probability 1
        # DETERMINISTIC = False: Maze exploration for transition probability estimation

        transition_tab = np.zeros((self.maze_size*self.maze_size,self.maze_size*self.maze_size,4),dtype=tuple)

        if DETERMINISTIC:
            #Deterministic case


            dic_action = ["N","S","E","W"]
            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    state = self.env.env.reset(np.array([i,j]))
                    state = tuple(state)

                    for a in range(4):
                        new_state = self.new_state_table[tuple(state)+(a,)]
                        lin_state = state[0]*self.maze_size+state[1]
                        #print(state,lin_state)
                        lin_new_state = new_state[0]*self.maze_size+new_state[1]
                        #print(state,"--",dic_action[a],"->",new_state)

                        if tuple(new_state)==state:
                            transition_tab[lin_state,lin_state,a] = 1  
                        else:
                            transition_tab[lin_state,lin_new_state,a] = 1
                    #print("\n")
        else:

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    for a in range(4):

                        for e in range(100):
                            state = self.env.env.reset(np.array([i,j]))
                            state = tuple(state)
                            new_state,reward,done,_ = self.env.step(a)
                            lin_state = state[0]*self.maze_size+state[1]
                            lin_new_state = new_state[0]*self.maze_size+new_state[1]
                            transition_tab[lin_state,lin_state,a]+=1

            transition_tab = transition_tab/100


        return transition_tab


    def set_reward(self,obstacle=[]):


        # Parameter: obstacle, ndarray
        # Output: None

        # Sets human reward table class attribute
        # Gives reward for taking action a at state s

        # If taking action a at state s bring the agent on one of the optimal paths: 
        #       reward = 1, 
        #       -1 otherwise (+10 for reaching final state)

        # Obstacles can be added (reward = -2)

        reward_table = np.zeros((self.maze_size,self.maze_size,4))

        end = tuple(self.env.observation_space.high)
        empty_obstacle = (obstacle==[])

        for i in range(self.maze_size):
            for j in range(self.maze_size):

                state = tuple([i,j])

                for action in range(4):

                    optimal_action = self.optimal_policy[i,j,action]
                    new_state = self.new_state_table[state+(action,)]

                    o = [(k==list(state)).all() for k in obstacle]
                    #print(o)
                    if not(empty_obstacle) and (True in o):
                        reward_table[state+(action,)] = -2
                    elif optimal_action:
                        if new_state==end:
                            reward_table[state+(action,)] = 10
                        else:
                            reward_table[state+(action,)] = 1
                    else:
                        reward_table[state+(action,)] = -1


        self.reward_table = reward_table

    def get_new_state(self):

        # Parameter: None
        # Return: new_state_tab, ndarray of size (maze_size,maze_size,number_of_actions)

        # Outputs an array with every new state reached after executing action a from a state s

        new_state_tab = np.zeros((self.maze_size,self.maze_size,4),dtype=tuple)
        state = self.env.reset()
        state = tuple(state)

        dic_action = ["N","S","E","W"]

        for i in range(self.maze_size):
            for j in range(self.maze_size):

                state = np.array([i,j])

                for a in range(4):
                    self.env.env.reset(state)
                    new_s,r,_,_ = self.env.step(int(a))
                    new_state_tab[tuple(state)+(a,)] = tuple(new_s)


        return new_state_tab



    def get_entropy_map_v(self,v_table,beta):

        # Parameters: - v_table, ndarray of size (number_of_states,number_of_states)
        #             - beta, float
        #
        # Return: entropy_map, ndarray of size (number_of_states,number_of_states)

        # Outputs entropy map using V-value softmax actor 


        entropy_map = np.zeros((self.maze_size,self.maze_size),dtype=float)

        for i in range(self.maze_size):
            for j in range(self.maze_size):


                state = tuple([i,j])
                lin_state = i*self.maze_size + j
                x = []

                for a in range(4):

                    non_zero_new_state = np.where(self.transition_table[lin_state,:,a]!=0)[0]
                        
                    for k in non_zero_new_state:

                        new_state = tuple([k//self.maze_size,k%self.maze_size])

                        if self.reward_type=="env":
                            if new_state == tuple(end):
                                reward = 1
                            else:
                                reward = -1/(self.maze_size**2)
                        else:
                            reward = self.reward_table[state+(a,)]

                        x.append(self.transition_table[lin_state,k,a]*(reward+self.discount*v_table[new_state]))

                x = np.array(x,dtype=float)
                b = np.max(x)
                p = np.exp((x-b)*beta)/np.sum(np.exp((x-b)*beta))
                entropy_map[i,j] = -np.sum(p*np.log(p))

        return entropy_map

    def get_entropy_map_q(self,q_table,beta):

        # Parameters: - q_table, ndarray of size (number_of_states,number_of_states)
        #             - beta, float
        #
        # Return: entropy_map, ndarray of size (number_of_states,number_of_states)

        # Outputs entropy map using Q-value softmax actor 

        entropy_map = np.zeros((self.maze_size,self.maze_size),dtype=float)

        for i in range(self.maze_size):
            for j in range(self.maze_size):

                state = tuple([i,j])
                b = np.max(q_table[state])
                p = np.exp((q_table[state]-b)*beta)/np.sum(np.exp((q_table[state]-b)*beta))
                entropy_map[i,j] = -np.sum(p*np.log(p))

        return entropy_map



    ##################### ACTOR ###############################################

    def set_graph(self):

        # Parameter: None
        # Return: metric_graph, instance of class Graph (see Graph.hpp and Graph.cpp)


        # Set Maze's graph to compute metrics

        self.env.reset()

        vertex = []
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                vertex.append(i*self.maze_size+j)

        connection = {}
        for v in vertex:
            c = []
            for a in range(4):
                self.env.env.reset(np.array([v//self.maze_size,v%self.maze_size],dtype=int))
                new_state,_,_,_ = self.env.step(a)
                lin_state = new_state[0]*self.maze_size+new_state[1]
                if lin_state!=v:
                    c.append(lin_state)

            connection[v] = c

        metric_graph = Graph.Graph()

        for key in range(self.maze_size*self.maze_size):

            metric_graph.add_node(key)

            for val in connection[key]:
                metric_graph.add_connection([key,val,1])

        return metric_graph




    def set_optimal_policy(self):

        # Parameter: None
        # Return: optimal_policy, ndarray of size (number_of_states,number_of_states,number_of_aactions)

        # Sets optimal policy:
        # If action A taken at state S lead to a state S1 located on one of the optimal paths, then 
        #   --> optimal_policy[S_row,S_column,A] = 1
        #   --> = 0 otherwise 

        self.env.reset()

        vertex = []
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                vertex.append(i*self.maze_size+j)

        connection = {}
        for v in vertex:
            c = []
            for a in range(4):
                self.env.env.reset(np.array([v//self.maze_size,v%self.maze_size],dtype=int))
                new_state,_,_,_ = self.env.step(a)
                lin_state = new_state[0]*self.maze_size+new_state[1]
                if lin_state!=v:
                    c.append(lin_state)

            connection[v] = c

        metric_graph = Graph.Graph()

        for key in range(self.maze_size*self.maze_size):

            metric_graph.add_node(key)

            for val in connection[key]:
                metric_graph.add_connection([key,val,1])

        # optimal_connections = metric_graph.dijkstra_optimal_connections(0,self.maze_size*self.maze_size -1)
        optimal_connections = metric_graph.optimal_connections_full_graph(self.maze_size*self.maze_size -1)
        #print(optimal_connections[0])

        optimal_policy = np.zeros((self.maze_size,self.maze_size,4))
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                lin_state = i*self.maze_size + j
                for a in range(4):
                    new_state = self.new_state_table[i,j,a]
                    lin_new_state = new_state[0]*self.maze_size + new_state[1]

                    if lin_state in optimal_connections:
                        if (lin_new_state in optimal_connections[lin_state]):
                            optimal_policy[i,j,a] = 1


        return optimal_policy



    def select_action_from_q(self,state,q,e):

        # Parameters: - state, list or tuple of length 2
        #             - q, ndarray of size (number_of_states,number_of_states,number_of_actions)
        #             - e, float 

        # Return: action, integer in [0,3]

        # Epsilon-greedy actor for q-learning

        e = np.random.rand(1)[0]
        epsi = self.get_epsilon(e)
        if e < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = int(np.argmax(q[state]))
        return action


    def select_action_from_v(self,state,v,reward_type,operator,beta):

        # Parameters: - state, list or tuple of length 2
        #             - v, ndarray of size (number_of_states,number_of_states)
        #             - reward_type, string wether "env" or "human"
        #             - operator, string wether "softmax" or "argmax"
        #             - beta, float (for softmax actor only)



        # Selects action following a policy given a Value-function

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
                reward = self.reward_table[tuple(state)+(a,)]

            elif reward_type=="env":

                if tuple(new_state)==tuple(self.env.observation_space.high):
                    reward = 1
                else:
                    reward = -1/(self.maze_size*self.maze_size)

            else:
                print("Unknown reward type")


            v_choice.append(reward + self.discount*v[tuple(new_state)])

       
        

        if operator=="argmax":
            action = np.argmax(v_choice)
            x = np.exp(v_choice)/np.sum(np.exp(v_choice))
            h = -np.sum(x*np.log(x))

        elif operator=="softmax":

            v_choice = np.array(v_choice)
            b = np.max(v_choice)
            x = np.exp((v_choice-b)*beta)/np.sum(np.exp((v_choice-b)*beta))
            h = -np.sum(x*np.log(x))
            action = np.random.choice([0,1,2,3],p=x)

        

        return action,h


    def generate_traj_v(self,v,operator,beta,max_step):

        # Parameters: - v, ndarray of size (number_of_states,number_of_states)
        #             - operator, String ("argmax" or "softmax")
        #             - beta, float ("softmax" actor only)
        #             - max_step, integer

        # Generates trajectory following a policy derived from value function v
   

        done=False
        obv = self.env.env.reset([0,0])
        s = tuple(obv)
        it=0
        action_ = []
        entropy = []
        traj = []
        traj.append(list(s))

        while not(done) and it<max_step:


            action, h = self.select_action_from_v(s,v,self.reward_type,operator,beta)
            entropy.append(h)
            new_s,reward,done,_ = self.env.step(int(action))

            it+=1
            action_.append(action)
            obv = new_s
            s = tuple(obv)
            traj.append(list(s))        

        return action_, traj


    

    
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
                action = self.select_action_from_q(state,q_table,epsi)
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

            # if RENDER:
            #     self.env.render()
            #     time.sleep(RENDER_TIME)

        return a


    ################################################################################
    #########################  IRRATIONAL BIASES  ##################################
    ################################################################################



    #########################  V vector from Q-table ##################################

    def v_from_q(self,q):

        v = np.zeros((self.maze_size,self.maze_size))

        for i in range(self.maze_size):
            for j in range(self.maze_size):
                state = tuple([i,j])
                v[i,j] = np.max(q[state])

        v[tuple(self.env.observation_space.high)] = 1

        return v


    #########################  Classical Value-iteration ##################################

    def value_iteration(self):

        # Regular Value iteration

        v_vector = np.zeros((self.maze_size,self.maze_size))

        end = self.env.observation_space.high
        #v_vector[tuple(end)] = 1

        self.env.reset()
        threshold = 1e-5
        err = 2
        start = time.time()

        while err > threshold:

            v_temp = np.copy(v_vector)
            err = 0

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    state = tuple([i,j])
                    lin_state = i*self.maze_size+j

                    

                    if state == tuple(end):
                        break

                    else:
                        v = v_temp[state]
                        x = []


                        for a in range(4):

                            y = 0

                            non_zero_new_state = np.where(self.transition_table[lin_state,:,a]!=0)[0]

                            if len(non_zero_new_state)==0:
                                print("Multiple new state")
                            
                            for k in non_zero_new_state:

                                new_state = tuple([k//self.maze_size,k%self.maze_size])

                                if self.reward_type=="env":
                                    if new_state == tuple(end):
                                        reward = 1
                                    else:
                                        reward = -1/(self.maze_size**2)
                                else:
                                    reward = self.reward_table[state+(a,)]

                                y+=self.transition_table[lin_state,k,a]*(reward+self.discount*v_temp[new_state])

                                

                            x.append(y)


                        v_vector[state] = np.max(np.array(x))
                        

                        err = max(err,abs(v-v_vector[tuple(state)]))
            #print(err)

        
        print("duration",(time.time()-start))
        print("VI done")

        return v_vector


    #########################  Boltzmann rationality ##################################

    def boltz_rational(self,beta):


        v_vector = np.zeros((self.maze_size,self.maze_size))

        end = self.env.observation_space.high

        #v_vector[tuple(end)] = 5

        self.env.reset()
        threshold = 1e-3
        err = 2

        start = time.time()

        while err > threshold:

            v_temp = np.copy(v_vector)
            err = 0

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    state = tuple([i,j])
                    lin_state = i*self.maze_size+j

                    if state == tuple(end):
                        break

                    else:
                        v = v_temp[state]
                        x = []
                        optimal_action = self.optimal_policy[state]

                        for a in range(4):

                            y = 0


                            non_zero_new_state = np.where(self.transition_table[lin_state,:,a]!=0)[0]

                            for k in non_zero_new_state:

                                new_state = tuple([k//self.maze_size,k%self.maze_size])

                                if self.reward_type=="env":
                                    if new_state == tuple(end):
                                        reward = 1
                                    else:
                                        reward = -1/(self.maze_size**2)
                                else:
                                    reward = self.reward_table[state+(a,)]

                                y+=self.transition_table[lin_state,k,a]*(reward+self.discount*v_temp[new_state])


                            x.append(y)


                        x = np.array(x)
                        b = np.max(x)
                        v_vector[tuple(state)] = np.sum(x*np.exp(beta*(x-b)))/(np.sum(np.exp(beta*(x-b))))

        
                        err = max(err,abs(v-v_vector[tuple(state)]))

            #print(err)


        print("duration",(time.time()-start))
        print("VI Boltz done")

        return v_vector


    #########################  Illusion of control ##################################

    def illusion_of_control(self,n,beta):


        v_vector = np.zeros((self.maze_size,self.maze_size))

        end = self.env.observation_space.high

        self.env.reset()
        threshold = 1e-8
        err = 2
        start = time.time()

        while err > threshold:

            v_temp = np.copy(v_vector)
            err = 0

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    state = tuple([i,j])
                    lin_state = i*self.maze_size+j

                    if state == tuple(end):
                        break

                    else:
                        v = v_temp[state]
                        x = []
                        optimal_action = self.optimal_policy[state]

                        for a in range(4):

                            y = 0

                            non_zero_new_state = np.where(self.transition_table[lin_state,:,a]!=0)[0]
                            
                            for k in non_zero_new_state:

                                new_state = tuple([k//self.maze_size,k%self.maze_size])

                                if self.reward_type=="env":
                                    if new_state == tuple(end):
                                        reward = 1
                                    else:
                                        reward = -1/(self.maze_size**2)
                                else:
                                    reward = self.reward_table[state+(a,)]

                                y+=((self.transition_table[lin_state,k,a])**n)*(reward+self.discount*v_temp[new_state])


                            x.append(y)

                        x = np.array(x)
                        b = np.max(x)
                        v_vector[tuple(state)] = np.sum(x*np.exp(beta*(x-b)))/(np.sum(np.exp(beta*(x-b))))

                        err = max(err,abs(v-v_vector[tuple(state)]))


        
        print("duration",(time.time()-start))
        print("Illusion of Control done")

        return v_vector


    #########################  Optimism / Pessimism ##################################

    def optimism_pessimism(self,omega,beta):


        v_vector = np.zeros((self.maze_size,self.maze_size))

        end = self.env.observation_space.high
        #v_vector[tuple(end)] = 5

        self.env.reset()
        threshold = 1e-8
        err = 2
        start = time.time()

        while err > threshold:

            v_temp = np.copy(v_vector)
            err = 0

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    state = tuple([i,j])
                    lin_state = i*self.maze_size+j

                    if state == tuple(end):
                        break

                    else:
                        v = v_temp[state]
                        x = []
                        #optimal_action = self.optimal_policy[state]

                        for a in range(4):

                            y = 0


                            non_zero_new_state = np.where(self.transition_table[lin_state,:,a]!=0)[0]
                            
                            for k in non_zero_new_state:

                                new_state = tuple([k//self.maze_size,k%self.maze_size])

                                if self.reward_type=="env":
                                    if new_state == tuple(end):
                                        reward = 1
                                    else:
                                        reward = -1/(self.maze_size**2)
                                else:
                                    reward = self.reward_table[state+(a,)]

                                modified_transition_matrix = self.transition_table[lin_state,k,a]*np.exp(omega*(reward+self.discount*v_temp[state]))

                                y+= modified_transition_matrix*(reward+self.discount*v_temp[new_state])


                            x.append(y)

                        x = np.array(x)
                        b = np.max(x)
                        v_vector[tuple(state)] = np.sum(x*np.exp(beta*(x-b)))/(np.sum(np.exp(beta*(x-b))))

                        err = max(err,abs(v-v_vector[tuple(state)]))


        
        print("duration",(time.time()-start))
        print("Optimism/Pessimism done")

        return v_vector


    ####################### PROSPECT BIAS (loss aversion + scope insensitivity) ############################

    def prospect_bias(self,c,beta):
        #v_vector = np.random.rand(env.size[1]*env.size[0])
        v_vector = np.zeros((self.maze_size,self.maze_size))
        end = self.env.observation_space.high
        #v_vector[tuple(end)] = 5

        self.env.reset()
        self.env.render()

        threshold = 1e-3
        err=2

        start = time.time()

        while err>threshold:

            v_temp = np.copy(v_vector)
            err=0

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    state = tuple([i,j])
                    lin_state = i*self.maze_size+j

                    if state==tuple(end):
                        pass
                    else:
                        self.env.env.reset(np.array(state))
                        v = v_temp[state]
                        x = []
                        #optimal_action = self.optimal_policy[state]

                        for a in range(4):


                            y = 0
                            # new_state = self.new_state_table[state + (a,)]
                            # reward = self.reward_table[state+(a,)]


                            non_zero_new_state = np.where(self.transition_table[lin_state,:,a]!=0)

                            for k in non_zero_new_state:

                                new_state = tuple([k//self.maze_size,k%self.maze_size])


                                if self.reward_type=="env":
                                    if new_state == tuple(end):
                                        reward = 1
                                    else:
                                        reward = -1/(self.maze_size**2)
                                else:
                                    reward = self.reward_table[state+(a,)]

                                if reward>0:
                                    reward = np.log(1+reward)
                                elif reward==0:
                                    reward = 0
                                else:
                                    reward = -c*np.log(1+abs(reward))


                                y+=self.transition_table[lin_state,k,a]*(reward+self.discount*v_temp[new_state])

                            x.append(y)

                        x = np.array(x,dtype=float)
                        b = np.max(x)
                        v_vector[tuple(state)] = np.sum(x*np.exp(beta*(x-b)))/(np.sum(np.exp(beta*(x-b))))


                        err = max(err,abs(v-v_vector[tuple(state)]))


            #print(err)

        print("duration",time.time()-start)
        print("Prospect bias done")
        return v_vector


    #########################  Extremal (duration neglect) ##################################

    def extremal(self,alpha,beta):

        # Regular Value iteration

        v_vector = np.zeros((self.maze_size,self.maze_size))

        end = self.env.observation_space.high
        #v_vector[tuple(end)] = 5

        self.env.reset()
        threshold = 1e-3
        err = 2
        start = time.time()

        while err > threshold:

            v_temp = np.copy(v_vector)
            err = 0

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    state = tuple([i,j])
                    lin_state = i*self.maze_size+j

                    if state == tuple(end):
                        break

                    else:
                        v = v_temp[state]
                        x = []
                        #optimal_action = self.optimal_policy[state]

                        for a in range(4):

                            y = 0


                            non_zero_new_state = np.where(self.transition_table[lin_state,:,a]!=0)[0]
                            
                            for k in non_zero_new_state:

                                new_state = tuple([k//self.maze_size,k%self.maze_size])

                                if self.reward_type=="env":
                                    if new_state == tuple(end):
                                        reward = 1
                                    else:
                                        reward = -1/(self.maze_size**2)
                                else:
                                    reward = self.reward_table[state+(a,)]

  
                                expected_reward = max(reward,(1-alpha)*reward+alpha*v_temp[new_state])

                                y+=self.transition_table[lin_state,k,a]*expected_reward


                            x.append(y)

                        x = np.array(x)
                        b = np.max(x)
                        v_vector[tuple(state)] = np.sum(x*np.exp(beta*(x-b)))/(np.sum(np.exp(beta*(x-b))))

                        err = max(err,abs(v-v_vector[tuple(state)]))


            #print(err)

        print("duration",(time.time()-start))
        print("Extremal done")

        return v_vector


    ######################### Myopic discount ##########################################################

    def myopic_discount(self,disc,beta):


        #v_vector = np.random.rand(self.maze_size,self.maze_size)
        v_vector = np.zeros((self.maze_size,self.maze_size),dtype=float)
        end = self.env.observation_space.high
        #v_vector[tuple(end)] = 5

        self.env.reset()

        err=2
        threshold = 1e-5

        start = time.time()

        while err > threshold:

            
            err = 0
            v_temp = np.copy(v_vector)
            #print(v_vector)

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    

                    state = tuple([i,j])
                    lin_state = i*self.maze_size+j
                    
                    if (state==tuple([self.maze_size-1,self.maze_size-1])):
                        break

                    self.env.env.reset(np.array(state))
                    v = v_temp[state]
                    x = []

                    #optimal_action = self.optimal_policy[state]


                    for a in range(4):

                        y = 0


                        non_zero_new_state = np.where(self.transition_table[lin_state,:,a]!=0)
                        for k in non_zero_new_state:

                            new_state = tuple([k//self.maze_size,k%self.maze_size])

                            if self.reward_type=="env":
                                if new_state == tuple(end):
                                    reward = 1
                                else:
                                    reward = -1/(self.maze_size**2)
                            else:
                                reward = self.reward_table[state+(a,)]

                            y+=self.transition_table[lin_state,k,a]*(reward+disc*v_temp[new_state])
                        
                        x.append(y)

 
                    x = np.array(x,dtype=float)
                    b = np.max(x)
                    v_vector[tuple(state)] = np.sum(x*np.exp(beta*(x-b)))/(np.sum(np.exp(beta*(x-b))))

                    err = max(err,abs(v-v_vector[tuple(state)]))
            #print(err)

        #end = self.env.observation_space.high
        #v_vector[tuple(end)] = 1
        print("duration ",time.time()-start)
        print("Myopic discount done")
        return v_vector


    
######################### Myopic value iteration ##########################################################

    def myopic_value_iteration(self,h,beta):


        v_vector = np.zeros((self.maze_size,self.maze_size),dtype=float)
        end = self.env.observation_space.high
        #v_vector[tuple(end)] = 5

        self.env.reset()

        err=2
        threshold = 1e-8

        it = 0

        start = time.time()

        while err > threshold and it < h:

            
            err = 0
            v_temp = np.copy(v_vector)
            #print(v_vector)

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    

                    state = tuple([i,j])
                    lin_state = i*self.maze_size+j
                    
                    if (state==tuple([self.maze_size-1,self.maze_size-1])):
                        break

                    self.env.env.reset(np.array(state))
                    v = v_temp[state]
                    x = []

                    #optimal_action = self.optimal_policy[state]


                    for a in range(4):

                        y = 0
                        non_zero_new_state = np.where(self.transition_table[lin_state,:,a]!=0)

                        for k in non_zero_new_state:

                            new_state = tuple([k//self.maze_size,k%self.maze_size])

                            if self.reward_type=="env":
                                if new_state == tuple(end):
                                    reward = 1
                                else:
                                    reward = -1/(self.maze_size**2)
                            else:
                                reward = self.reward_table[state+(a,)]

                            y+=self.transition_table[lin_state,k,a]*(reward+self.discount*v_temp[new_state])
                        
                        x.append(y)

 
                    x = np.array(x,dtype=float)
                    b = np.max(x)
                    v_vector[tuple(state)] = np.sum(x*np.exp(beta*(x-b)))/(np.sum(np.exp(beta*(x-b))))

                    err = max(err,abs(v-v_vector[tuple(state)]))
                    
            it+=1

        print("duration",(time.time()-start))
        print("Myopic value iteration done")
        return v_vector



######################### Hyperbolic discount ##########################################################

    def hyperbolic_discount(self,k_,beta):


        v_vector = np.zeros((self.maze_size,self.maze_size),dtype=float)
        end = self.env.observation_space.high
        #v_vector[tuple(end)] = 10

        self.env.reset()

        err=2
        threshold = 1e-8

        start = time.time()

        while err > threshold:

            
            err = 0
            v_temp = np.copy(v_vector)
            #print(v_vector)

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    

                    state = tuple([i,j])
                    lin_state = i*self.maze_size+j
                    
                    if (state==tuple([self.maze_size-1,self.maze_size-1])):
                        break

                    self.env.env.reset(np.array(state))
                    v = v_temp[state]
                    x = []

                    #optimal_action = self.optimal_policy[state]


                    for a in range(4):

                        y = 0
                        new_state = self.new_state_table[state+(a,)]

                        non_zero_new_state = np.where(self.transition_table[lin_state,:,a]!=0)

                        for k in non_zero_new_state:

                            new_state = tuple([k//self.maze_size,k%self.maze_size])

                            if self.reward_type=="env":
                                if new_state == tuple(end):
                                    reward = 1
                                else:
                                    reward = -1/(self.maze_size**2)
                            else:
                                reward = self.reward_table[state+(a,)]

                            y+=self.transition_table[lin_state,k,a]*(reward+self.discount*v_temp[new_state])/(1+k_*v_temp[new_state])
                        
                        x.append(y)

 
                    x = np.array(x,dtype=float)
                    b = np.max(x)
                    v_vector[tuple(state)] = np.sum(x*np.exp(beta*(x-b)))/(np.sum(np.exp(beta*(x-b))))

                    err = max(err,abs(v-v_vector[tuple(state)]))

            #print(err)

        print("duration",(time.time()-start))
        print("Hyperbolic discount done")
        return v_vector



####################### Local myopia ################################################################

    def local_discount(self,uncertain_state,radius,beta):


        # uncertain_state = tuple(np.random.randint(0,self.maze_size,size=2))
        # radius = np.random.randint(0,6)

        v_vector = np.zeros((self.maze_size,self.maze_size),dtype=float)
        end = self.env.observation_space.high
        #v_vector[tuple(end)] = 10

        self.env.reset()

        err=2
        threshold = 1e-5


        while err > threshold:

            

            err = 0
            v_temp = np.copy(v_vector)
            #print(v_vector)

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    state = tuple([i,j])
                    lin_state = i*self.maze_size+j
                    
                    if (state==tuple([self.maze_size-1,self.maze_size-1])):
                        break

                    in_uncertain_area = False

                    distance_to_uncertain = np.sqrt((i-uncertain_state[0])**2+(j-uncertain_state[1])**2)

                    if distance_to_uncertain < radius:
                        in_uncertain_area = True

                        #print("In uncertain area",state)


                    self.env.env.reset(np.array(state))
                    v = v_temp[state]
                    x = []

                    #optimal_action = self.optimal_policy[state]


                    for a in range(4):

                        y = 0
                        
                        
                        if not(in_uncertain_area):
                            disc=self.discount
                        else:
                            disc=self.discount/2


                        non_zero_new_state = np.where(self.transition_table[lin_state,:,a]!=0)
                        for k in non_zero_new_state:

                            new_state = tuple([k//self.maze_size,k%self.maze_size])

                            reward = self.reward_table[state+(a,)]

                            y+=self.transition_table[lin_state,k,a]*(reward+disc*v_vector[new_state])

                        x.append(y)

 
                    x = np.array(x,dtype=float)
                    b = np.max(x)
                    v_vector[tuple(state)] = np.sum(x*np.exp(beta*(x-b)))/(np.sum(np.exp(beta*(x-b))))

                    err = max(err,abs(v-v_vector[tuple(state)]))

            #print(err)

        print("Local uncertainty done")
        return v_vector

    #########################  Random boltzmann rationality ##################################v

    def random_boltz_rational(self,beta_max,beta_min):


        v_vector = np.zeros((self.maze_size,self.maze_size))

        end = self.env.observation_space.high

        #v_vector[tuple(end)] = 5

        self.env.reset()
        threshold = 1e-5
        err = 2

        start = time.time()

        beta = beta_max*np.ones((self.maze_size,self.maze_size))
        p=0.2

        for i in range(self.maze_size):
            for j in range(self.maze_size):
                if np.random.rand() < p:
                    beta[i,j] = beta_min


        while err > threshold:
            v_temp = np.copy(v_vector)
            err = 0

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    state = tuple([i,j])
                    lin_state = i*self.maze_size+j

                    if state == tuple(end):
                        break

                    else:
                        v = v_temp[state]
                        x = []
                        #optimal_action = self.optimal_policy[state]

                        for a in range(4):

                            y = 0


                            non_zero_new_state = np.where(self.transition_table[lin_state,:,a]!=0)[0]

                            for k in non_zero_new_state:

                                new_state = tuple([k//self.maze_size,k%self.maze_size])

                                if self.reward_type=="env":
                                    if new_state == tuple(end):
                                        reward = 1
                                    else:
                                        reward = -1/(self.maze_size**2)
                                else:
                                    reward = self.reward_table[state+(a,)]

                                y+=self.transition_table[lin_state,k,a]*(reward+self.discount*v_temp[new_state])


                            x.append(y)


                        x = np.array(x)
                        b = np.max(x)

                        v_vector[tuple(state)] = np.sum(x*np.exp(beta[state]*(x-b)))/(np.sum(np.exp(beta[state]*(x-b))))

        
                        err = max(err,abs(v-v_vector[tuple(state)]))

            #print(err)


        print("duration",(time.time()-start))
        print("Random boltzmann done")

        return v_vector




    #########################  Local uncertainty (boltzmann following the entropy) ##################################

    def local_uncertainty(self,table):


        v_vector = np.zeros((self.maze_size,self.maze_size))

        end = self.env.observation_space.high

        #v_vector[tuple(end)] = 5

        self.env.reset()
        threshold = 1e-5
        err = 2

        start = time.time()

        beta = self.get_entropy_map_q(table)


        while err > threshold:
            v_temp = np.copy(v_vector)
            err = 0

            for i in range(self.maze_size):
                for j in range(self.maze_size):

                    state = tuple([i,j])
                    lin_state = i*self.maze_size+j

                    if state == tuple(end):
                        break

                    else:
                        v = v_temp[state]
                        x = []
                        #optimal_action = self.optimal_policy[state]

                        for a in range(4):

                            y = 0


                            non_zero_new_state = np.where(self.transition_table[lin_state,:,a]!=0)[0]

                            for k in non_zero_new_state:

                                new_state = tuple([k//self.maze_size,k%self.maze_size])

                                if self.reward_type=="env":
                                    if new_state == tuple(end):
                                        reward = 1
                                    else:
                                        reward = -1/(self.maze_size**2)
                                else:
                                    reward = self.reward_table[state+(a,)]

                                y+=self.transition_table[lin_state,k,a]*(reward+self.discount*v_temp[new_state])


                            x.append(y)


                        x = np.array(x)
                        b = np.max(x)

                        v_vector[tuple(state)] = np.sum(x*np.exp(beta[state]*(x-b)))/(np.sum(np.exp(beta[state]*(x-b))))


                        err = max(err,abs(v-v_vector[tuple(state)]))

            #print(err)


        print("duration",(time.time()-start))
        print("Random boltzmann done")

        return v_vector

    