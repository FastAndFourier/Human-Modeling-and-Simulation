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
MAX_STEP = 200
# DISCOUNT = 1
MIN_STREAK = MAX_EPISODE
RENDER = True
SIMULATE = False
RENDER_TIME = 0.1



class MyMaze():

	def __init__(self,env_name,lr=0.05,epsi=0.02,disc=1):

		self.env = gym.make(env_name)
		#print(self.env)
		self.env.render()

		self.maze_size = tuple((self.env.observation_space.high + np.ones(self.env.observation_space.shape)).astype(int))[0]
		self.lr = lr
		self.epsilon = epsi
		self.discount = disc

		self.reward_table = self.get_reward()
		self.new_state_table = self.get_new_state()

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

	    reward_tab = np.zeros((self.maze_size,self.maze_size,4))
	    state = self.env.reset()
	    state = tuple(state)

	    for i in range(self.maze_size):
	        for j in range(self.maze_size):

	            state = tuple([i,j])

	            for a in range(4):
	                self.env.env.reset(state)
	                new_s,r,_,_ = self.env.step(a)
	                reward_tab[state][a] = r

	    return reward_tab


	def get_new_state(self):

	    new_state_tab = np.zeros((self.maze_size,self.maze_size,4),dtype=tuple)
	    state = self.env.reset()
	    state = tuple(state)

	    for i in range(self.maze_size):
	        for j in range(self.maze_size):

	            state = tuple([i,j])

	            for a in range(4):
	                self.env.env.reset(state)
	                new_s,r,_,_ = self.env.step(a)
	                new_state_tab[state+(a,)] = tuple(new_s)


	    return new_state_tab

	def select_action_from_v(self,state,v):

	    v_choice = []
	    self.env.env.reset(state)


	    for a in range(4):

	        new_state = self.new_state_table[tuple(state) + (a,)]
	        reward = self.reward_table[tuple(state)+(a,)]
	        v_choice.append(reward + self.discount*v[tuple(new_state)])

	    action = np.argmax(v_choice)

	    return action

	############################ Q-LEARNING ########################################

	def simulate(self):

	    #env.render()
	    q_table = np.zeros((self.maze_size,self.maze_size,4), dtype=float)
	    streak = 0
	    reach = 0

	    for e in tqdm(range(MAX_EPISODE)):

	        obv = self.env.env.reset(np.array([0,0]))
	        self.env._elapsed_steps = 0

	        state = tuple(obv)

	        if e%1000==0:
	            #print(get_epsilon(e))
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



	def update_q(self,q,a,s,s1,r,done):

	    s = tuple(s)
	    s1 = tuple(s1)

	    if done:
	        td = r - q[s+(a,)]
	        #print("IN")
	    else:
	        td = r + self.discount*np.max(q[s1+(a,)]) - q[s+(a,)]

	    q[s+(a,)] += self.lr*td


	def get_epsilon(self,e):

	    return max(self.epsilon,0.1 - e*self.epsilon/(MAX_EPISODE*0.60))

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


	"""

	    IRRATIONAL BIASES

	"""

	#########################  BOLTZMANN RATIONAL ##################################

	def v_from_q(self,q):

		v = np.zeros((self.maze_size,self.maze_size))

		for i in range(self.maze_size):
			for j in range(self.maze_size):
				state = tuple([i,j])
				v[i,j] = np.max(q[state])

		v[tuple(self.env.observation_space.high)] = 1

		return v

	def boltz_rational(self,beta,theta):

	    v_vector = np.zeros((self.maze_size,self.maze_size))

	    end = self.env.observation_space.high
	    v_vector[tuple(end)] = 1

	    self.env.reset()
	    self.env.render()
	    #theta=0.05
	    err=2

	    while err>theta:

	        v_temp = np.copy(v_vector)
	        err=0

	        for i in range(self.maze_size):
	            for j in range(self.maze_size):

	                state = self.env.env.reset([i,j])

	                v = v_vector[tuple(state)]
	                x = []
	                for a in range(4):
	                    new_state = self.new_state_table[tuple(state) + (a,)]
	                    reward = self.reward_table[tuple(state)+(a,)]
	                    x.append(reward + self.discount*v_temp[tuple(new_state)])

	                #print(x)
	                x = np.array(x)
	                b = np.max(x)
	                v_vector[tuple(state)] = np.sum(x*np.exp((x-b)*beta))/np.sum(np.exp((x-b)*beta))

	                err = max(err,abs(v_vector[tuple(state)]-v))


	        #print(err)
	    print("VI Boltz done")

	    return v_vector

	def generate_traj_v(self,v):

	    done=False
	    obv = self.env.env.reset([0,0])
	    s = tuple(obv)
	    it=0
	    action_ = []

	    self.env.render()

	    while not(done) and it<1000:

	        action = self.select_action_from_v(s,v)
	        new_s,reward,done,_ = self.env.step(int(action))

	        it+=1
	        action_.append(action)
	        obv = new_s
	        s = tuple(obv)


	        if RENDER:
	            self.env.render()
	            time.sleep(RENDER_TIME)
	    print("Start ",self.env.reset(),"->",it,"iterations",self.action2str(action_))



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
