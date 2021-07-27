import numpy as np
from Graph import HanoiGraph
import random
import pygame

MAX_EPISODE = 5000

WIDTH6 = 240
WIDTH5 = 200
WIDTH4 = 160
WIDTH3 = 120
WIDTH2 = 80
WIDTH1 = 40

def action2lin(a):
  if a==[0,1]:
    return 0
  elif a==[0,2]:
    return 1
  elif a==[1,0]:
    return 2
  elif a==[1,2]:
    return 3
  elif a==[2,0]:
    return 4
  else:
    return 5

class HanoiEnv():
  
  
  
  def __init__(self,n,epsi=0.02,disc=0.99,lr=0.05):

    self.nb_disk = n
    self.Hgraph = HanoiGraph(n)
    self.state = tuple([0]*self.nb_disk)
    self.epsilon = epsi
    self.discount = disc
    self.lr = lr
    self.action_space = [[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]]
    self.final_state = tuple([2]*self.nb_disk)
    self.MAX_STEP = self.nb_disk*((2**self.nb_disk-1))
    
    pygame.init()
    self.win = pygame.display.set_mode((800,300))
    pygame.display.set_caption("Tower of Hanoi")
    self.render()

    reward_news = self.set_reward_newstate()
    self.reward_table = reward_news[0]
    self.new_state_table = reward_news[1]


  def sub2lin(self,v):
    res = 0
    for k in range(self.nb_disk):
      res+= v[k]*np.power(3,k)
    return res

  def lin2sub(self,i):

    res = [i//3**(self.nb_disk-1)]
    temp = i%3**(self.nb_disk-1)
    for k in range(self.nb_disk-2,0,-1):
      res.append(temp//3**k)
      temp = i%3**k
    res.append(i%3)

    return res

  def set_optimal_policy(self,q_table):

    # Sets optimal policy from a given Q-table

    optimal_policy = np.zeros(3**self.nb_disk,dtype=int)

    for k in range(3**self.nb_disk):
      optimal_policy[k] = np.argmax(q_table[k])


    self.optimal_policy = optimal_policy


  def set_reward_newstate(self):

    reward_table = np.zeros((3**self.nb_disk,6))
    new_state_table = np.zeros((3**self.nb_disk,6),dtype=list)

    for v in self.Hgraph.vertex:
      for action in self.action_space:

        state = self.reset(list(v))

        if action[0] not in list(self.state): # No disk around the giving peg
          new_state = self.state
          reward = -1
          done = False
        else:
          disk_to_move = np.argwhere(np.array(self.state)==action[0])[0][0]
          temp_state = np.copy(self.state)
          temp_state[disk_to_move] = action[1]
          temp_state = tuple(temp_state)


          if temp_state in tuple(self.Hgraph.graph[tuple(self.state)]): # Allowed action
            new_state = temp_state
            done = (new_state == self.final_state)
            if done:
              reward = 5
            else:
              reward = -1/(self.nb_disk**2)
          else:
            new_state = self.state
            reward = -1
            done = 0



        reward_table[self.sub2lin(v),action2lin(action)] = reward
        new_state_table[self.sub2lin(v),action2lin(action)] = new_state


    return reward_table, new_state_table


  def reset(self,state):
    self.state = state
    return self.state

  def step(self,action):   

    if type(action)!=int:
      action = action2lin(action)

    new_state = self.new_state_table[self.sub2lin(self.state),action]
    reward = self.reward_table[self.sub2lin(self.state),action]
    done = (new_state == self.final_state)

    self.state = new_state
    return [new_state, reward, done]


  # def render(self):
  #   self.Hgraph.print_env(self.state)


  def update_q(self,q_table,action,state,new_state,reward,done):

    state = self.sub2lin(state)
    new_state = self.sub2lin(new_state)
    
    action = action2lin(action)

    if done:
        td = reward - q_table[state,action]
    else:
        td = reward + self.discount*np.max(q_table[new_state]) - q_table[state,action]

    
    q_table[state,action] += self.lr*td




  def select_action(self,state,q):
    e = np.random.rand(1)[0]
    state = self.sub2lin(state)
    if e < self.epsilon:
        action = random.sample(self.action_space,1)[0]
    else:
        action_pool = np.argwhere(q[state]==np.max(q[state])).flatten()
        action = self.action_space[np.random.choice(action_pool)]

    #print(action)
    return action

  def q_learning(self):

    q_table = np.zeros((3**self.nb_disk,6), dtype=float)
    nb_done = 0

    for e in range(MAX_EPISODE):

        state = self.reset(np.random.randint(3,size=self.nb_disk))

        for k in range(self.MAX_STEP):

            action = self.select_action(state,q_table)
            #print("Before stepping",self.state,action)
            new_s, reward, done = self.step(action)
            new_s = tuple(new_s)
            self.update_q(q_table,action,state,new_s,reward,done)
            
            state = new_s

            if done :
              nb_done+=1
              break

        if not(e%500):
          print("Episodes ",e,"(",nb_done,")")
          nb_done = 0
        

    return  q_table


  def boltzmann_traj_qtable(self,qtable,beta):

    state = self.reset([0]*self.nb_disk)
    done = False

    traj = [state]

    while not(done):

      x = np.array(qtable[self.sub2lin(state),:],dtype=float)
      b = np.max(x)
      p = np.exp((x-b)*beta)/(np.exp((x-b)*beta)).sum()

      action = np.random.choice(list(range(6)),p=p)
      new_state, reward, done = self.step(int(action))
      state = new_state
      traj.append(state)

    return traj


  def render(self):

    self.win.fill((255,255,255))

    pygame.draw.rect(self.win,(128,128,128),(150,100,10,300)) #left pole
    pygame.draw.rect(self.win,(128,128,128),(400,100,10,300)) #middle pole
    pygame.draw.rect(self.win,(128,128,128),(650,100,10,300)) #right pole



    width = [WIDTH1,WIDTH2,WIDTH3,WIDTH4,WIDTH5,WIDTH6][:self.nb_disk]
    color = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(0,255,255),(255,255,0)][:self.nb_disk]

    position_queue = [0]*len(self.state)
    reversed_state = list(self.state[::-1])

    cpt = 0
    for i0 in np.where(np.array(reversed_state)==0)[0]:
      position_queue[i0] = cpt
      cpt+=1

    cpt = 0
    for i1 in np.where(np.array(reversed_state)==1)[0]:
      position_queue[i1] = cpt
      cpt+=1

    cpt = 0
    for i2 in np.where(np.array(reversed_state)==2)[0]:
      position_queue[i2] = cpt
      cpt+=1

    position_queue = position_queue[::-1]


    for s,w,c,p in zip(reversed_state,width[::-1],color[::-1],position_queue[::-1]):
      if s==0:
        pygame.draw.rect(self.win,c,(150-w//2 + 5,280-p*20,w,20))
      if s==1:
        pygame.draw.rect(self.win,c,(400-w//2 + 5,280-p*20,w,20))
      if s==2:
        pygame.draw.rect(self.win,c,(650-w//2 + 5,280-p*20,w,20))



    pygame.display.update()


  def distance_state(self,start,end):

    if tuple(start)==tuple(end):
      return 0

    nb_state = len(self.Hgraph.vertex)

    source = self.sub2lin(start)
    stop = self.sub2lin(end)

    dist = np.inf*np.ones(nb_state)
    prev = np.zeros(nb_state)
    visited = np.array([False]*nb_state)
    index_visited = np.arange(nb_state)
    dist[source] = 0

    while index_visited.size!=0:

      u = index_visited[dist[index_visited]==np.min(dist[index_visited])][0]

      visited[u] == False
      index_visited = np.delete(index_visited,np.where(index_visited==u))

      for v in self.Hgraph.graph[tuple(self.lin2sub(u)[::-1])]:
        lin_v = self.sub2lin(v)
        if lin_v in index_visited and (dist[u] + 1 < dist[lin_v]) :
          dist[lin_v] = dist[u] + 1
          prev[lin_v] = u

        if lin_v==stop:
          return int(dist[lin_v])
      
    return -1







