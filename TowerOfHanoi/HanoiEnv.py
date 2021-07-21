import numpy as np
from Graph import HanoiGraph
import random
import pygame

MAX_EPISODE = 5000

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
  
  
  
  def __init__(self,n,epsi=0.02,disc=0.99,lr=0.2):

    self.nb_disk = n
    self.Hgraph = HanoiGraph(n)
    self.state = tuple([0]*self.nb_disk)
    self.epsilon = epsi
    self.discount = disc
    self.lr = lr
    self.action_space = [[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]]
    self.final_state = tuple([0]*self.nb_disk)
    self.MAX_STEP = self.nb_disk*((2**self.nb_disk-1)-1)
    
    pygame.init()
    self.win = pygame.display.set_mode()
    pygame.display.set_caption("Tower of Hanoi")


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


  def reset(self,state):
    self.state = state
    return self.state

  def step(self,action):   

    if type(action)==int:
      action = self.action_space[action]

    if action[0] not in list(self.state): # No disk around the giving pole
      new_state = self.state
      reward = -1/self.nb_disk
      done = False
      #print("No disk in th giving pole")
    else:
      disk_to_move = np.argwhere(np.array(self.state)==action[0])[0][0]
      temp_state = np.copy(self.state)
      temp_state[disk_to_move] = action[1]
      temp_state = tuple(temp_state)

      #print(self.Hgraph.graph[tuple(self.state)])
      if temp_state in tuple(self.Hgraph.graph[tuple(self.state)]):
        new_state = temp_state
        done = (new_state == tuple([2]*self.nb_disk))
        if done:
          reward = 5
        else:
          reward = -1/(self.nb_disk**2)
      else:
        new_state = self.state
        reward = -1/self.nb_disk
        done = 0

    self.state = new_state
    return [new_state, reward, done]


  def render(self):
    self.Hgraph.print_env(self.state)


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

        state = tuple(self.reset([0]*self.nb_disk))
        
        # if e%1000==0:
        #     print("Episode #",e,"(",reach,")")
        for k in range(self.MAX_STEP):

            action = self.select_action(state,q_table)
            #print("Before stepping",self.state,action)
            new_s, reward, done = self.step(action)
            new_s = tuple(new_s)
            self.update_q(q_table,action,state,new_s,reward,done)
            
            #print("After stepping",self.state,new_s)
            state = new_s

            # env.render()
            #time.sleep(1)
            # clear_output()

            if done :
              nb_done+=1
              break

        if not(e%500):
          print("Episodes ",e,"(",nb_done,")")
          nb_done = 0
        

    return  q_table


    def render(self):
