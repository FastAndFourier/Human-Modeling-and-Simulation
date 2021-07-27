import numpy as np
import itertools
import time

def hamming(s1,s2):
  distance = 0
  index = []
  for x,(i,j) in enumerate(zip(s1,s2)):
    if i!=j:
      distance += 1
      index.append(x)
  return distance,index


class Node:

  def __init__(self,id,edge):
    self.id = id
    self.edge = edge

  def print_node(self):
    print("{",self.id,",",self.edge,"}")

class HanoiGraph():

  def __init__(self,n):

    self.nb_disk = n
    self.vertex = list(itertools.product([0,1,2],repeat=self.nb_disk))
    self.edge = self.set_edge()
    self.graph = self.build_hanoi_graph()


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

  def set_edge(self):

    edge = []

    for v in self.vertex:
      e = []
      for v1 in self.vertex:
        distance = hamming(v,v1)
        if distance[0]==1:
          index = distance[1][0]

          # Move the smallest disk
          if index==0: 
            e.append(v1)
                      
          # Cdt: the disk must be unstackable AND stackable
          elif (v[index] not in v[:index]) and (v1[index] not in v1[:index]): 
            e.append(v1)

        else:
          pass
      edge.append(e)


    return edge


  def build_hanoi_graph(self):
    graph = {}
    for i,e in enumerate(self.vertex):
      graph[e] = self.edge[i]

    return graph

  def print_env(self,state):

    pole = np.zeros([self.nb_disk,3],dtype=int)
    state = np.array([int(s) for s in state])
    disk_num = self.nb_disk

    for disk in state[::-1]:
      row = np.argwhere(pole[:,disk]==0)[0] 
      pole[row,disk] = disk_num
      disk_num -= 1

    pole = np.flipud(pole)

    for k in range(self.nb_disk):
      for l in range(3):
        if pole[k,l] != 0:
          print(pole[k,l],end="\t")
        else:
          print("|",end="\t")

      print("\n")


  def distance_state(self,start,end):

    if tuple(start)==tuple(end):
      return 0

    nb_state = len(self.vertex)

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

      for v in self.graph[tuple(self.lin2sub(u))]:
        lin_v = self.sub2lin(v)
        if lin_v in index_visited and (dist[u] + 1 < dist[lin_v]) :
          dist[lin_v] = dist[u] + 1
          prev[lin_v] = u

        if lin_v==stop:
          return dist[lin_v]
      
    return -1

    
