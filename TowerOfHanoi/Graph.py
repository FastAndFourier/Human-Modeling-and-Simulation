import numpy as np
import itertools

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
          elif v[index] not in v[:index] and v1[index] not in v1[:index]: 
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

    
