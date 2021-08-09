import Graph
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import time
import pandas as pd

from Tool_MyMaze import *
from MyMaze import *


m = MyMaze('maze-sample-20x20-v0',reward="human")
path = "./Q-Table/qtable1_20x20.npy"
q_table = np.load(open(path,'rb'))
m.set_optimal_policy(q_table)
m.set_reward([])

measure = Metrics(m,q_table,0,0,0)

metric_graph = Graph.Graph()

maze_graph = measure.get_graph_lin()

for key in maze_graph:
	metric_graph.add_node(key)

	for val in maze_graph[key]:
		metric_graph.add_connection([key,val,1])



print(metric_graph.dijkstra_distance(0,399))
traj_opti = metric_graph.dijkstra_optimal_connections(0,399)

traj_map = np.zeros((m.maze_size,m.maze_size))

for key in traj_opti:

	sub_state = [key//m.maze_size,key%m.maze_size]
	print(sub_state)
	traj_map[tuple(sub_state)] +=1


plt.figure()
plt.imshow(np.transpose(traj_map))
plt.show()
