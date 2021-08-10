import Graph
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import time
import pandas as pd

from Tool_MyMaze import *
from MyMaze1 import *


m = MyMaze1('maze-sample-20x20-v0',reward="human")
path = "./Q-Table/qtable1_20x20.npy"
q_table = np.load(open(path,'rb'))
m.set_optimal_policy()
m.set_reward([])


measure = Metrics(m,q_table,1,500,2)

# metric_graph = Graph.Graph()

# maze_graph = measure.get_graph_lin()

# for key in range(m.maze_size*m.maze_size):#maze_graph:
# 	metric_graph.add_node(key)

# 	for val in maze_graph[key]:
# 		metric_graph.add_connection([key,val,1])

# print(metric_graph.get_connections())

optimal_connections = m.metric_graph.dijkstra_optimal_connections(0,399)
print(optimal_connections)


traj_opti = measure.get_optimal_path(q_table)
print(traj_opti)

traj_map = np.zeros((m.maze_size,m.maze_size))

for key in traj_opti:
	traj_map[tuple(key)] +=1


traj_opti_lin = []
for step in traj_opti:
	traj_opti_lin.append(step[0]*m.maze_size + step[1])


v_boltz = m.boltz_rational(1)
traj_boltz = m.generate_traj_v(v_boltz,"softmax",1,500)[1]
print(traj_boltz)

traj_boltz_lin = []

for step in traj_boltz:
	traj_boltz_lin.append(step[0]*m.maze_size + step[1])




start = time.time()
dist_felix = m.metric_graph.dtw(traj_opti_lin,traj_boltz_lin)
print("DTW Felix:",time.time()-start,"secondes")
#print("Distance dtw:",dist_felix)

start = time.time()
matrice_distance = measure.distance_matrix_dijstra(measure.optimal_path,traj_boltz)
dist_louis = distanceDTW(measure.optimal_path,traj_boltz,matrice_distance)
print("DTW Louis:",time.time()-start,"secondes")
#print("Distance dtw:",dist_louis)

plt.figure()
plt.imshow(np.transpose(traj_map))
plt.show()
