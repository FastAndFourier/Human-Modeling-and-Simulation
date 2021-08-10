import Graph
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import time
import pandas as pd

from Tool_MyMaze import *
from MyMaze import *



def pointwise_distance(traj1,traj2):

	distance = 0
	for it in range(len(traj1)):
		if traj1[it]!=traj2[it]:
			distance+=1

	return distance

def dfs(sub_graph,path,optimal_path=[]):   
    
    last_step = path[-1]              
    
    if last_step in sub_graph:
        for vertex in sub_graph[last_step]:
            new_path = path + [vertex]
            optimal_path = dfs(sub_graph,new_path,optimal_path)
    else:
        optimal_path += [path]
    
    return optimal_path


def generate_lin_traj(m,v,b,nb):

	total_traj_lin = []
	total_traj = []

	for e in range(nb):

		traj_boltz = m.generate_traj_v(v,"softmax",b,500)[1]
		

		traj_boltz_lin = []

		for step in traj_boltz:
			traj_boltz_lin.append(step[0]*m.maze_size + step[1])

		total_traj.append(traj_boltz)
		total_traj_lin.append(traj_boltz_lin)

	if nb==1:
		return total_traj[0], total_traj_lin[0]
	else:
		return total_traj, total_traj_lin


if __name__ == "__main__":

	m = MyMaze1('maze-sample-20x20-v0',reward="human")
	path = "./Q-Table/qtable1_20x20.npy"
	q_table = np.load(open(path,'rb'))
	m.set_optimal_policy()
	m.set_reward([])


	measure = Metrics(m,q_table,1,500,2)

	optimal_connections = m.metric_graph.dijkstra_optimal_connections(0,399)
	print(optimal_connections)

	paths = dfs(optimal_connections,[0],[])
	


	traj_map = np.zeros((m.maze_size,m.maze_size))
	for p in paths:
		for state in p:
			traj_map[tuple(state)] +=1

	

	# plt.colorbar(im)
	plt.figure()
	im = plt.imshow(np.transpose(traj_map))
	plt.show()

	traj_opti = measure.get_optimal_path(q_table)

	

	traj_opti_lin = []
	for step in traj_opti:
		traj_opti_lin.append(step[0]*m.maze_size + step[1])


	beta = 100

	v_boltz = m.boltz_rational(beta)
	traj_boltz, traj_boltz_lin = generate_lin_traj(m,v_boltz,beta,10)

	

	k = 0

	start = time.time()
	# for traj_opti_lin in paths:
	dist_felix = m.metric_graph.dtw(traj_opti_lin,traj_boltz_lin[0])
	# print(k)
	# k+=1

	print("DTW Felix:",time.time()-start,"secondes")

	# plt.figure()
	# plt.imshow(np.transpose(traj_map))
	while True:
		m.env.render()
