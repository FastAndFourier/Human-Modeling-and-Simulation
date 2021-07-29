import gym
import gym_maze
from gym_maze.envs import MazeEnvSample3x3, MazeEnvSample5x5, MazeEnvSample10x10, MazeEnvSample100x100,MazeEnvSample20x20
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import time

from maze_function import *
from MyMaze import *




if __name__ == "__main__":

	# fig_policy = plt.figure()
	# ax_policy = fig_policy.gca()

	fig_V = plt.figure()
	ax_V = fig_V.gca()

	fig_traj = plt.figure()
	ax_traj = fig_traj.gca()


	# fig_policy1 = plt.figure()
	# ax_policy1 = fig_policy1.gca()

	fig_V1 = plt.figure()
	ax_V1 = fig_V1.gca()

	fig_traj1 = plt.figure()
	ax_traj1 = fig_traj1.gca()

	# fig_compar1 = plt.figure()
	# ax_compar1 = fig_compar1.gca()

	# fig_compar2 = plt.figure()
	# ax_compar2 = fig_compar2.gca()


	# fig_policy1 = plt.figure()
	# ax_policy1 = fig_policy1.gca()
	# fig_V1 = plt.figure()
	# ax_V1 = fig_V1.gca()

	


	
	m = MyMaze('maze-sample-20x20-v0',reward="human")
	path = "./Q-Table/qtable1_20x20.npy"

	
	

	#q_table = m.q_learning()
	#np.save(path,q_table)
	q_table = np.load(open(path,'rb'))
	m.set_optimal_policy(q_table)
	obstacle = [] #np.array([[0,3],[1,0],[15,17]])
	m.set_reward(obstacle)

	m_obstacle = MyMaze('maze-sample-20x20-v0',reward="human")
	m_obstacle.set_optimal_policy(q_table)
	obstacle = np.array([[3,6],[1,4]])
	m_obstacle.set_reward(obstacle)


	plt.ion()

	beta1 = 2
	beta2 = 2

	v_table0 = m.myopic_value_iteration(20,beta1)


	# h_map = plt.figure()
	# h_map_ax = h_map.gca()
	# entropy = m.get_entropy_map_q(q_table)
	# im_h = h_map_ax.imshow(np.transpose(np.exp(entropy)))
	# h_map.colorbar(im_h)
	# print(entropy)

	v_table1 = m.myopic_value_iteration(200,beta2)

	operator = "softmax"

	# traj_opti = m.generate_traj_v(m.v_from_q(q_table),operator)[1]
	# traj0 = m_obstacle.generate_traj_v(v_table0,operator)[1]
	# traj1 = m_obstacle.generate_traj_v(v_table1,operator)[1]


	
	m.plot_v_value(fig_V,ax_V,v_table0,"")
	#plot_policy(fig_policy,ax_policy,m,v_table0,"","argmax")
	m.plot_traj(fig_traj,ax_traj,v_table0,1,1000,"Extremal alpha 0 beta = "+str(beta1),operator,beta1) #m.v_from_q(q_table)

	m.plot_v_value(fig_V1,ax_V1,v_table1,"")
	#plot_policy(fig_policy1,ax_policy1,m,v_table1,"",operator)
	#ax_policy.scatter(10,m.maze_size-8, marker="o", s=100,c="g")
	m.plot_traj(fig_traj1,ax_traj1,v_table1,1,1000,"Extremal alpha 1 beta = "+str(beta2),operator,beta2)
	
	
	
	plt.ioff()
	plt.show()
