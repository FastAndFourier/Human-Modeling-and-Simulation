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




	# fig_policy1 = plt.figure()
	# ax_policy1 = fig_policy1.gca()
	# fig_V1 = plt.figure()
	# ax_V1 = fig_V1.gca()

	


	
	m = MyMaze('maze-sample-20x20-v0')


	path = "./Q-Table/qtable1_20x20.npy"
	#q_table = m.q_learning()
	#np.save(path,q_table)
	q_table = np.load(open(path,'rb'))
	m.set_optimal_policy(q_table)

	# h = m.get_entropy_map(q_table)

	# fig_h = plt.figure()
	# ax_h = fig_h.gca()
	# ax_h.imshow(h)

	plt.ion()
	plt.pause(0.2)


	#vi_vector = m.value_iteration()
	#print(vi_vector)
	#v_from_q = m.v_from_q(q_table)


	# v_boltz0 = m.boltz_rational(0.1)
	# v_boltz1 = m.boltz_rational(5)
	
	# v_prospect0 = m.prospect_bias(0.1)
	# v_prospect1 = m.prospect_bias(10)

	# v_myopic099 = m.myopic_discount(0.99)
	# v_myopic01 = m.myopic_discount(0.3)	

	# m.generate_traj_v(v_boltz01,operator)
	# m.generate_traj_v(v_boltz1,operator)

	#m.local_uncertainty([10,m.maze_size-8],4)
	#m.hyperbolic_discount(0)


	# diff_myopic = []
	# diff_hyper = []
	# disc = [0.1,0.25,0.50,0.75,0.8,0.9,0.95,0.99]
	# for d in disc:
	# 	print(d)
	# 	v_table0 = m.myopic_discount(d)
	# 	v_table1 = m.hyperbolic_discount(d)
	# 	diff_myopic.append(abs(v_table0[0,0]-v_table0[m.maze_size-1,m.maze_size-1]))
	# 	diff_hyper.append(abs(v_table1[0,0]-v_table1[m.maze_size-1,m.maze_size-1]))


	#m.local_uncertainty([1,1],3)#([10,m.maze_size-8],4)

	v_table0 = m.value_iteration()
	v_table1 = m.illusion_of_control(0.1)
	operator = "softmax"

	#m.generate_traj_v(v_table0,operator)
	#m.generate_traj_v(v_table1,operator)
	
	plot_v_value(fig_V,ax_V,m,v_table0,"")
	#plot_policy(fig_policy,ax_policy,m,v_table0,"","argmax")
	#ax_policy.scatter(10,m.maze_size-8, marker="o", s=100,c="g")
	plot_traj(fig_traj,ax_traj,m,v_table0,100,1000,"",operator)

	plot_v_value(fig_V1,ax_V1,m,v_table1,"")
	#plot_policy(fig_policy1,ax_policy1,m,v_table1,"",operator)
	plot_traj(fig_traj1,ax_traj1,m,v_table1,100,1000,"",operator)
	
	
	
	plt.ioff()
	plt.show()
