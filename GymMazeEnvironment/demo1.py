import gym
import gym_maze
from gym_maze.envs import MazeEnvSample3x3, MazeEnvSample5x5, MazeEnvSample10x10, MazeEnvSample100x100,MazeEnvSample20x20
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm
from tqdm import tqdm
import time
from matplotlib import colors
from colorsys import hsv_to_rgb

from Tool_MyMaze import *
from MyMaze import *
from MyMaze1 import *

import pandas as pd





# def display_entropy(m,v_bias,beta,nb_traj):

# 	entropy_traj = np.zeros((m.maze_size,m.maze_size))

# 	for e in range(nb_traj):
# 		state = m.env.env.reset(np.array([0,0]))
# 		done = False

# 		while not(done):

# 			action,h = m.select_action_from_v(state,v_bias,m.reward_type,"softmax",beta) 
# 			entropy_traj[tuple(state)] += h
# 			new_state, reward, done, _ = m.env.step(int(action))
# 			state = new_state


# 	entropy_traj = entropy_traj/nb_traj
# 	viridis = cm.get_cmap('viridis',128)
# 	newcolor = viridis(np.linspace(0,1,128))
# 	newcolor[:1,:] = np.array([1,1,1,1])

# 	cmap = colors.ListedColormap(newcolor)


# 	entropy_traj = np.transpose(entropy_traj)

# 	fig = plt.figure()
# 	ax = fig.gca()

# 	im = ax.imshow(entropy_traj,cmap=cmap)
# 	fig.colorbar(im)

# 	_, walls_list = m.edges_and_walls_list_extractor()

# 	for i in walls_list:
# 		ax.add_line(mlines.Line2D([i[1][0]-0.5,i[1][1]-0.5],[i[0][0]-0.5,i[0][1]-0.5],color='k'))

# 	for i in range(0,m.maze_size):
# 		ax.add_line(mlines.Line2D([-0.5,-0.5],[i-0.5,i+0.5],color='k'))
# 		ax.add_line(mlines.Line2D([i-0.5,i+0.5],[-0.5,-0.5],color='k'))
# 		ax.add_line(mlines.Line2D([m.maze_size-0.5,m.maze_size-0.5],[i-0.5,i+0.5],color='k'))
# 		ax.add_line(mlines.Line2D([i-0.5,i+0.5],[m.maze_size-0.5,m.maze_size-0.5],color='k'))



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

	fig_V2 = plt.figure()
	ax_V2 = fig_V2.gca()


	# fig_traj1 = plt.figure()
	# ax_traj1 = fig_traj1.gca()

	# fig_compar1 = plt.figure()
	# ax_compar1 = fig_compar1.gca()

	# fig_compar2 = plt.figure()
	# ax_compar2 = fig_compar2.gca()


	# fig_policy1 = plt.figure()
	# ax_policy1 = fig_policy1.gca()
	# fig_V1 = plt.figure()
	# ax_V1 = fig_V1.gca()

	
	m1 = MyMaze1('maze-sample-20x20-v0',reward="human")
	path = "./Q-Table/qtable1_20x20.npy"
	q_table = np.load(open(path,'rb'))
	m1.env.reset()
	m1.set_optimal_policy()
	m1.set_reward(np.array([[3,6],[1,4]]))

	#print(m1.metric_graph.get_connections())

	vi = m1.prospect_bias(50,2)
	m1.plot_v_value(fig_V,ax_V,vi,"")
	m1.plot_traj(fig_traj,ax_traj,vi,20,1000,"","softmax",2)
	plt.show()

	
	m = MyMaze('maze-sample-20x20-v0',reward="human")
	path = "./Q-Table/qtable1_20x20.npy"


	#q_table = m.q_learning()
	#np.save(path,q_table)
	q_table = np.load(open(path,'rb'))
	m.set_optimal_policy(q_table)
	obstacle = [] #np.array([[0,3],[1,0],[15,17]])
	m.set_reward(obstacle)



	m_obstacle = MyMaze('maze-sample-20x20-v0',reward="human")
	path_obs = "./Q-Table/qtable1_20x20_obs.npy"
	q_table_obs = np.load(open(path,'rb'))
	m_obstacle.set_optimal_policy(q_table_obs)
	obstacle = np.array([[3,6],[1,4]])
	m_obstacle.set_reward(obstacle)

	# plt.figure()
	# im = plt.imshow(m.get_entropy_map_v(m.value_iteration(),2))
	# plt.colorbar(im)


	#beta1 = 100
	#beta2 = 100

	title = "Boltz 100"
	
	
	measure = Metrics(m=m,qtab=q_table,nb_traj=20,max_step=500,beta=2)
	optimal_traj = measure.get_optimal_path(q_table)

	v_table0 = m.boltz_rational(1)



	# demonstration = []

	# for k in range(100):
	# 	traj = m.generate_traj_v(v_table0,"softmax",1,500)[1]
	# 	demonstration.append(traj)

	# np.save("../../demo_dagger.npy",np.array(demonstration,dtype=list),allow_pickle=True)



	#############################################################
	

	#v_table1 = m.boltz_rational(beta2)

	measure.compare_traj_map(m_obstacle,optimal_traj,v_table0,2,500,title)
	#measure.display_entropy(v_table0,measure.beta_actor,500,title)

	plt.figure()
	im = plt.imshow(m.get_entropy_map_v(v_table0,2))
	plt.colorbar(im)


	operator = "softmax"


	# plt.figure()
	# plt.imshow(m.get_entropy_map_v(v_table0,measure.beta_actor))



	
	



	# measure.compare_traj_map(m,optimal_traj,v_table0,measure.beta_actor,500,False)
	# measure.compare_traj_map(m,optimal_traj,v_table1,measure.beta_actor,500,False)

	m_obstacle.plot_v_value(fig_V,ax_V,v_table0,"")
	# #plot_policy(fig_policy,ax_policy,m,v_table0,"","argmax")
	# m.plot_traj(fig_traj,ax_traj,v_table0,1,1000,"Extremal alpha 0 beta = "+str(beta1),operator,beta1) #m.v_from_q(q_table)

	m.plot_v_value(fig_V1,ax_V1,m.prospect_bias(50,measure.beta_actor),"")
	m.plot_v_value(fig_V2,ax_V2,m.value_iteration(),"")
	#plot_policy(fig_policy1,ax_policy1,m,v_table1,"",operator)
	#ax_policy.scatter(10,m.maze_size-8, marker="o", s=100,c="g")
	#m.plot_traj(fig_traj1,ax_traj1,m.v_from_q(q_table),1,1000,"Extremal alpha 1 beta = "+str(beta2),"argmax",beta2)
	#m.plot_traj(fig_traj1,ax_traj1,v_table1,1,1000,"Extremal alpha 1 beta = "+str(beta2),operator,beta2)
	
	
	
	plt.ioff()
	plt.show()
