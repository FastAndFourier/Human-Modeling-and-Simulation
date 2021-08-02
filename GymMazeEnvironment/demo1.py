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


# def compare_traj_map(maze,maze1,optimal_traj,v_bias,beta,nb_demo):
	

# 	# Trajectory map optimal demonstration
# 	optimal_traj_map = np.zeros((maze.maze_size,maze.maze_size))
# 	for i in optimal_traj:
# 		optimal_traj_map[tuple(i)] += 1

# 	optimal_traj_map = np.transpose(optimal_traj_map)

# 	plt.figure()
# 	plt.imshow(optimal_traj_map)


# 	# Optimal trajectory 2D state to linear state
# 	lin_optimal_traj = []
# 	for t in optimal_traj:
# 		lin_optimal_traj.append(t[0]*maze.maze_size+t[1])


# 	# Biased trajectories map and linear states
# 	total_lin_traj = []
# 	traj_map = np.zeros((maze1.maze_size,maze1.maze_size))

# 	for k in range(nb_demo):
# 		traj = maze1.generate_traj_v(v_bias,"softmax",beta)[1]
# 		for t in traj:
# 			lin_t = t[0]*maze1.maze_size + t[1]			
# 			total_lin_traj.append(lin_t)
# 			traj_map[tuple(t)] += 1
	
# 	traj_map = np.transpose(traj_map)

# 	plt.figure()
# 	plt.imshow(traj_map)




# 	# Trajectories comparaison
# 	diff_map = np.zeros((maze.maze_size,maze.maze_size))
# 	for i in total_lin_traj:

# 		if i not in lin_optimal_traj:
# 			diff_map[i//maze.maze_size,i%maze.maze_size] =  -total_lin_traj.count(i)

# 	for i in lin_optimal_traj:
# 		diff_map[i//maze.maze_size,i%maze.maze_size] = 2
# 		if i not in total_lin_traj:
# 			diff_map[i//maze.maze_size,i%maze.maze_size] = 3

# 	diff_map = np.transpose(diff_map)
# 	fig = plt.figure()
# 	ax = fig.gca()
# 	ax.set_xlim(-1,maze.maze_size)
# 	ax.set_ylim(maze.maze_size,-1)
# 	ax.set_aspect('equal')

# 	color = []
# 	sorted_it = np.sort(diff_map[diff_map<0])

# 	boundaries = []

# 	if sorted_it.size > 0:
# 		bound = [sorted_it[len(sorted_it)*k//3] for k in range(3)]
# 		boundaries = bound	
# 		color.append((0,0,1))
# 		color.append((0,0.4,1))
# 		color.append((0.5,0.7,1))	
	
# 	boundaries.extend([0,2,2.1,3])
	
# 	color.append((1,1,1)) # White : normal tiles
# 	color.append((0,1,0)) # Green : common tiles
# 	color.append((1,0,0)) # Red : only optimal traj's tiles

# 	cmap = colors.LinearSegmentedColormap.from_list('my_cmap',color,N=len(color))
# 	norm = colors.BoundaryNorm(boundaries,cmap.N,clip=True)
# 	# print(norm)
# 	ax.imshow(diff_map,cmap=cmap,norm=norm)
# 	print(diff_map)
	

# 	_, walls_list = maze.edges_and_walls_list_extractor()

# 	for i in walls_list:
# 		ax.add_line(mlines.Line2D([i[1][0]-0.5,i[1][1]-0.5],[i[0][0]-0.5,i[0][1]-0.5],color='k'))

# 	for i in range(0,maze.maze_size):
# 		ax.add_line(mlines.Line2D([-0.5,-0.5],[i-0.5,i+0.5],color='k'))
# 		ax.add_line(mlines.Line2D([i-0.5,i+0.5],[-0.5,-0.5],color='k'))
# 		ax.add_line(mlines.Line2D([maze.maze_size-0.5,maze.maze_size-0.5],[i-0.5,i+0.5],color='k'))
# 		ax.add_line(mlines.Line2D([i-0.5,i+0.5],[maze.maze_size-0.5,maze.maze_size-0.5],color='k'))

	

	
# 	# plt.figure()
# 	# plt.imshow(diff_map,cmap=cmap)


def display_entropy(m,v_bias,beta,nb_traj):

	entropy_traj = np.zeros((m.maze_size,m.maze_size))

	for e in range(nb_traj):
		state = m.env.env.reset(np.array([0,0]))
		done = False

		while not(done):

			action,h = m.select_action_from_v(state,v_bias,m.reward_type,"softmax",beta) 
			entropy_traj[tuple(state)] += h
			new_state, reward, done, _ = m.env.step(int(action))
			state = new_state


	entropy_traj = entropy_traj/nb_traj
	viridis = cm.get_cmap('viridis',128)
	newcolor = viridis(np.linspace(0,1,128))
	newcolor[:1,:] = np.array([1,1,1,1])

	cmap = colors.ListedColormap(newcolor)


	entropy_traj = np.transpose(entropy_traj)

	fig = plt.figure()
	ax = fig.gca()

	im = ax.imshow(entropy_traj,cmap=cmap)
	fig.colorbar(im)

	_, walls_list = m.edges_and_walls_list_extractor()

	for i in walls_list:
		ax.add_line(mlines.Line2D([i[1][0]-0.5,i[1][1]-0.5],[i[0][0]-0.5,i[0][1]-0.5],color='k'))

	for i in range(0,m.maze_size):
		ax.add_line(mlines.Line2D([-0.5,-0.5],[i-0.5,i+0.5],color='k'))
		ax.add_line(mlines.Line2D([i-0.5,i+0.5],[-0.5,-0.5],color='k'))
		ax.add_line(mlines.Line2D([m.maze_size-0.5,m.maze_size-0.5],[i-0.5,i+0.5],color='k'))
		ax.add_line(mlines.Line2D([i-0.5,i+0.5],[m.maze_size-0.5,m.maze_size-0.5],color='k'))



if __name__ == "__main__":

	# fig_policy = plt.figure()
	# ax_policy = fig_policy.gca()

	# fig_V = plt.figure()
	# ax_V = fig_V.gca()

	# fig_traj = plt.figure()
	# ax_traj = fig_traj.gca()


	# fig_policy1 = plt.figure()
	# ax_policy1 = fig_policy1.gca()

	# fig_V1 = plt.figure()
	# ax_V1 = fig_V1.gca()

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



	beta1 = 2
	beta2 = 100

	
	
	measure = Metrics(m=m,qtab=q_table,nb_traj=20,max_step=500,beta=2)
	optimal_traj = measure.get_optimal_path(q_table)

	v_table0 = m_obstacle.prospect_bias(2,beta1)
	v_table1 = m.boltz_rational(beta2)

	display_entropy(m,v_table0,beta1,500)
	#display_entropy(m_obstacle,v_table1,beta2,500)

	operator = "softmax"


	plt.figure()
	plt.imshow(m.get_entropy_map_v(v_table1,measure.beta_actor))



	
	



	# measure.compare_traj_map(m,optimal_traj,v_table0,measure.beta_actor,500,False)
	# measure.compare_traj_map(m,optimal_traj,v_table1,measure.beta_actor,500,False)

	# m.plot_v_value(fig_V,ax_V,v_table0,"")
	# #plot_policy(fig_policy,ax_policy,m,v_table0,"","argmax")
	# m.plot_traj(fig_traj,ax_traj,v_table0,1,1000,"Extremal alpha 0 beta = "+str(beta1),operator,beta1) #m.v_from_q(q_table)

	# m.plot_v_value(fig_V1,ax_V1,v_table1,"")
	#plot_policy(fig_policy1,ax_policy1,m,v_table1,"",operator)
	#ax_policy.scatter(10,m.maze_size-8, marker="o", s=100,c="g")
	#m.plot_traj(fig_traj1,ax_traj1,m.v_from_q(q_table),1,1000,"Extremal alpha 1 beta = "+str(beta2),"argmax",beta2)
	#m.plot_traj(fig_traj1,ax_traj1,v_table1,1,1000,"Extremal alpha 1 beta = "+str(beta2),operator,beta2)
	
	
	
	plt.ioff()
	plt.show()
