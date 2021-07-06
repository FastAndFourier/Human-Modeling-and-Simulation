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

	#fig_policy = plt.figure()
	#ax_policy = fig_policy.gca()
	fig_V = plt.figure()
	ax_V = fig_V.gca()
	fig_V1 = plt.figure()
	ax_V1 = fig_V1.gca()
	plt.ion()
	plt.pause(0.2)



	
	m = MyMaze('maze-sample-20x20-v0')


	path = "./Q-Table/qtable1_20x20.npy"
	#q_table = m.q_learning()
	#np.save(path,q_table)
	q_table = np.load(open(path,'rb'))
	m.set_optimal_policy(q_table)

	v_from_q = m.v_from_q(q_table)
	# v_boltz0 = m.boltz_rational(0)
	v_boltz0 = m.boltz_rational(0)
	v_boltz5 = m.boltz_rational(5)
	# v_myopic099 = m.myopic_discount(0.99)
	# v_myopic01 = m.myopic_discount(0.1)	


	#m.generate_traj_v(v_boltz5)




	_, walls_list = m.edges_and_walls_list_extractor()
	maze_size = m.maze_size

	ax_policy.set_xlim(-1.5,maze_size+0.5)
	ax_policy.set_ylim(maze_size+0.5,-1.5)
	ax_policy.set_aspect('equal')
	
	ax_V.set_xlim(-1,maze_size)
	ax_V.set_ylim(maze_size,-1)
	ax_V.set_aspect('equal')

	ax_V1.set_xlim(-1,maze_size)
	ax_V1.set_ylim(maze_size,-1)
	ax_V1.set_aspect('equal')
	
	
	value_table = v_boltz0
	value_table1 = v_boltz5
	operator = "softmax"
	
	
	
	# for i in range(maze_size):
	# 	for j in range(maze_size):
	# 		if ([i,j]==[maze_size-1,maze_size-1]):
	# 			break
	# 		action = m.select_action_from_v([i,j],value_table,"human",operator)[0]

	# 		if action==0:
	# 			ax_policy.quiver(i,j,0,.75,color='c')
	# 		if action==1:
	# 			ax_policy.quiver(i,j,0,-.75,color='c')
	# 		if action==2:
	# 			ax_policy.quiver(i,j,.75,0,color='c')
	# 		if action==3:
	# 			ax_policy.quiver(i,j,-.75,0,color='c')


	traj = np.zeros((m.maze_size,m.maze_size),dtype=int)
	total_length = []
	for epoch in tqdm(range(100)):
		m.env.reset()
		state = [0,0]
		traj[tuple(state)]+=1
		length = 0
		
		while (m.env.state!=m.env.observation_space.high).any():
			action = m.select_action_from_v(state,value_table,"human",softmax)[0]
			new_s,reward,done,_ = m.env.step(int(action))
			state = new_s
			traj[tuple(state)]+=1
			length+=1
		total_length.append(length)

	fig_V.suptitle("Mean demonstration length = "+str(int(np.array(total_length).mean())))

	#Draw value table
	im = ax_V.imshow(np.transpose(traj.reshape(maze_size,maze_size)))
	for state in range(0,m.maze_size*m.maze_size):
		i=state//maze_size
		j=state%maze_size
		text = ax_V.text(i,j, str(traj[i,j])[0:4],ha="center", va="center", color="black")


	##########################################################################################

	traj = np.zeros((m.maze_size,m.maze_size),dtype=int)
	total_length = []
	for epoch in tqdm(range(100)):
		m.env.reset()
		state = [0,0]
		traj[tuple(state)]+=1
		length = 0
		
		while (m.env.state!=m.env.observation_space.high).any():
			action = m.select_action_from_v(state,value_table1,"human","softmax")[0]
			new_s,reward,done,_ = m.env.step(int(action))
			state = new_s
			traj[tuple(state)]+=1
			length+=1
		total_length.append(length)

	fig_V1.suptitle("Mean demonstration length = "+str(int(np.array(total_length).mean())))

   
	
	#Draw value table
	im = ax_V1.imshow(np.transpose(traj.reshape(maze_size,maze_size)))
	#if maze_size<=20:
	for state in range(0,m.maze_size*m.maze_size):
		i=state//maze_size
		j=state%maze_size
		text = ax_V1.text(i,j, str(traj[i,j])[0:4],ha="center", va="center", color="black")
	


	##########################################################################################
	
	# # draw start and end position
	plot_start_marker = ax_policy.scatter(0,0, marker="o", s=100,c="b") # s = #size_of_the_marker#
	plot_end_marker = ax_policy.scatter(maze_size-1,maze_size-1, marker="o", s=100,c="r")

	#draw maze walls
	for i in walls_list:
		ax_policy.add_line(mlines.Line2D([i[1][0]-0.5,i[1][1]-0.5],[i[0][0]-0.5,i[0][1]-0.5],color='k'))
		ax_V.add_line(mlines.Line2D([i[1][0]-0.5,i[1][1]-0.5],[i[0][0]-0.5,i[0][1]-0.5],color='k'))
	
	# add east and top walls
	for i in range(0,maze_size):
		ax_policy.add_line(mlines.Line2D([-0.5,-0.5],[i-0.5,i+0.5],color='k'))
		ax_policy.add_line(mlines.Line2D([i-0.5,i+0.5],[-0.5,-0.5],color='k'))
		ax_policy.add_line(mlines.Line2D([maze_size-0.5,maze_size-0.5],[i-0.5,i+0.5],color='k'))
		ax_policy.add_line(mlines.Line2D([i-0.5,i+0.5],[maze_size-0.5,maze_size-0.5],color='k'))
		ax_V.add_line(mlines.Line2D([-0.5,-0.5],[i-0.5,i+0.5],color='k'))
		ax_V.add_line(mlines.Line2D([i-0.5,i+0.5],[-0.5,-0.5],color='k'))
		ax_V.add_line(mlines.Line2D([maze_size-0.5,maze_size-0.5],[i-0.5,i+0.5],color='k'))
		ax_V.add_line(mlines.Line2D([i-0.5,i+0.5],[maze_size-0.5,maze_size-0.5],color='k'))

	
	plt.ioff()
	plt.show()
