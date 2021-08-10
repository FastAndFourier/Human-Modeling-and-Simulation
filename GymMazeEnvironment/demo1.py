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

import pandas as pd



if __name__ == "__main__":

	fig_V = plt.figure()
	ax_V = fig_V.gca()

	fig_traj = plt.figure()
	ax_traj = fig_traj.gca()

	fig_V1 = plt.figure()
	ax_V1 = fig_V1.gca()

	fig_traj1 = plt.figure()
	ax_traj1 = fig_traj1.gca()

	
	m1 = MyMaze1('maze-sample-20x20-v0',reward="human")
	path = "./Q-Table/qtable1_20x20.npy"
	q_table = np.load(open(path,'rb'))
	m1.env.reset()
	m1.set_optimal_policy()
	m1.set_reward(np.array([[0,3],[1,2]]))


	beta = 2

	vi = m1.prospect_bias(2,beta)
	m1.plot_v_value(fig_V,ax_V,vi,"")
	m1.plot_traj(fig_traj,ax_traj,vi,20,1000,"","softmax",beta)

	vi1 = m1.prospect_bias(50,beta)
	m1.plot_v_value(fig_V1,ax_V1,vi1,"")
	m1.plot_traj(fig_traj1,ax_traj1,vi1,20,1000,"","softmax",beta)

	plt.show()

	