import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import time
import pandas as pd

from maze_function import *
from MyMaze import *


#def DTW(traj1,traj2)


def evaluate_v(maze,v_vector,nb_traj,max_step,operator):

	traj = np.zeros((maze.maze_size,maze.maze_size),dtype=int)
	total_length = []

	for epoch in tqdm(range(nb_traj)):
	    maze.env.reset()
	    state = [0,0]
	    traj[tuple(state)]+=1
	    length = 0
	    
	    while (maze.env.state!=maze.env.observation_space.high).any() and length < max_step:
	        action = maze.select_action_from_v(state,v_vector,"human",operator)[0]
	        new_s,reward,done,_ = maze.env.step(int(action))
	        state = new_s
	        traj[tuple(state)]+=1
	        length+=1
	    total_length.append(length)

	return int(np.array(total_length).mean()),int(np.array(total_length).std())


	
m = MyMaze('maze-sample-20x20-v0')


path = "./Q-Table/qtable1_20x20.npy"
#q_table = m.q_learning()
#np.save(path,q_table)
q_table = np.load(open(path,'rb'))
m.set_optimal_policy(q_table)

obstacle = np.array([[0,3],[1,0],[15,17]])

m.set_reward(obstacle)



v_table_optimal = m.v_from_q(q_table)
v_boltz0 = m.boltz_rational(0)
v_boltz10 = m.boltz_rational(10)
v_prospect_bias1 = m.prospect_bias(1)
v_prospect_bias10 = m.prospect_bias(10)
v_extremal0 = m.extremal(0)
v_extremal05 = m.extremal(0.5)
v_extremal09 = m.extremal(0.9)

operator = "softmax"

res = []

res.append(evaluate_v(m,v_table_optimal,100,1000,operator))
res.append(evaluate_v(m,v_boltz0,100,1000,operator))
res.append(evaluate_v(m,v_boltz10,100,1000,operator))
res.append(evaluate_v(m,v_prospect_bias1,100,1000,operator))
res.append(evaluate_v(m,v_prospect_bias10,100,1000,operator))
res.append(evaluate_v(m,v_extremal0,100,1000,operator))
res.append(evaluate_v(m,v_extremal05,100,1000,operator))
res.append(evaluate_v(m,v_extremal09,100,1000,operator))

mean_ = []
std_ = []

for k in range(len(res)):
	mean_.append(res[k][0])
	std_.append(res[k][1])

data = {"mean":mean_,"std":std_}
results = pd.DataFrame(data=data)
results.index = ["Optimal","Boltzmann (0)","Boltzmann (10)","Prospect bias (0)","Prospect bias (10)","Extremal (0)","Extremal (0.5)","Extremal (0.9)"]

print(results)

	

