import numpy as np
from MyMaze import *

def DTW(traj1,traj2):

	print(len(traj1),len(traj2))
	time.sleep(0.1)
	if len(traj1)==0 and len(traj2)==0:
		print("In")
		return 0
	elif len(traj1)==0 or len(traj2)==0:
		return 1000
	else:

		head1 = traj1[0]
		head2 = traj2[0]

		rest1 = traj1[1:]
		rest2 = traj2[1:]
 
		return np.sqrt((head1[0]-head2[0])**2+(head1[1]-head2[1])**2) + min(DTW(traj1,rest2),DTW(rest1,traj2),DTW(rest1,rest2))


maze = MyMaze('maze-sample-20x20-v0')


path = "./Q-Table/qtable1_20x20.npy"
q_table = np.load(open(path,'rb'))
maze.set_optimal_policy(q_table)
maze.set_reward()

v1 = maze.v_from_q(q_table)
traj1 = []

state = maze.env.reset()

while (state!=maze.env.observation_space.high).any():

	traj1.append(list(state))
	action = maze.select_action_from_v(state,v1,"human","softmax")[0]
	new_s,reward,done,_ = maze.env.step(int(action))
	state = new_s


v2 = maze.prospect_bias(5)
traj2 = []

state = maze.env.reset()

while (state!=maze.env.observation_space.high).any():

	traj2.append(list(state))
	action = maze.select_action_from_v(state,v2,"human","softmax")[0]
	new_s,reward,done,_ = maze.env.step(int(action))
	state = new_s

print(traj1)
print(traj2)
print(DTW(traj1,traj2))