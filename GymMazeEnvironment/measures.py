import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import time
import pandas as pd

from maze_function import *
from MyMaze import *


def distanceDTW(traj1,traj2):

	matrice_distance = np.zeros((len(traj1),len(traj2)))
	matrice_DTW = np.zeros((len(traj1),len(traj2)))

	for i in range(len(traj1)):
		for j in range(len(traj2)):

			matrice_distance[i,j] = np.sqrt((traj1[i][0]-traj2[j][0])**2 + (traj1[i][1]-traj2[j][1])**2)

	matrice_DTW[0,0] = matrice_distance[0,0]

	for m in range(1,len(traj1)):
		matrice_DTW[m,0] = matrice_distance[m,0] + matrice_DTW[m-1,0]

	for n in range(1,len(traj2)):
		matrice_DTW[0,n] = matrice_distance[0,n] + matrice_DTW[0,n-1]


	for k in range(1,len(traj1)):
		for l in range(1,len(traj2)):
			matrice_DTW[k,l] = matrice_distance[k,l] + min(matrice_DTW[k-1,l],matrice_DTW[k,l-1],matrice_DTW[k-1,l-1])

	return matrice_DTW[-1,-1]


def distanceFrechet(traj1,traj2):

	matrice_distance = np.zeros((len(traj1),len(traj2)))
	matrice_frechet = np.zeros((len(traj1),len(traj2)))

	for i in range(len(traj1)):
		for j in range(len(traj2)):
			matrice_distance[i,j] = np.sqrt((traj1[i][0]-traj2[j][0])**2 + (traj1[i][1]-traj2[j][1])**2)

	matrice_frechet[0,0] = matrice_distance[0,0]

	for m in range(1,len(traj1)):
		matrice_frechet[m,0] = max(matrice_distance[m,0],matrice_frechet[m-1,0])

	for n in range(1,len(traj2)):
		matrice_frechet[0,n] = max(matrice_distance[0,n],matrice_frechet[0,n-1])

	for k in range(1,len(traj1)):
		for l in range(1,len(traj2)):
			matrice_frechet[k,l] = max(matrice_distance[k,l],min(matrice_frechet[k-1,l],matrice_frechet[k,l-1],matrice_frechet[k-1,l-1]))

	return matrice_frechet[-1,-1]

def evaluate_v(maze,v_vector,nb_traj,max_step,operator,optimal_traj):

	traj_map = np.zeros((maze.maze_size,maze.maze_size),dtype=int)
	length_list = []
	
	dtw_list = []
	frechet_list = []

	for epoch in tqdm(range(nb_traj)):
	    maze.env.reset()
	    state = [0,0]
	    traj_map[tuple(state)]+=1
	    traj = []
	    traj.append(list(state))
	    length = 0
	    
	    while (maze.env.state!=maze.env.observation_space.high).any() and length < max_step:
	        action = maze.select_action_from_v(state,v_vector,"human",operator)[0]
	        new_s,reward,done,_ = maze.env.step(int(action))
	        state = new_s
	        traj.append(list(state))
	        traj_map[tuple(state)]+=1
	        length+=1

	    dtw_list.append(distanceDTW(traj,optimal_traj))
	    frechet_list.append(distanceFrechet(traj,optimal_traj))
	    length_list.append(length)

	# print(dtw_list)
	# print(length_list)
	return int(np.array(length_list).mean()),int(np.array(length_list).std()), int(np.array(dtw_list).mean()), np.array(frechet_list).mean(), traj_map


	
m = MyMaze('maze-sample-20x20-v0')
m_obstacle = MyMaze('maze-sample-20x20-v0')


path = "./Q-Table/qtable1_20x20.npy"
#q_table = m.q_learning()
#np.save(path,q_table)
q_table = np.load(open(path,'rb'))
m.set_optimal_policy(q_table)
m_obstacle.set_optimal_policy(q_table)

obstacle = np.array([[0,3],[1,0],[15,17]])

m.set_reward([])
m_obstacle.set_reward(obstacle)



v_table_optimal = m.v_from_q(q_table)


state = m.env.reset()
optimal_traj = []
optimal_traj.append(list(state))

while (m.env.state!=m.env.observation_space.high).any():
	action = m.select_action_from_v(state,v_table_optimal,"human","argmax")[0]
	new_s, reward, done,_ = m.env.step(int(action))
	state = new_s
	optimal_traj.append(list(state))




v_table_optimal_obstacle = m.v_from_q(q_table)

state = m_obstacle.env.reset()
optimal_traj_obstacle = []
optimal_traj_obstacle.append(list(state))

while (m_obstacle.env.state!=m_obstacle.env.observation_space.high).any():
	action = m_obstacle.select_action_from_v(state,v_table_optimal_obstacle,"human","argmax")[0]
	new_s, reward, done,_ = m_obstacle.env.step(int(action))
	state = new_s
	optimal_traj_obstacle.append(list(state))


v_boltz01 = m.boltz_rational(0.1)
v_boltz10 = m.boltz_rational(10)
v_prospect_bias2 = m_obstacle.prospect_bias(2)
v_prospect_bias10 = m_obstacle.prospect_bias(50)
v_extremal0 = m.extremal(0)
# v_extremal05 = m.extremal(0.5)
v_extremal09 = m.extremal(0.99)
v_random_boltz = m.random_boltz_rational(50,0)

operator = "softmax"

res = []

print("Measures optimal")
res.append(evaluate_v(m,v_table_optimal,100,200,operator,optimal_traj))
print("Measures boltzmann 0")
res.append(evaluate_v(m,v_boltz01,100,200,operator,optimal_traj))
print("Measures boltzmann 10")
res.append(evaluate_v(m,v_boltz10,100,200,operator,optimal_traj))
print("Measures prospect_bias 1")
res.append(evaluate_v(m_obstacle,v_prospect_bias2 ,100,200,operator,optimal_traj_obstacle))
print("Measures prospect_bias 10")
res.append(evaluate_v(m_obstacle,v_prospect_bias10,100,200,operator,optimal_traj_obstacle))
print("Measures extremal0")
res.append(evaluate_v(m,v_extremal0,100,200,operator,optimal_traj))
# print("Measures extremal05")
# res.append(evaluate_v(m,v_extremal05,100,200,operator,optimal_traj))
print("Measures extremal09")
res.append(evaluate_v(m,v_extremal09,100,200,operator,optimal_traj))
print("Measures random boltz")
res.append(evaluate_v(m,v_random_boltz,100,200,operator,optimal_traj))


bias = ["optimal","boltz 0","boltz 10","prospect 2","prospect 50","extremal 0","extremal 0.99","random boltz"] #"extremal 0.5",

mean_len = []
std_len = []
mean_dtw = []
mean_frechet = []

for k in range(len(res)):
	mean_len.append(res[k][0])
	std_len.append(res[k][1])
	mean_dtw.append(res[k][2])
	mean_frechet.append(res[k][3])

data = {"mean len":mean_len,"std len":std_len,"mean dtw":mean_dtw,"mean frechet dist":mean_frechet}
results = pd.DataFrame(data=data)
results.index = ["Optimal","Boltzmann (0.1)","Boltzmann (10)","Prospect bias (2)","Prospect bias (50)","Extremal (0)","Extremal (0.99)","Random Boltz (0-50)"] #"Extremal (0.5)",

print(results)


for k in range(len(res)):
	ax = plt.subplot(2,4,k+1)
	ax.imshow(np.transpose(res[k][-1]))
	plt.title(bias[k])
	ax.set_xlim(-1,m.maze_size)
	ax.set_ylim(m.maze_size,-1)
	ax.set_aspect('equal')
	#ax.colorbar()


#plt.savefig('traj_measures.png')
results.to_csv('bias_measures.csv')

plt.show()



	

