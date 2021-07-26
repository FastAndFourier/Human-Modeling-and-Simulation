import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import time
import pandas as pd

from maze_function import *
from MyMaze import *


def distance_dijkstra(env,start,end,connection):

	if tuple(start)==tuple(end):
		return 0

	size_maze = env.observation_space.high[0] + 1

	source = start[0]*size_maze + start[1]
	

	dist = np.inf*np.ones(size_maze**2)
	prev = np.zeros(size_maze**2)
	visited = [False]*size_maze**2#np.copy(vertex)
	dist[source] = 0



	while (visited!=[True]*size_maze**2):

		min = size_maze**2
		for v in range(size_maze**2):
			if dist[v] < min and visited[v] == False:
				min = dist[v]
				u = v

		visited[u] = True

		for v in connection[tuple([u//size_maze,u%size_maze])]:
			lin_v = v[0]*size_maze+v[1]
			if visited[lin_v]==False and (dist[u] + 1 < dist[lin_v]) :
				dist[lin_v] = dist[u] + 1
				prev[lin_v] = u

			if (v==end).all():
				return dist[lin_v]

		
	return -1

def distance_matrix_dijstra(env,traj1,traj2,connection):

	matrice_distance = np.zeros((len(traj1),len(traj2)))

	start_time = time.time()

	for i in range(len(traj1)):
		for j in range(len(traj2)):

			matrice_distance[i,j] = distance_dijkstra(env,traj1[i],traj2[j],connection)

	print("Distance matrix computed (",time.time()-start_time,"secondes )")

	return matrice_distance


def levenshteinDistance(traj1,traj2):

	if len(traj1)==0:
		return len(traj2)

	elif len(traj2)==0:
		return len(traj1)

	elif traj1[0]==traj2[0]:
		return levenshteinDistance(traj1[1:],traj2[1:])

	else:
		return 1 + min(levenshteinDistance(traj1,traj2[1:]),levenshteinDistance(traj1[1:],traj2),levenshteinDistance(traj1[1:],traj2[1:]))


def distanceDTW(traj1,traj2,matrice_distance):


	matrice_DTW = np.zeros((len(traj1),len(traj2)))

	matrice_DTW[0,0] = matrice_distance[0,0]

	for m in range(1,len(traj1)):
		matrice_DTW[m,0] =matrice_distance[m,0] + matrice_DTW[m-1,0]

	for n in range(1,len(traj2)):
		matrice_DTW[0,n] = matrice_distance[0,n] + matrice_DTW[0,n-1]


	for k in range(1,len(traj1)):
		for l in range(1,len(traj2)):
			matrice_DTW[k,l] = matrice_distance[k,l] + min(matrice_DTW[k-1,l],matrice_DTW[k,l-1],matrice_DTW[k-1,l-1])


	# pd.DataFrame(matrice_distance).to_csv('distance_matrix.csv')
	# pd.DataFrame(matrice_frechet).to_csv('frechet_matrix.csv')
	# pd.DataFrame(matrice_DTW).to_csv('DTW_matrix.csv')

	return matrice_DTW[-1,-1]


def distanceFrechet(traj1,traj2,matrice_distance):

	matrice_frechet = np.zeros((len(traj1),len(traj2)))

	
	matrice_frechet[0,0] = matrice_distance[0,0]

	for m in range(1,len(traj1)):
		matrice_frechet[m,0] = max(matrice_distance[m,0],matrice_frechet[m-1,0])

	for n in range(1,len(traj2)):
		matrice_frechet[0,n] = max(matrice_distance[0,n],matrice_frechet[0,n-1])

	for k in range(1,len(traj1)):
		for l in range(1,len(traj2)):
			matrice_frechet[k,l] = max(matrice_distance[k,l],min(matrice_frechet[k-1,l],matrice_frechet[k,l-1],matrice_frechet[k-1,l-1]))

	# pd.DataFrame(matrice_distance).to_csv('distance_matrix.csv')
	# pd.DataFrame(matrice_frechet).to_csv('frechet_matrix.csv')
	# pd.DataFrame(matrice_DTW).to_csv('DTW_matrix.csv')

	return matrice_frechet[-1,-1]

def evaluate_v(maze,v_vector,beta,nb_traj,max_step,operator,optimal_traj,connection):

	traj_map = np.zeros((maze.maze_size,maze.maze_size),dtype=int)
	length_list = []
	
	dtw_list = []
	frechet_list = []

	mean_step_per_tile = []



	for epoch in tqdm(range(nb_traj)):

		epoch_traj_map = np.zeros((maze.maze_size,maze.maze_size),dtype=int)
		maze.env.reset()
		state = [0,0]
		traj_map[tuple(state)]+=1
		epoch_traj_map[tuple(state)]+=1
		traj = []
		traj.append(list(state))
		length = 0
	    
		while (maze.env.state!=maze.env.observation_space.high).any() and length < max_step:
			action = maze.select_action_from_v(state,v_vector,maze.reward_type,operator,beta)[0]
			new_s,reward,done,_ = maze.env.step(int(action))
			state = new_s
			traj.append(list(state))
			traj_map[tuple(state)]+=1
			epoch_traj_map[tuple(state)]+=1
			length+=1

		matrice_distance = distance_matrix_dijstra(maze.env,traj,optimal_traj,connection)

		dtw_list.append(distanceDTW(traj,optimal_traj,matrice_distance))
		frechet_list.append(distanceFrechet(traj,optimal_traj,matrice_distance))
		length_list.append(length)
		mean_step_per_tile.append((epoch_traj_map[epoch_traj_map!=0]).mean())


	print(mean_step_per_tile)
	print(np.array(mean_step_per_tile).mean())
	return int(np.array(length_list).mean()),int(np.array(length_list).std()), int(np.array(dtw_list).mean()), np.array(frechet_list).mean(), np.transpose(traj_map)


	
m = MyMaze('maze-sample-20x20-v0',reward="human")
m_obstacle = MyMaze('maze-sample-20x20-v0',reward="human")


path = "./Q-Table/qtable1_20x20.npy"
#q_table = m.q_learning()
#np.save(path,q_table)
q_table = np.load(open(path,'rb'))
m.set_optimal_policy(q_table)
m_obstacle.set_optimal_policy(q_table)

obstacle = np.array([[0,3],[1,0],[15,17]])

m.set_reward([])
m_obstacle.set_reward(obstacle)

vertex = []
for i in range(m.maze_size):
	for j in range(m.maze_size):
		vertex.append([i,j])

connection = {}
for v in vertex:
	c = []
	for a in range(4):
		m.env.env.reset(np.array(v))
		new_state,_,_,_ = m.env.step(a)
		if (new_state!=v).any():
			c.append(new_state)

	connection[tuple(v)] = c

v_table_optimal = m.v_from_q(q_table)

print("Connection computed")


state = m.env.reset()
optimal_traj = []
optimal_traj.append(list(state))

while (m.env.state!=m.env.observation_space.high).any():
	action = m.select_action_from_v(state,v_table_optimal,m.reward_type,"argmax",0)[0]
	new_s, reward, done,_ = m.env.step(int(action))
	state = new_s
	optimal_traj.append(list(state))




v_table_optimal_obstacle = m.v_from_q(q_table)

state = m_obstacle.env.reset()
optimal_traj_obstacle = []
optimal_traj_obstacle.append(list(state))

while (m_obstacle.env.state!=m_obstacle.env.observation_space.high).any():
	action = m_obstacle.select_action_from_v(state,v_table_optimal_obstacle,m_obstacle.reward_type,"argmax",0)[0]
	new_s, reward, done,_ = m_obstacle.env.step(int(action))
	state = new_s
	optimal_traj_obstacle.append(list(state))


print("Optimal trajectories computed")

beta_boltz1 = 1
beta_boltz2 = 10

beta_all = 2

v_boltz01 = m.boltz_rational(beta_boltz1)
v_boltz10 = m.boltz_rational(beta_boltz2)

v_prospect_bias2 = m_obstacle.prospect_bias(2,beta_all)
v_prospect_bias10 = m_obstacle.prospect_bias(50,beta_all)

v_extremal0 = m.extremal(0,beta_all)
v_extremal09 = m.extremal(0.99,beta_all)

v_random_boltz = m.random_boltz_rational(5,0.5)

v_myopic01 = m.myopic_discount(0.1,beta_all)
v_myopic099 = m.myopic_discount(0.99,beta_all)

v_myopicVI5 = m.myopic_value_iteration(5,beta_all)
v_myopicVI50 = m.myopic_value_iteration(50,beta_all)

v_hyperdisc0 = m.hyperbolic_discount(0,beta_all)
v_hyperdisc10 = m.hyperbolic_discount(2,beta_all)

operator = "softmax"

res = []


nb_traj = 5
max_step = 1000

print("Measures optimal")
res.append(evaluate_v(m,v_table_optimal,beta_all,nb_traj,max_step,operator,optimal_traj,connection))
print(res[-1])

print("Measures boltzmann 0")
res.append(evaluate_v(m,v_boltz01,beta_boltz1,nb_traj,max_step,operator,optimal_traj,connection))
print(res[-1])
print("Measures boltzmann 10")
res.append(evaluate_v(m,v_boltz10,beta_boltz2,nb_traj,max_step,operator,optimal_traj,connection))
print(res[-1])

print("Measures prospect_bias 1")
res.append(evaluate_v(m_obstacle,v_prospect_bias2,beta_all,nb_traj,max_step,operator,optimal_traj_obstacle,connection))
print(res[-1])
print("Measures prospect_bias 10")
res.append(evaluate_v(m_obstacle,v_prospect_bias10,beta_all,nb_traj,max_step,operator,optimal_traj_obstacle,connection))
print(res[-1])

print("Measures extremal0")
res.append(evaluate_v(m,v_extremal0,beta_all,nb_traj,max_step,operator,optimal_traj,connection))
print(res[-1])
# print("Measures extremal05")
# res.append(evaluate_v(m,v_extremal05,nb_traj,max_step,operator,optimal_traj))
print("Measures extremal09")
res.append(evaluate_v(m,v_extremal09,beta_all,nb_traj,max_step,operator,optimal_traj,connection))
print(res[-1])

print("Measures random boltz")
res.append(evaluate_v(m,v_random_boltz,beta_all,nb_traj,max_step,operator,optimal_traj,connection))
print(res[-1])

print("Measures myopic discount 0.1")
res.append(evaluate_v(m,v_myopic01,beta_all,nb_traj,max_step,operator,optimal_traj,connection))
print(res[-1])
print("Measures myopic discount 0.99")
res.append(evaluate_v(m,v_myopic099,beta_all,nb_traj,max_step,operator,optimal_traj,connection))
print(res[-1])

print("Measures myopic VI 5")
res.append(evaluate_v(m,v_myopicVI5,beta_all,nb_traj,max_step,operator,optimal_traj,connection))
print(res[-1])
print("Measures myopic VI 50")
res.append(evaluate_v(m,v_myopicVI50,beta_all,nb_traj,max_step,operator,optimal_traj,connection))
print(res[-1])

print("Measures hyperbolic discount 0")
res.append(evaluate_v(m,v_hyperdisc0,beta_all,nb_traj,max_step,operator,optimal_traj,connection))
print(res[-1])
print("Measures hyperbolic discount 10")
res.append(evaluate_v(m,v_hyperdisc10,beta_all,nb_traj,max_step,operator,optimal_traj,connection))
print(res[-1])




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
results.index = ["Optimal","Boltzmann (0.1)","Boltzmann (10)","Prospect bias (2)","Prospect bias (50)","Extremal (0)",\
				 "Extremal (0.99)","Random Boltz (0-50)","Myopic Discount (0.1)","Myopic Discount (0.99)","Myopic VI (5)",\
				 "Myopic VI (50)","Hyperbolic Discount (0)","Hyperbolic Discount (10)"] #"Extremal (0.5)",

print(results)
results.to_csv('bias_measures2.csv')


bias_label = ["optimal","boltz 0","boltz 10","prospect 2","prospect 50","extremal 0","extremal 0.99","random boltz","myopic disc 0.1","myopic disc 0.99",\
			  "myopic VI 5","myopic disc 50","hyper disc 0","hyper disc 10"] #"extremal 0.5",

bias_index_display = [0,2,4,7,9,10,11,12]

_, walls_list = m.edges_and_walls_list_extractor()


res_display = list(np.array(res,dtype=list)[bias_index_display])
bias_label_display = list(np.array(bias_label)[bias_index_display])

for k in range(len(res_display)):
	ax = plt.subplot(2,4,k+1)
	ax.imshow(res_display[k][-1])
	plt.title(bias_label_display[k])
	ax.set_xlim(-1,m.maze_size)
	ax.set_ylim(m.maze_size,-1)
	ax.set_aspect('equal')

	for i in walls_list:
		ax.add_line(mlines.Line2D([i[1][0]-0.5,i[1][1]-0.5],[i[0][0]-0.5,i[0][1]-0.5],color='k'))

	for i in range(0,m.maze_size):
		ax.add_line(mlines.Line2D([-0.5,-0.5],[i-0.5,i+0.5],color='k'))
		ax.add_line(mlines.Line2D([i-0.5,i+0.5],[-0.5,-0.5],color='k'))
		ax.add_line(mlines.Line2D([m.maze_size-0.5,m.maze_size-0.5],[i-0.5,i+0.5],color='k'))
		ax.add_line(mlines.Line2D([i-0.5,i+0.5],[m.maze_size-0.5,m.maze_size-0.5],color='k'))

	#ax.colorbar()


# #plt.savefig('traj_measures.png')


plt.show()



	

