import numpy as np
from MyMaze import *
import pandas as pd
from tqdm import tqdm

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

#np.sqrt((traj1[0][0]-traj2[0][0])**2 + (traj1[0][1]-traj2[0][1])**2)

fig_traj = plt.figure()
ax_traj = fig_traj.gca()

fig_traj1 = plt.figure()
ax_traj1 = fig_traj1.gca()


maze = MyMaze('maze-sample-20x20-v0',reward="human")


path = "./Q-Table/qtable1_20x20.npy"
q_table = np.load(open(path,'rb'))
maze.set_optimal_policy(q_table)
maze.set_reward()



vertex = []
for i in range(maze.maze_size):
	for j in range(maze.maze_size):
		vertex.append([i,j])

connection = {}
for v in vertex:
	c = []
	for a in range(4):
		maze.env.env.reset(np.array(v))
		new_state,_,_,_ = maze.env.step(a)
		if (new_state!=v).any():
			c.append(new_state)

	connection[tuple(v)] = c



v1 = maze.value_iteration()#maze.v_from_q(q_table)
beta = 10
v2 = maze.boltz_rational(beta)

fig_V = plt.figure()
ax_V = fig_V.gca()

plot_v_value(fig_V,ax_V,maze,v2,"")

# plt.ioff()
# plt.show()

distance_list_frechet = []
distance_list_DTW = []
distance_list_lev = []

traj_map1 = np.zeros((maze.maze_size,maze.maze_size))
traj_map2 = np.zeros((maze.maze_size,maze.maze_size))

for epoch in range(1):


	traj1 = []
	

	state = maze.env.reset()
	traj_map1[tuple(state)]+=1

	while (state!=maze.env.observation_space.high).any():

		traj1.append(list(state))
		action = maze.select_action_from_v(state,v1,maze.reward_type,"argmax",0)[0]
		new_s,reward,done,_ = maze.env.step(int(action))
		state = new_s
		traj_map1[tuple(state)]+=1



	traj2 = []
	

	state = maze.env.reset()
	traj_map2[tuple(state)]+=1

	while (state!=maze.env.observation_space.high).any():

		traj2.append(list(state))
		action = maze.select_action_from_v(state,v2,maze.reward_type,"softmax",beta)[0]
		new_s,reward,done,_ = maze.env.step(int(action))
		state = new_s
		traj_map2[tuple(state)]+=1

	print("Trajectories computed")

	matrice_distance = distance_matrix_dijstra(maze.env,traj1,traj2,connection)

	distance_list_frechet.append(distanceFrechet(traj1,traj2,matrice_distance))
	print("Frechet computed")
	distance_list_DTW.append(distanceDTW(traj1,traj2,matrice_distance))
	print("DTW computed")

	min_len = min(len(traj1),len(traj2))
	traj = [traj1,traj2]
	min_len_traj = traj[[len(traj1),len(traj2)]==min_len]
	max_len_traj = traj[traj!=min_len_traj]

	distance_lev = 0 

	for k in range(min_len//5):
		distance_lev += levenshteinDistance(traj1[k*5:(k+1)*5],traj2[k*5:(k+1)*5])

	distance_lev += levenshteinDistance(min_len_traj[5*min_len//5:],max_len_traj[5*min_len//5:])

	distance_list_lev.append(distance_lev)

im = ax_traj.imshow(np.transpose(traj_map1.reshape(maze.maze_size,maze.maze_size)))
for state in range(0,maze.maze_size*maze.maze_size):
    i=state//maze.maze_size
    j=state%maze.maze_size
    text = ax_traj.text(i,j, str(traj_map1[i,j])[0:4],ha="center", va="center", color="black")
fig_traj.suptitle("From Q*")

im = ax_traj1.imshow(np.transpose(traj_map2.reshape(maze.maze_size,maze.maze_size)))
for state in range(0,maze.maze_size*maze.maze_size):
    i=state//maze.maze_size
    j=state%maze.maze_size
    text = ax_traj1.text(i,j, str(traj_map2[i,j])[0:4],ha="center", va="center", color="black")
fig_traj1.suptitle("Boltzmann")

print("Distance de Frechet moyenne :",np.mean(distance_list_frechet))
print("Dynamic Time Warping moyen :",np.mean(distance_list_DTW))
print("Distance de Levenshtein moyenne :",np.mean(distance_list_lev))

print("\nFrechet max :",np.max(distance_list_frechet))
print("DTW max :",np.max(distance_list_DTW))
print("Lev max :",np.max(distance_list_lev))

print("\nFrechet last :",distance_list_frechet[-1])
print("DTW last :",distance_list_DTW[-1])
print("Lev last :",distance_list_lev[-1])

print(np.sum(traj_map1),np.sum(traj_map2))

# min_len = min(len(traj1),len(traj2))
# traj = [traj1,traj2]
# min_len_traj = traj[[len(traj1),len(traj2)]==min_len]
# max_len_traj = traj[traj!=min_len_traj]

# distance = 0 

# for k in range(min_len//5):
# 	distance += levenshteinDistance(traj1[k*5:(k+1)*5],traj2[k*5:(k+1)*5])

# distance += levenshteinDistance(min_len_traj[5*min_len//5:],max_len_traj[5*min_len//5:])
# print(distance)
# distance_list.append(distance)


#print("Distance moyenne",np.mean(distance_list))

# plot_traj(fig_traj,ax_traj,maze,v1,100,1000,"From Q*","softmax")
# plot_traj(fig_traj1,ax_traj1,maze,v2,100,1000,"Boltz","softmax")


plt.ioff()
plt.show()
