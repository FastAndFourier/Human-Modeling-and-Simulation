import numpy as np
import gym
import gym_maze
import time

def distance_dijkstra(env,start,end,vertex):

	size_maze = env.observation_space.high[0] + 1
	
	source = [0,0]
	source = source[0]*size_maze + source[1]
	

	dist = np.inf*np.ones(len(vertex))
	prev = np.zeros(len(vertex))
	visited = [False]*len(vertex)#np.copy(vertex)
	dist[source] = 0


	connection = {}
	for v in vertex:
		neighbor = []
		for a in range(4):
			state = env.env.reset(np.array(v))
			new_state,_,_,_ = env.step(a)
			if (new_state!=state).any():
				neighbor.append(new_state)

		connection[tuple(v)] = neighbor


	while (visited!=[True]*len(vertex)):

		min = size_maze**2
		for v in range(len(vertex)):
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

env = gym.make('maze-sample-20x20-v0')
env.reset()
size_maze = env.observation_space.high[0] + 1

vertex = []
for i in range(size_maze):
	for j in range(size_maze):
		vertex.append([i,j])

for i in range(size_maze):
	for j in range(size_maze):
		print(distance_dijkstra(env,[0,0],[i,j],vertex))



	
	
