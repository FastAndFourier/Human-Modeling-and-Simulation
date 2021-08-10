from MyMaze import *
from matplotlib import colors
from colorsys import hsv_to_rgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import numpy as np
#from math import ceil

class Metrics():

	def __init__(self,m,qtab,nb_traj,max_step,beta):

		self.maze = m
		self.graph = self.get_graph()
		self.optimal_path = self.get_optimal_path(self.maze.v_from_q(qtab))
		self.nb_traj = nb_traj
		self.max_step = max_step
		self.beta_actor = beta



	########### CORE EVALUATION FUNCTION ################################

	def evaluate(self,v_vector,operator,beta):

		traj_map = np.zeros((self.maze.maze_size,self.maze.maze_size),dtype=int)
		length_list = []
		
		dtw_list = []
		frechet_list = []

		mean_step_per_tile = []



		#for epoch in tqdm(range(self.nb_traj)):
		for epoch in range(self.nb_traj):

			epoch_traj_map = np.zeros((self.maze.maze_size,self.maze.maze_size),dtype=int)
			self.maze.env.reset()
			state = [0,0]
			traj_map[tuple(state)]+=1
			epoch_traj_map[tuple(state)]+=1
			traj = []
			traj.append(list(state))
			length = 0
		    
			while (self.maze.env.state!=self.maze.env.observation_space.high).any() and length < self.max_step:

				action = self.maze.select_action_from_v(state,v_vector,self.maze.reward_type,operator,beta)[0]
				new_s,reward,done,_ = self.maze.env.step(int(action))
				state = new_s
				traj.append(list(state))
				traj_map[tuple(state)]+=1
				epoch_traj_map[tuple(state)]+=1
				length+=1

			matrice_distance = self.distance_matrix_dijstra(traj,self.optimal_path)


			dtw_list.append(distanceDTW(traj,self.optimal_path,matrice_distance))
			frechet_list.append(distanceFrechet(traj,self.optimal_path,matrice_distance))
			length_list.append(length)
			mean_step_per_tile.append((epoch_traj_map[epoch_traj_map!=0]).mean())


		print("Step per tile",mean_step_per_tile)
		print("Mean step per tile",np.array(mean_step_per_tile).mean())
		#return int(np.array(length_list).mean()),int(np.array(length_list).std()), int(np.array(dtw_list).mean()), np.array(frechet_list).mean(), np.transpose(traj_map)
		return np.array(length_list), np.array(dtw_list), np.array(frechet_list), np.transpose(traj_map)


	# def compare_traj(self,traj):

	#     distance_traj = np.zeros((self.maze.maze_size,self.maze.maze_size))

	#     for k in range(len(traj)):

	#         state_opti = tuple(traj_opti[k])
	#         state = tuple(traj[k])

	#         distance_traj[tuple(state_opti)] = abs(state_opti[0]-state[0]) + abs(state_opti[1]-state[1])

	#     return np.transpose(distance_traj)

	############# HELPER FUNCTIONS ######################################

	def get_graph(self):

	    vertex = []
	    for i in range(self.maze.maze_size):
	        for j in range(self.maze.maze_size):
	            vertex.append([i,j])

	    connection = {}
	    for v in vertex:
	        c = []
	        for a in range(4):
	            self.maze.env.env.reset(np.array(v))
	            new_state,_,_,_ = self.maze.env.step(a)
	            if (new_state!=v).any():
	                c.append(new_state)

	        connection[tuple(v)] = c

	    return connection

	def get_graph_lin(self):

	    vertex = []
	    for i in range(self.maze.maze_size):
	        for j in range(self.maze.maze_size):
	            vertex.append(i*self.maze.maze_size+j)

	    connection = {}
	    for v in vertex:
	        c = []
	        for a in range(4):
	            self.maze.env.env.reset(np.array([v//self.maze.maze_size,v%self.maze.maze_size]))
	            new_state,_,_,_ = self.maze.env.step(a)
	            lin_state = new_state[0]*self.maze.maze_size+new_state[1]
	            if lin_state!=v:
	                c.append(lin_state)

	        connection[v] = c

	    return connection


	def get_optimal_path(self,q_table):

		v_vector = self.maze.v_from_q(q_table)

		state = self.maze.env.reset()
		optimal_traj = []
		optimal_traj.append(list(state))

		while (self.maze.env.state!=self.maze.env.observation_space.high).any():
			action = self.maze.select_action_from_v(state,v_vector,self.maze.reward_type,"argmax",0)[0]
			new_s, reward, done,_ = self.maze.env.step(int(action))
			state = new_s
			optimal_traj.append(list(state))

		return optimal_traj



	################# DIJKSTRA ######################################################

	def distance_dijkstra(self,start,end):

		if tuple(start)==tuple(end):
			return 0

		size_maze = self.maze.maze_size

		source = start[0]*size_maze + start[1]
		stop = end[0]*size_maze + end[1]

		dist = np.inf*np.ones(size_maze**2)
		prev = np.zeros(size_maze**2)
		index_visited = np.arange(size_maze**2)
		dist[source] = 0



		while index_visited.size!=0:

			u = index_visited[dist[index_visited]==np.min(dist[index_visited])][0]

			index_visited = np.delete(index_visited,np.where(index_visited==u))

			for v in self.graph[tuple([u//size_maze,u%size_maze])]:
				lin_v = v[0]*size_maze + v[1]
				if lin_v in index_visited and (dist[u] + 1 < dist[lin_v]) :
			  		dist[lin_v] = dist[u] + 1
			  		prev[lin_v] = u

				if lin_v==stop:
					return int(dist[lin_v])
	      

			
		return -1



	def distance_matrix_dijstra(self,traj1,traj2):

		matrice_distance = np.zeros((len(traj1),len(traj2)))

		start_time = time.time()

		for i in range(len(traj1)):
			for j in range(len(traj2)):

				matrice_distance[i,j] = self.distance_dijkstra(traj1[i],traj2[j])

		#print("Distance matrix computed (",time.time()-start_time,"secondes )")

		return matrice_distance



	############### TRAJ COMPARAISON ###############################################

	def compare_traj_map(self,maze1,optimal_traj,v_bias,beta,nb_demo,title):
	

		# Trajectory map optimal demonstration
		optimal_traj_map = np.zeros((maze1.maze_size,maze1.maze_size))
		for i in optimal_traj:
			optimal_traj_map[tuple(i)] += 1

		optimal_traj_map = np.transpose(optimal_traj_map)

		# plt.figure()
		# plt.imshow(optimal_traj_map)


		# Optimal trajectory 2D state to linear state
		lin_optimal_traj = []
		for t in optimal_traj:
			lin_optimal_traj.append(t[0]*maze1.maze_size+t[1])


		# Biased trajectories map and linear states
		total_lin_traj = []
		traj_map = np.zeros((maze1.maze_size,maze1.maze_size))

		for k in range(nb_demo):
			traj = maze1.generate_traj_v(v_bias,"softmax",beta,self.max_step)[1]
			for t in traj:
				lin_t = t[0]*maze1.maze_size + t[1]			
				total_lin_traj.append(lin_t)
				traj_map[tuple(t)] += 1
		
		traj_map = np.transpose(traj_map)

		# plt.figure()
		# plt.imshow(traj_map)




		# Trajectories comparaison
		diff_map = np.zeros((maze1.maze_size,maze1.maze_size))
		for i in total_lin_traj:

			if i not in lin_optimal_traj:
				diff_map[i//maze1.maze_size,i%maze1.maze_size] =  -total_lin_traj.count(i)

		for i in lin_optimal_traj:
			diff_map[i//maze1.maze_size,i%maze1.maze_size] = 2
			if i not in total_lin_traj:
				diff_map[i//maze1.maze_size,i%maze1.maze_size] = 3

		diff_map = np.transpose(diff_map)

		fig = plt.figure()
		ax = fig.gca()
		ax.set_xlim(-1,maze1.maze_size)
		ax.set_ylim(maze1.maze_size,-1)
		ax.set_aspect('equal')

		color = []
		sorted_it = np.sort(diff_map[diff_map<0])

		boundaries = []
		label = []

		if sorted_it.size > 0:
			bound = [sorted_it[len(sorted_it)*k//3] for k in range(3)]
			boundaries = bound	
			color.append((0,0,1))
			color.append((0,0.4,1))
			color.append((0.5,0.7,1))	

			label = ["<"+str(int(abs(boundaries[0])))+" steps", \
				 "<"+str(int(abs(boundaries[1])))+" steps", \
				 "<"+str(int(abs(boundaries[2])))+" steps"]
		
		boundaries.extend([0,2,2.1,3])
		label.extend(["Tiles in common","Only optimal tiles"])

		
		color.append((1,1,1)) # White : normal tiles
		color.append((0,1,0)) # Green : common tiles
		color.append((1,0,0)) # Red : only optimal traj's tiles

		cmap = colors.LinearSegmentedColormap.from_list('my_cmap',color,N=len(color))
		norm = colors.BoundaryNorm(boundaries,cmap.N,clip=True)
		im = ax.imshow(diff_map,cmap=cmap,norm=norm)
		

		color_label = color
		color_label.remove((1,1,1))


		patches = [mpatches.Patch(color=color_label[i],label=label[i]) for i in range(len(label))]
		fig.legend(handles=patches)
		

		_, walls_list = maze1.edges_and_walls_list_extractor()

		for i in walls_list:
			ax.add_line(mlines.Line2D([i[1][0]-0.5,i[1][1]-0.5],[i[0][0]-0.5,i[0][1]-0.5],color='k'))

		for i in range(0,maze1.maze_size):
			ax.add_line(mlines.Line2D([-0.5,-0.5],[i-0.5,i+0.5],color='k'))
			ax.add_line(mlines.Line2D([i-0.5,i+0.5],[-0.5,-0.5],color='k'))
			ax.add_line(mlines.Line2D([maze1.maze_size-0.5,maze1.maze_size-0.5],[i-0.5,i+0.5],color='k'))
			ax.add_line(mlines.Line2D([i-0.5,i+0.5],[maze1.maze_size-0.5,maze1.maze_size-0.5],color='k'))

		fig.suptitle("Trust map for optimal and biased trajectories ("+str(nb_demo)+" demonstrations): "+title)


	def display_entropy(self,v_bias,beta,nb_demo,title):

		entropy_traj = np.zeros((self.maze.maze_size,self.maze.maze_size))

		for e in range(nb_demo):
			state = self.maze.env.env.reset(np.array([0,0]))
			done = False

			while not(done):

				action,h = self.maze.select_action_from_v(state,v_bias,self.maze.reward_type,"softmax",beta) 
				entropy_traj[tuple(state)] += h
				new_state, reward, done, _ = self.maze.env.step(int(action))
				state = new_state


		entropy_traj = entropy_traj/nb_demo
		viridis = cm.get_cmap('viridis',1024)
		newcolor = viridis(np.linspace(0,1,1024))
		newcolor[:1,:] = np.array([1,1,1,1])

		cmap = colors.ListedColormap(newcolor)


		entropy_traj = np.transpose(entropy_traj)

		fig = plt.figure()
		ax = fig.gca()

		im = ax.imshow(entropy_traj,cmap=cmap)
		fig.colorbar(im)

		_, walls_list = self.maze.edges_and_walls_list_extractor()

		for i in walls_list:
			ax.add_line(mlines.Line2D([i[1][0]-0.5,i[1][1]-0.5],[i[0][0]-0.5,i[0][1]-0.5],color='k'))

		for i in range(0,self.maze.maze_size):
			ax.add_line(mlines.Line2D([-0.5,-0.5],[i-0.5,i+0.5],color='k'))
			ax.add_line(mlines.Line2D([i-0.5,i+0.5],[-0.5,-0.5],color='k'))
			ax.add_line(mlines.Line2D([self.maze.maze_size-0.5,self.maze.maze_size-0.5],[i-0.5,i+0.5],color='k'))
			ax.add_line(mlines.Line2D([i-0.5,i+0.5],[self.maze.maze_size-0.5,self.maze.maze_size-0.5],color='k'))

		fig.show()
		fig.suptitle("Entropy on biased trajectories ("+str(nb_demo)+" demonstrations): "+title)



############## DISTANCES #########################################################

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


	return matrice_frechet[-1,-1]


