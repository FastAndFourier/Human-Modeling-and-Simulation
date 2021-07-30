from MyMaze import *

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
		return int(np.array(length_list).mean()),int(np.array(length_list).std()), int(np.array(dtw_list).mean()), np.array(frechet_list).mean(), np.transpose(traj_map)


	def compare_traj(self,traj):

	    distance_traj = np.zeros((self.maze.maze_size,self.maze.maze_size))

	    for k in range(len(traj)):

	        state_opti = tuple(traj_opti[k])
	        state = tuple(traj[k])

	        distance_traj[tuple(state_opti)] = abs(state_opti[0]-state[0]) + abs(state_opti[1]-state[1])

	    return np.transpose(distance_traj)

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

		print("Distance matrix computed (",time.time()-start_time,"secondes )")

		return matrice_distance



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


