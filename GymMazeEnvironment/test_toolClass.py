from MyMaze import *
from Tool_MyMaze import *

maze = MyMaze('maze-sample-20x20-v0',reward="human")

path = "./Q-Table/qtable1_20x20.npy"
q_table = np.load(open(path,'rb'))
maze.set_optimal_policy(q_table)

obstacle = []
maze.set_reward(obstacle)

measure = Metrics(m=maze,qtab=q_table,nb_traj=1,max_step=1000,beta=2)
optimal_traj = measure.get_optimal_path(q_table)

v_boltz01 = maze.boltz_rational(0.5)
measure.evaluate(v_boltz01,"softmax")