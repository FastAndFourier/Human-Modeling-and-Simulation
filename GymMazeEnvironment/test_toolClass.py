from MyMaze import *
from Tool_MyMaze import *


fig_V = plt.figure()
ax_V = fig_V.gca()

fig_traj = plt.figure()
ax_traj = fig_traj.gca()

maze = MyMaze('maze-sample-20x20-v0',reward="human")

path = "./Q-Table/qtable1_20x20.npy"
q_table = np.load(open(path,'rb'))
maze.set_optimal_policy(q_table)

obstacle = []
maze.set_reward(obstacle)

measure = Metrics(m=maze,qtab=q_table,nb_traj=1,max_step=1000,beta=2)
optimal_traj = measure.get_optimal_path(q_table)

v_boltz01 = maze.boltz_rational(1)

measure_boltz01 = measure.evaluate(v_boltz01,"softmax",1)
traj = measure_boltz01[-1]

maze.plot_v_value(fig_V,ax_V,v_boltz01,"V value boltzmann 0.1")
im = ax_traj.imshow(traj.reshape(maze.maze_size,maze.maze_size))
for state in range(0,maze.maze_size*maze.maze_size):
    i=state//maze.maze_size
    j=state%maze.maze_size
    text = ax_traj.text(i,j, str(traj[j,i])[0:4],ha="center", va="center", color="black")


plt.show()