from MyMaze import *
from Tool_MyMaze import *
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


fig_V = plt.figure()
ax_V = fig_V.gca()

nb_traj = 1

maze = MyMaze('maze-sample-20x20-v0',reward="human")

path = "./Q-Table/qtable1_20x20.npy"
q_table = np.load(open(path,'rb'))
maze.set_optimal_policy(q_table)

obstacle = []
maze.set_reward(obstacle)

measure = Metrics(m=maze,qtab=q_table,nb_traj=nb_traj,max_step=500,beta=2)
optimal_traj = measure.get_optimal_path(q_table)

#################################################################


maze_obs = MyMaze('maze-sample-20x20-v0',reward="human")

path_obs = "./Q-Table/qtable1_20x20_obs.npy"
q_table_obs = np.load(open(path_obs,'rb'))
maze_obs.set_optimal_policy(q_table_obs)

obstacle = np.array([[3,6],[1,4]])
maze_obs.set_reward(obstacle)

measure_obs = Metrics(m=maze_obs,qtab=q_table_obs,nb_traj=nb_traj,max_step=500,beta=2)
optimal_traj_obs = measure_obs.get_optimal_path(q_table_obs)


###############################################################

start_measure = time.time()


res = []


v_boltz0 = maze.boltz_rational(1)
measure_boltz0 = measure.evaluate(v_boltz0,"softmax",1)
res.append(measure_boltz0)
print("\n")

v_boltz1 = maze.boltz_rational(10)
measure_boltz1 = measure.evaluate(v_boltz1,"softmax",10)
res.append(measure_boltz1)
print("\n")

v_prospect0 = maze_obs.prospect_bias(2,measure_obs.beta_actor)
measure_prospect0 = measure_obs.evaluate(v_prospect0,"softmax",measure.beta_actor)
res.append(measure_prospect0)
print("\n")

v_prospect1 = maze_obs.prospect_bias(50,measure_obs.beta_actor)
measure_prospect1 = measure_obs.evaluate(v_prospect1,"softmax",measure_obs.beta_actor)
res.append(measure_prospect1)
print("\n")

v_extremal0 = maze.extremal(0,measure.beta_actor)
measure_extremal0 = measure.evaluate(v_extremal0,"softmax",measure.beta_actor)
res.append(measure_extremal0)
print("\n")

v_extremal1 = maze.extremal(0.99,measure.beta_actor)
measure_extremal1 = measure.evaluate(v_extremal1,"softmax",measure.beta_actor)
res.append(measure_extremal1)
print("\n")

v_myopic_disc0 = maze.myopic_discount(0.1,measure.beta_actor)
measure_myopic_disc0 = measure.evaluate(v_myopic_disc0,"softmax",measure.beta_actor)
res.append(measure_myopic_disc0)
print("\n")

v_myopic_disc1 = maze.myopic_discount(0.99,measure.beta_actor)
measure_myopic_disc1 = measure.evaluate(v_myopic_disc1,"softmax",measure.beta_actor)
res.append(measure_myopic_disc1)
print("\n")

v_myopic_vi0 = maze.myopic_value_iteration(5,measure.beta_actor)
measure_myopic_vi0 = measure.evaluate(v_myopic_vi0,"softmax",measure.beta_actor)
res.append(measure_myopic_vi0)
print("\n")

v_myopic_vi1 = maze.myopic_value_iteration(200,measure.beta_actor)
measure_myopic_vi1 = measure.evaluate(v_myopic_vi1,"softmax",measure.beta_actor)
res.append(measure_myopic_vi1)
print("\n")

v_hyperdisc0 = maze.hyperbolic_discount(0,measure.beta_actor)
measure_hyperdisc0 = measure.evaluate(v_hyperdisc0,"softmax",measure.beta_actor)
res.append(measure_hyperdisc0)
print("\n")

v_hyperdisc1 = maze.hyperbolic_discount(5,measure.beta_actor)
measure_hyperdisc1 = measure.evaluate(v_hyperdisc1,"softmax",measure.beta_actor)
res.append(measure_hyperdisc1)
print("\n")


print("Measures finished ! (",(time.time()-start_measure)//3600,"h",((time.time()-start_measure)%3600)//60,"min",(time.time()-start_measure)%60,"sec )")

index = ["boltz 1","boltz 10","prospect 2","prospect 50","extremal 0","extremal 0.99","myopic disc 0.1","myopic disc 0.99",\
         "myopic VI 5","myopic VI 200","hyper disc 0","hyper disc 5"]

length = np.copy(res[0][0])
dtw = np.copy(res[0][1])
frechet = np.copy(res[0][2])
bias_name = np.array([index[0]]*nb_traj)


for k in range(1,len(res)):
    length = np.concatenate((length,res[k][0]))
    dtw = np.concatenate((dtw,res[k][1]))
    frechet = np.concatenate((frechet,res[k][2]))
    bias_name = np.concatenate((bias_name,np.array([index[k]]*nb_traj)))

print(length)
print(dtw)
print(frechet)
print(bias_name)

#data = {"mean len":mean_len,"std len":std_len,"mean dtw":mean_dtw,"mean frechet dist":mean_frechet}
#results = pd.DataFrame(data=data)
#results.index = ["boltz 1","boltz 10","prospect 2","prospect 50","extremal 0","extremal 0.99","myopic disc 0.1","myopic disc 0.99",\
              #"myopic VI 5","myopic VI 200","hyper disc 0","hyper disc 5"] 


data = {"Bias":bias_name,"length":length,"DTW":dtw,"Frechet":frechet}
results = pd.DataFrame(data=data)
results.pdDataFrame(data=data)

print(results)
#results.to_csv('bias_measures_v2.csv')

plt.figure()

for k in range(6):
    ax = plt.subplot(2,3,k+1)
    ax.imshow(res[k][-1])
    plt.title(results.index[k])
    ax.set_xlim(-1,maze.maze_size)
    ax.set_ylim(maze.maze_size,-1)
    ax.set_aspect('equal')


plt.figure()

for k in range(6,len(res)):
    ax = plt.subplot(2,3,k-5)
    ax.imshow(res[k][-1])
    plt.title(results.index[k])
    ax.set_xlim(-1,maze.maze_size)
    ax.set_ylim(maze.maze_size,-1)
    ax.set_aspect('equal')

plt.show()