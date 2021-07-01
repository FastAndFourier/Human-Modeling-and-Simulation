import gym
import gym_maze
from gym_maze.envs import MazeEnvSample3x3, MazeEnvSample5x5, MazeEnvSample10x10, MazeEnvSample100x100
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import time

from maze_function import *
from MyMaze import *



if __name__ == "__main__":


    fig_policy = plt.figure()
    ax_policy = fig_policy.gca()
    #fig_delta = plt.figure()
    #ax_delta = fig_delta.gca()
    fig_V = plt.figure()
    ax_V = fig_V.gca()
    plt.ion()
    plt.pause(0.2)

    m = MyMaze("maze-sample-10x10-v0")


    path = "qtable3_10x10.npy"
    q_table = np.load(open(path,'rb'))

    #q_table = m.simulate()
    #np.save(path,q_table)

    m.set_optimal_policy(q_table)
    
    # m.env.reset()
    # m.env.render()
    # while (m.env.state!=[9,9]).any():
    #     m.env.step(int(m.optimal_policy[tuple(m.env.state)]))
    #     m.env.render()
    #     time.sleep(0.2)


    v_from_q = m.v_from_q(q_table)
    
    #m.generate_traj_v(v_from_q)
    #v_vector = m.boltz_rational(1,1e-7)
    #v_vector_prospect = m.prospect_bias(1e8,1e-7)
    v_vector_discount1 = m.myopic_discount(0.5)
    #print("Vdiscount : \n",v_vector_discount1)
    #m.generate_traj_v(v_vector)

    #v_vector_boltz1000 = m.boltz_rational(1)
    # v_vector_boltz1 = m.boltz_rational(1)
    # v_vector_boltz0 = m.boltz_rational(0)
    #v_vector_dbs, q_table_dbs = m.value_iteration_dynamic_boltzmann(1000)
    #v_from_dbs_q = m.v_from_q(q_table_dbs)
    #m.generate_traj_v(v_vector_boltz1000)
    # m.generate_traj_v(v_vector_boltz1)
    # m.generate_traj_v(v_vector_boltz0)


    _, walls_list = edges_and_walls_list_extractor(m.env)
    maze_size = m.maze_size

    ax_policy.set_xlim(-1.5,maze_size+0.5)
    ax_policy.set_ylim(maze_size+0.5,-1.5)
    ax_policy.set_aspect('equal')
    
    ax_V.set_xlim(-1,maze_size)
    ax_V.set_ylim(maze_size,-1)
    ax_V.set_aspect('equal')
    
    value_table = v_vector_discount1
    
    position = np.zeros(NUM_BUCKETS,dtype=list)
    direction = np.zeros(NUM_BUCKETS,dtype=list)
    
    
    for i in range(maze_size):
        for j in range(maze_size):
            if ([i,j]==[maze_size-1,maze_size-1]):
                break
            action = m.select_action_from_v([i,j],value_table)
            #print([i,j],action)

            if action==0:
                ax_policy.quiver(i,j,0,.75,color='c')
                #arrow.append([0,-1])
            if action==1:
                ax_policy.quiver(i,j,0,-.75,color='c')
                #arrow.append([0,1])
            if action==2:
                ax_policy.quiver(i,j,.75,0,color='c')
                #arrow.append([1,0])
            if action==3:
                ax_policy.quiver(i,j,-.75,0,color='c')
                #arrow.append([-1,0])

   
    
    #Draw value table
    im = ax_V.imshow(np.transpose(value_table.reshape(maze_size,maze_size)))
    if maze_size<=20:
        for state in range(0,MAX_SIZE[0]*MAX_SIZE[1]):
            i=state//maze_size
            j=state%maze_size
            text = ax_V.text(i,j, str(value_table[i,j])[0:4],ha="center", va="center", color="black")
    
    
    # # draw start and end position
    plot_start_marker = ax_policy.scatter(0,0, marker="o", s=100,c="b") # s = #size_of_the_marker#
    plot_end_marker = ax_policy.scatter(maze_size-1,maze_size-1, marker="o", s=100,c="r")
    
    # # draw maze boxes
    # for i in range(0,maze_size):
    #     for j in range(0,maze_size):
    #         # draw east line
    #         ax_policy.add_line(mlines.Line2D([i+0.5,i+0.5],[j-0.5,j+0.5],linewidth=0.2,color='gray'))
    #         ax_V.add_line(mlines.Line2D([i+0.5,i+0.5],[j-0.5,j+0.5],linewidth=0.2,color='gray'))
    #         # draw south line
    #         ax_policy.add_line(mlines.Line2D([i-0.5,i+0.5],[j+0.5,j+0.5],linewidth=0.2,color='gray'))
    #         ax_V.add_line(mlines.Line2D([i-0.5,i+0.5],[j+0.5,j+0.5],linewidth=0.2,color='gray'))
    
    #draw maze walls
    for i in walls_list:
        ax_policy.add_line(mlines.Line2D([i[1][0]-0.5,i[1][1]-0.5],[i[0][0]-0.5,i[0][1]-0.5],color='k'))
        ax_V.add_line(mlines.Line2D([i[1][0]-0.5,i[1][1]-0.5],[i[0][0]-0.5,i[0][1]-0.5],color='k'))
    
    # add east and top walls
    for i in range(0,maze_size):
        ax_policy.add_line(mlines.Line2D([-0.5,-0.5],[i-0.5,i+0.5],color='k'))
        ax_policy.add_line(mlines.Line2D([i-0.5,i+0.5],[-0.5,-0.5],color='k'))
        ax_policy.add_line(mlines.Line2D([maze_size-0.5,maze_size-0.5],[i-0.5,i+0.5],color='k'))
        ax_policy.add_line(mlines.Line2D([i-0.5,i+0.5],[maze_size-0.5,maze_size-0.5],color='k'))
        ax_V.add_line(mlines.Line2D([-0.5,-0.5],[i-0.5,i+0.5],color='k'))
        ax_V.add_line(mlines.Line2D([i-0.5,i+0.5],[-0.5,-0.5],color='k'))
        ax_V.add_line(mlines.Line2D([maze_size-0.5,maze_size-0.5],[i-0.5,i+0.5],color='k'))
        ax_V.add_line(mlines.Line2D([i-0.5,i+0.5],[maze_size-0.5,maze_size-0.5],color='k'))
    
    #ax_delta.plot([i for i in range(0,delta_list.size)],delta_list)
    
    plt.ioff()
    plt.show()

    

    

    