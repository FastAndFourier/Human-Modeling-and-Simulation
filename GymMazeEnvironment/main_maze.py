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

    m = MyMaze("maze-sample-5x5-v0")
    path = "./Q-Table/qtable1_5x5.npy"
    #q_table = m.q_learning()
    #np.save(path,q_table)
    q_table = np.load(open(path,'rb'))
    m.set_optimal_policy(q_table)

    vi_vector = m.boltz_value_iteration(0)
    v_from_q = m.v_from_q(q_table)

    m.generate_traj_v(vi_vector)

    while True:
        m.env.render()

    path = "./Q-Table/qtable2_5x5.npy"
    q_table = np.load(open(path,'rb'))

    #q_table = m.simulate()

    #m.boltz_rational_noisy(q_table,1e-4)

    #v_from_q = m.v_from_q(q_table)
    #print(v_from_q)
    
    #m.generate_traj_v(v_from_q)
    v_vector = m.boltz_rational(beta=0.05,theta=0.1)
    #m.generate_traj_v(v_vector)


    _, walls_list = m.edges_and_walls_list_extractor()
    maze_size = m.maze_size

    ax_policy.set_xlim(-1.5,maze_size+0.5)
    ax_policy.set_ylim(maze_size+0.5,-1.5)
    ax_policy.set_aspect('equal')
    
    ax_V.set_xlim(-1.5,maze_size+0.5)
    ax_V.set_ylim(maze_size+0.5,-1.5)
    ax_V.set_aspect('equal')
    
    value_table = v_vector
    
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
            text = ax_V.text(i,j, str(value_table[i,j])[0:4],ha="center", va="center", color="w")
    
    
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

    

    

    # #env = MazeEnvSample10x10()
    # env = gym.make("maze-sample-10x10-v0")


    # env.render()
    # #env1 = gym.make("maze-random-10x10-plus-v0")
    # #env = gym.make("maze2d_10x10")

    # MAX_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    # NUM_BUCKETS = MAX_SIZE
    # NUM_ACTIONS = env.action_space.n
    # STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))


    # LR = 0.05
    # EPSILON = 0.02
    # MAX_EPISODE = 50000
    # MAX_STEP = 200
    # DISCOUNT = 1 #0.99
    # MIN_STREAK = MAX_EPISODE
    # RENDER = True
    # SIMULATE = False

    # OP_VI = 1 # 0 = Argmax, 1 = Softmax

    # print(env.state)

    # if SIMULATE:
    #     q_table = simulate()
    #     path = "qtable2_10x10"
    #     np.save(path,q_table)
    # else:
    #     path = "qtable2_10x10.npy"
    #     q_table = np.load(open(path,'rb'))

    # v_from_q = np.zeros((NUM_BUCKETS[0],NUM_BUCKETS[1]))
    # v_vector = boltz_rational(env,0.01)

    # for i in range(NUM_BUCKETS[0]):
    #     for j in range(NUM_BUCKETS[1]):
    #         state = state_to_bucket([i,j])
    #         v_from_q[i,j] = np.max(q_table[state])
    # v_from_q[tuple(env.observation_space.high)] = 1
    # _, walls_list = edges_and_walls_list_extractor(env)


    # #print(select_action_from_v(env,[0,0],v_from_q))
    # #reward_tab = get_reward(env)

    # # print(env.state)
    # # print(env.step("E")[0])
    # # print(env.step("S")[0])

    # boltz_rational_noisy(env,q_table,1e-4)
    # generate_traj_v(env,v_vector)
    # generate_traj_v(env,v_from_q)


    # maze_size = MAX_SIZE[0]
    
    





    # v_vector = boltz_rational(env,1)
    # print("Boltzmann value function :\n",v_vector)
    #
    # v_from_q = np.zeros((NUM_BUCKETS[0],NUM_BUCKETS[1]))
    # for i in range(NUM_BUCKETS[0]):
    #     for j in range(NUM_BUCKETS[1]):
    #         state = state_to_bucket([i,j])
    #         v_from_q[i,j] = np.max(q_table[state])
    #
    # print("\nV(s) = Q(s,pi(s)) :\n",v_from_q)

    # generate_traj_v(env,v_vector)
    # generate_traj_v(env,v_from_q)


    # traj = []
    # beta = [1e-1,1e-3]
    # beta = [1e-4,1e-5]
    # for b in beta:
    #     print(b)
    # demo = boltz_rational_noisy(env,q_table,1e-4)
    # print(env.state)
    # s = env.env.reset([0,0])
    # env.render()
    # print(env.state)
    #demo = boltz_rational_noisy(env,q_table,1e-5)

    #while True:

    # _=env.reset()
    # env.env.reset(np.array([0,0]))
    # demo = boltz_rational_noisy(env,q_table,1e-4)
    # traj.append(demo)
    #
    # print("Trajectory length",[len(t) for t in traj])
    # len_traj = [len(t) for t in traj]
    #
    # plt.hist(len_traj,density=True)
    # plt.show()
    # print("Min trajectory length",np.min([len(t) for t in traj]))
    # print("Max trajectory length",np.max([len(t) for t in traj]))
    # print("Mean",np.mean([len(t) for t in traj]))
    # print("Standard deviation",np.std([len(t) for t in traj]))

    # print("IN")
    # state = state_to_bucket(env.state)
    # print(state)
    # EPSILON = 0
    # a = []
    # env.render()
    #
    # for k in range(MAX_STEP):
    #     action = select_action(state,q_table,0)
    #     a.append(action)
    #     new_s, reward, done, _ = env.step(action)
    #     new_s = state_to_bucket(new_s)
    #     state = new_s
    #
    #     if RENDER:
    #         env.render()
    #         time.sleep(0.1)
    #     if done :
    #         break
    #
    # print(len(a),"itérations -> ",action2str(a))
