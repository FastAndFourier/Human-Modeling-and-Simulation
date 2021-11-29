import matplotlib.pyplot as plt
from Grid import *
from boltzmann_rational import *
from irrational_behavior import *
import numpy as np
from time import sleep
import pygame

print("Environment 1 initialization (boltzmann noisy-rational and irrational bias) ... \n")
oox = grid([5,6])
init_start = oox.start
oox.add_end([2,5])
oox.add_danger([2,4])
oox.add_danger([2,3])
oox.add_danger([3,4])
oox.add_danger([4,4])
oox.add_danger([4,3])
oox.add_danger([4,2])
oox.add_danger([4,1])
oox.render()


step_, err_ = oox.q_learning(50,50)
oox.display_qtable()

oox.reset(init_start)

demo_action, _, _ = boltz_rational_noisy(oox,0.1,1,oox.start)


oox.reset(init_start)
for action in demo_action[0]:
    new_state, reward, done = oox.step(action)
    oox.state = new_state
    oox.render()
    sleep(.5)

run = True
while run:
    pygame.time.delay(100)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
pygame.quit()

    
    

# vi = value_iterate(oox)

# print("Trajectories Boltzmann VI :")
# beta = [0,0.5,1,10]
# for b in beta:
#     vB = boltz_rational(oox,b)
#     print(vB)
#     print("Beta = ",b," :",end=" ")
#     generate_traj_v(oox,vB,[3,3])
#     print("\n")

# print("\nTrajectories Boltzmann noisy-rational :")
# beta = [10000,2,0.1,0.02]
# for b in beta:
#     print("Beta = ",1/b," :",end=" ")
#     demo, start_ = boltz_rational_noisy(oox,b,1,[3,3])
#     oox.reset([3,3])
#     print("Start =",start_,"-> ",len(demo[0]),"iterations",end="")
#     print(action2str(demo[0]),"\n")


# print("\n\nEnvironment 2 initialization (prospect bias) (...\n")

# grid1 = grid([5,5])
# grid1.add_end([0,4])



# grid1.add_danger([0,1])
# grid1.add_danger([0,2])
# grid1.add_danger([1,1])
# grid1.add_danger([1,2])
# grid1.add_danger([2,2])
# grid1.add_danger([3,2])
# grid1.add_danger([3,3])

# grid1.print_env()

# step_, err_ = grid1.q_learning(50,50000)



# print("Prospect bias :")
# vP = prospect_bias(grid1,4)
# vP1 = prospect_bias(grid1,5)
# print("\n")
# generate_traj_v(grid1,vP,[0,0])
# generate_traj_v(grid1,vP1,[0,0])

# print("\n\nEnvironment 3 initialization (extremal)...\n")

# grid2 = grid([5,5])
# grid2.add_end([1,4])
# for i in [1,2]:
#     for j in [1,2]:
#         grid2.add_danger([i,j])
# grid2.add_danger([0,1])
# grid2.add_danger([3,1])
# grid2.add_danger([4,3])
# grid2.print_env()

# step_, err_ = grid2.q_learning(200,50000)



# print("\nExtremal :")
# vE = extremal(grid2,0)
# vE1 = extremal(grid2,1)

# print("\n")
# generate_traj_v(grid2,vE,[0,0])
# generate_traj_v(grid2,vE1,[0,0])


