from HanoiEnv import HanoiEnv, distance_matrix, distanceDTW
from IPython.display import clear_output 
import time
import pygame
import numpy as np

env = HanoiEnv(3,"env")



path = "./Q-Table/qtable1_3.npy"
q_table = np.load(open(path,"rb"))
env.set_optimal_policy(q_table)
# q_table = env.q_learning()
# np.save(path,q_table)

v_boltz = env.boltzmann_rational(100)
print(v_boltz,"\n",env.value_iteration())
traj_boltz = env.demonstration_from_v([0]*env.nb_disk,v_boltz,100)
print(traj_boltz,len(traj_boltz)-1)

env.reset([0]*env.nb_disk)
for t in traj_boltz:
  env.reset(list(t))
  env.render()
  time.sleep(0.5)


RENDER = 1

# done = 0
# state = env.reset([0]*env.nb_disk)#env.reset(np.random.randint(3,size=env.nb_disk))
# print("Start ",state)

# it = 0

# while not(done):
#   if RENDER:
#     env.render()
#     time.sleep(0.5)
#   action = env.select_action_q(state,q_table)
#   new_s, reward, done = env.step(action)

#   state = tuple(new_s)

#   print(env.distance_state(list(env.state),env.final_state))
  
#   it+=1

# print(it,"iterations (optimal if p0-problem :",(2**env.nb_disk)-1,")")
# #print(it,"iterations","Optimal ?",it==(2**env.nb_disk)-1)



traj_boltz0 = env.boltzmann_traj_qtable(q_table,0.5)
traj_boltz1 = env.boltzmann_traj_qtable(q_table,10)

optimal = env.get_optimal_traj([0]*env.nb_disk)
print(optimal)

print(len(traj_boltz0),len(traj_boltz1))


for k in range(min(len(traj_boltz0),len(traj_boltz1))):
  print(traj_boltz0[k],traj_boltz1[k])

for k in range(min(len(traj_boltz0),len(traj_boltz1)),len(traj_boltz0)):
  print(traj_boltz0[k])

distance_matrix = distance_matrix(env,traj_boltz0,traj_boltz1)
print(distanceDTW(traj_boltz0,traj_boltz1,distance_matrix))


env.reset([0]*env.nb_disk)
for t in traj_boltz0:
  env.reset(list(t))
  env.render()
  time.sleep(0.5)

env.reset([0]*env.nb_disk)
for t in traj_boltz1:
  env.reset(list(t))
  env.render()
  time.sleep(0.5)





if RENDER:
  stop_display = False

  env.render()
  time.sleep(1)

  while not(stop_display):
    for e in pygame.event.get():
      if e.type == pygame.QUIT:
        stop_display = True
    
