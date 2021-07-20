from HanoiEnv import HanoiEnv
from IPython.display import clear_output 
import time

env = HanoiEnv(4)

q_table = env.q_learning()

RENDER = 1

done = 0
state = tuple(env.reset([0]*env.nb_disk))

it = 0

print(q_table[env.sub2lin(state)])
clear_output()
while not(done):
  if RENDER:
    env.render()
    time.sleep(1)
    clear_output()
  action = env.select_action(state,q_table)
  new_s, reward, done = env.step(action)

  state = tuple(new_s)
  
  it+=1

if RENDER:
  env.render()
  time.sleep(1)
print(it,"iterations","Optimal ?",it==(2**env.nb_disk)-1)