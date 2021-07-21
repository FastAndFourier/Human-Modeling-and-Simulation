from HanoiEnv import HanoiEnv
from IPython.display import clear_output 
import time
import pygame

env = HanoiEnv(6)

q_table = env.q_learning()

RENDER = 1

done = 0
state = tuple(env.reset([0]*env.nb_disk))

it = 0

while not(done):
  if RENDER:
    env.render()
    time.sleep(1)
  action = env.select_action(state,q_table)
  new_s, reward, done = env.step(action)

  state = tuple(new_s)
  
  it+=1

print(it,"iterations","Optimal ?",it==(2**env.nb_disk)-1)


if RENDER:
  stop_display = False

  env.render()
  time.sleep(1)

  while not(stop_display):
    for e in pygame.event.get():
      if e.type == pygame.QUIT:
        stop_display = True
    
