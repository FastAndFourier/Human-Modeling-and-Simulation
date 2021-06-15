import matplotlib.pyplot as plt
from Grid import *
from boltzmann_rational import *
from irrational_behavior import *
import numpy as np

oox = grid([5,6])
oox.add_end([2,5])
oox.add_danger([2,4])
oox.add_danger([2,3])
oox.add_danger([3,4])
oox.add_danger([4,4])
oox.add_danger([4,3])
oox.add_danger([4,2])
oox.add_danger([4,1])

oox.print_env()

print(oox.tab)
step_, err_ = oox.q_learning(50,10001)

v = boltz_rational(oox,50)
print(v.reshape(oox.size))


"""
plt.plot(err_)
plt.title("Temporal difference")
plt.show()


n_demo = 5
tau = 0.02

oox.rand_reset()

print("Tau = 0.02")
demonstration,start_= boltz_rational_noisy(oox,tau,n_demo,oox.start)
print("Mean demonstration length =",sum([np.size(s) for s in demonstration ])//n_demo)
oox.reset(oox.start)
oox.print_env()
print("Start =",start_,end="\n\n")
for k in range(n_demo):
    print("  * Demonstration #",k+1,"=",action2str(demonstration[k]))

print("\nTau = 0.1")
tau=0.1
demonstration,start_= boltz_rational_noisy(oox,tau,n_demo,oox.start)
print("Mean demonstration length =",sum([np.size(s) for s in demonstration ])//n_demo)
oox.reset(oox.start)
oox.print_env()
print("Start =",start_,end="\n\n")
for k in range(n_demo):
    print("  * Demonstration #",k+1,"=",action2str(demonstration[k]))

print("\nTau = 1")
tau=1
demonstration,start_= boltz_rational_noisy(oox,tau,n_demo,oox.start)
print("Mean demonstration length =",sum([np.size(s) for s in demonstration ])//n_demo)
oox.reset(oox.start)
oox.print_env()
print("Start =",start_,end="\n\n")
for k in range(n_demo):
    print("  * Demonstration #",k+1,"=",action2str(demonstration[k]))
"""
