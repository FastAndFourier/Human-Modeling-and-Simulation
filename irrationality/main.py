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

print(oox.tab)
oox.print_env()

step_, err_ = oox.q_learning(50,50001)

vi = value_iterate(oox)

beta = [0,0.025,0.1,1]
for b in beta:
    vB = boltz_rational(oox,b)
    generate_traj_v(oox,vB,[3,3])
    #print("V_boltz (beta = ",beta,") = \n",vB,"\n")
# v_from_q = np.zeros(v.shape)
# for k in range(np.size(v_from_q)):
#     print("k = ",k,oox.q_table[:,k])
#     v_from_q[k] = np.max(oox.q_table[:,k])
# print("V = Q(s,pi(s)) = \n",v_from_q)

grid1 = grid([5,5])
grid1.add_end([0,3])
for i in range(2):
    for j in range(1,2):
        grid1.add_danger([i,j])

for i in range(1,2):
    for j in range(2,4):
        grid1.add_danger([i,j])
grid1.print_env()
step_, err_ = grid1.q_learning(50,50001)

vP = prospect_bias(grid1,10)
vP1 = prospect_bias(grid1,100)
print("\n")
generate_traj_v(grid1,vP,[0,0])
generate_traj_v(grid1,vP1,[0,0])

"""
plt.plot(err_)
plt.title("Temporal difference")
plt.show()


n_demo = 10
tau = 0.02

oox.reset([3,3])
oox.reset(oox.start)
oox.print_env()

print("Tau = 0.02")
demonstration,start_= boltz_rational_noisy(oox,tau,n_demo,oox.start)
print("Mean demonstration length =",sum([np.size(s) for s in demonstration ])//n_demo)
print("Std demonstration length =",np.std([np.size(s) for s in demonstration ]))

oox.reset(oox.start)
oox.print_env()
print("Start =",start_,end="\n\n")
for k in range(n_demo):
    print("  * Demonstration #",k+1,"=",action2str(demonstration[k]))


print("\nTau = 0.1")
tau=0.1
demonstration,start_= boltz_rational_noisy(oox,tau,n_demo,oox.start)
print("Mean demonstration length =",sum([np.size(s) for s in demonstration ])//n_demo)
print("Std demonstration length =",np.std([np.size(s) for s in demonstration ]))

# oox.reset(oox.start)
# oox.print_env()
# print("Start =",start_,end="\n\n")
# for k in range(n_demo):
#     print("  * Demonstration #",k+1,"=",action2str(demonstration[k]))


print("\nTau = 1")
tau=1
demonstration,start_= boltz_rational_noisy(oox,tau,n_demo,oox.start)
print("Mean demonstration length =",sum([np.size(s) for s in demonstration ])//n_demo)
print("Std demonstration length =",np.std([np.size(s) for s in demonstration ]))

# oox.reset(oox.start)
# oox.print_env()
# print("Start =",start_,end="\n\n")
# for k in range(n_demo):
#     print("  * Demonstration #",k+1,"=",action2str(demonstration[k]))

"""
