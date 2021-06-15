from Grid import *
import numpy as np

def boltz_rational(grid_,tau):

    v_vector = np.zeros((grid_.size[1]*grid_.size[0]))

    err=10
    while err>0.5:
        v_temp = np.copy(v_vector)
        for s in range(grid_.size[0]*grid_.size[1]):
            grid_.reset(sub2ind(grid_.size[1],s))
            x = np.zeros(4)
            vs_sum=0
            for a in range(4):
                [new_s,r,done] = grid_.step(a)

                if sub2ind(grid_.size[1],s)==grid_.end:
                    s_ = sub2ind(grid_.size[1],s)
                    new_s = s_
                    if a==0 and s_[0]-1>=0: #North
                        new_s[0]-=1
                    elif a==1 and s_[1]-1>=0: #West
                        new_s[1]-=1
                    elif a==2 and s_[0]+1<grid_.size[0]:#South
                        new_s[0]+=1
                    elif a==3 and s_[1]+1<grid_.size[1]:#East
                        new_s[1]+=1
                    #print(new_s)
                    r = grid_.tab[new_s[0],new_s[1]]

                if new_s!=sub2ind(grid_.size[1],s):
                    x[a] = r + grid_.discount*v_temp[ind2sub(grid_.size[1],new_s)]


            #print(x)

            #x = x+grid_.discount*sum_vs1
            v_vector[s] = max(x)#np.sum(x*np.exp(x/tau))/np.sum(np.exp(x/tau))
        err=np.mean(v_vector-v_temp)


        #print(v_vector.reshape(grid_.size))
    return v_vector
    """
    for s1 in range(grid_.size[0]*grid_.size[1]):


        new_state = grid_.state
        reward=0
        it=0
        while grid_.state != s1 and it<10:
            action = np.argmax(grid_.q_table[:,ind2sub(grid_.size[1],new_state)])
            [new_state,r,done] = grid_.step(action)
            #print(new_state)
            reward +=  r
            grid_.state = new_state
            it+=1

        print(reward)
    """

    """
    for k in range(np.size(v_vector)):
        v_vector[k] = np.max(grid_.q_table[:,k])



    return v_vector
    """
