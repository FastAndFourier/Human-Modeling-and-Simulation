#from Grid import *
import irrationality.Grid
import numpy as np


def value_iterate(env):
    v_vector = np.zeros((env.size[1]*env.size[0]))

    theta=0.5
    err=2

    #for k in range(2):
    while err>theta:
        err=0

        for s in range(env.size[0]*env.size[1]):
            if s == ind2sub(env.size[1],env.end):
                pass
            else:
                env.reset(sub2ind(env.size[1],s))
                v_temp = np.copy(v_vector)
                v = v_vector[s]
                action = np.argmax(env.q_table[:,s])
                [new_s,reward,done] = env.step(action)
                if new_s!=env.state:
                    v_vector[s] = reward + env.discount*v_temp[ind2sub(env.size[1],new_s)]

                err = max(err,abs(v_vector[s]-v))

    v_vector[ind2sub(env.size[1],env.end)] = env.tab[env.end[0],env.end[1]]
    return v_vector

def boltz_rational(env,beta):

    v_vector = np.random.rand(env.size[1]*env.size[0])
    #v_vector = np.zeros((env.size[1]*env.size[0]))

    theta=0.5
    err=2

    while err>theta:

        err=0

        for s in range(env.size[0]*env.size[1]):
            if s == ind2sub(env.size[1],env.end):
                pass
            env.reset(sub2ind(env.size[1],s))
            v_temp = np.copy(v_vector)
            v = v_vector[s]
            x = []
            for a in range(4):
                [new_s,reward,done] =.reshape(grid2.shape) env.step(a)
                if new_s!=env.state:
                    x.append(reward + env.discount*v_temp[ind2sub(env.size[1],new_s)])
                else:
                    x.append(0)
            x = np.array(x)
            v_vector[s] = np.sum(x*np.exp(x*beta))/np.sum(np.exp(x*beta))
            err = max(err,abs(v_vector[s]-v))

    v_vector[ind2sub(env.size[1],env.end)] = env.tab[env.end[0],env.end[1]]
    return v_vector


def prospect_bias(env,c):
    v_vector = np.random.rand(env.size[1]*env.size[0])
    #np.zeros((env.size[1]*env.size[0]))

    theta=0.5
    err=2

    while err>theta:
        err=0

        for s in range(env.size[0]*env.size[1]):
            if s == ind2sub(env.size[1],env.end):
                pass
            else:
                env.reset(sub2ind(env.size[1],s))
                v_temp = np.copy(v_vector)
                v = v_vector[s]
                action = np.argmax(env.q_table[:,s])
                [new_s,reward,done] = env.step(action)

                if reward > 0:
                    reward = np.log(1+abs(reward))
                elif reward==0:
                    reward = 0
                else:
                    reward = -c*np.log(1+abs(reward))

                if new_s!=env.state:
                    v_vector[s] = reward + env.discount*v_temp[ind2sub(env.size[1],new_s)]
                #else :
                    #v_vector[s] = env.discount*v_temp[ind2sub(env.size[1],new_s)]

                err = max(err,abs(v_vector[s]-v))

    v_vector[ind2sub(env.size[1],env.end)] = env.tab[env.end[0],env.end[1]]
    return v_vector


def extremal(env,alpha):

    v_vector = np.random.rand(env.size[0]*env.size[1])
    v_vector[ind2sub(env.size[1],env.end)] = env.tab[env.end[0],env.end[1]]
    theta=0.5
    err=2

    while err>theta:
        err=0

        for s in range(env.size[0]*env.size[1]):
            if s == ind2sub(env.size[1],env.end):
                pass
            else:
                env.reset(sub2ind(env.size[1],s))
                v_temp = np.copy(v_vector)
                v = v_vector[s]
                action = np.argmax(env.q_table[:,s])
                [new_s,reward,done] = env.step(action)

                reward = max(reward,(1-alpha)*reward + alpha*v_temp[ind2sub(env.size[1],new_s)])

                if new_s!=env.state:
                    v_vector[s] = reward + env.discount*v_temp[ind2sub(env.size[1],new_s)]
                #else :
                    #v_vector[s] = env.discount*v_temp[ind2sub(env.size[1],new_s)]

                err = max(err,abs(v_vector[s]-v))

    #v_vector[ind2sub(env.size[1],env.end)] = env.tab[env.end[0],env.end[1]]
    return v_vector

def generate_traj_v(env,v,start_):

    done=False
    env.reset(start_)
    it=0
    action_ = []

    while not(done) and it<200:
        if env.start == env.end:
            break

        v_choice = []
        for a in range(4):
            [ns,r,done] = env.step(a)
            v_choice.append(r + env.discount*v[ind2sub(env.size[1],ns)])

        action = np.argmax(v_choice)
        [new_s,reward,done] = env.step(action)
        env.state = new_s
        it+=1
        action_.append(action)
    if len(action_)==200:
        print("Démonstration trop longue (> 200 actions)")
    else:
        print("Start ",env.start,"->",it,"iterations",action2str(action_))
