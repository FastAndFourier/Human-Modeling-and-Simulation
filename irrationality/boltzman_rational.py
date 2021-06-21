import irrationality.Grid

## Noisily-rational behaviour generation ##

def boltz_rational_noisy(grid_,tau,n,start_):
    # Tau : temperature coefficient
    # n : number of demonstrations generated from the same start

    grid_.reset(start_)
    a=[]
    for demo in range(n):
        done=0
        a.append([])
        while not(done):
            if grid_.start==grid_.end:
                break
            actions=grid_.q_table[:,ind2sub(grid_.size[1],grid_.state)]
            boltz_distribution = np.exp(actions/tau)/np.sum(np.exp(actions/tau))
            noisy_behaviour = np.random.choice([0,1,2,3],p=boltz_distribution)
            [new_state,reward,done] = grid_.step(noisy_behaviour)
            grid_.state=new_state
            a[demo].append(noisy_behaviour)
        a[demo] = np.array(a[demo])
        grid_.reset(start_)
    return a, grid_.start


def any_traj(grid_):

    for k in range(grid_.size[0]*grid_.size[1]):
        grid_.start=sub2ind(grid_.size[1],k)
        grid_.state=grid_.start
        done=False
        it=0
        action_=[]
        new_state=grid_.start
        while not(done) and it<100:
            if grid_.start==grid_.end:
                break
            action = np.argmax(grid_.q_table[:,ind2sub(grid_.size[1],new_state)])
            [new_state, reward, done] = grid_.step(action)
            grid_.state=new_state
            #print(new_state)
            it+=1
            action_.append(action)
        print("Start ",grid_.start,"->",it,"iterations",action2str(action_))
