import numpy as np
from tqdm import tqdm
import pygame
import matplotlib.pyplot as plt
#from matplotlib.lines import Line2D

############# HELPER FUNCTIONS #################################################

def ind2sub(shape,s):
    return s[0]*shape + s[1]

def sub2ind(shape,i):
    return [i//shape,i%shape]

def action2str(demo):
    #Turn action index into str
    res=[]
    for i in demo:
        if i==0:
            res.append("North")
        elif i==1:
            res.append("West")
        elif i==2:
            res.append("South")
        else :
            res.append("East")

    return res

###################### GRID CLASS ##############################################

class grid:

    def __init__(self,s):
        #Initialization of the grid environment and the q_learning objects and parameters
        self.size = s

        self.value_end = 10
        self.value_danger = -10
        self.value_safe = -1

        self.tab = self.value_safe*np.ones((self.size[0],self.size[1]))#np.zeros((self.size[0],self.size[1]))
        self.start = [0,0]
        self.end = [self.size[0],self.size[1]]
        self.state = self.start

        self.q_table = np.zeros((4,self.size[0]*self.size[1]))

        self.discount=0.99
        self.lr = 0.1
        self.epsilon=0.1




    ## Creation of the maze ##
    def add_end(self,s):
        if not(s==self.start):
            self.tab[s[0],s[1]] = self.value_end
            self.end = s

        else:
            print("This position corresponds to the starting point!")


    def add_safe(self,s):
        if not(s==self.start):
            self.tab[s[0],s[1]] = self.value_safe
        else:
            print("This position corresponds to the starting point!")

    def add_danger(self,s):
        if not(s==self.start):
            self.tab[s[0],s[1]] = self.value_danger
        else:
            print("This position corresponds to the starting point!")



    ## State modification and display ##

    def step(self,o):

        new_s=[self.state[0],self.state[1]]

        if self.start==self.end:
            return self.end,10,True

        if o==0 and new_s[0]-1>=0: #North
            new_s[0]-=1
        elif o==1 and new_s[1]-1>=0: #West
            new_s[1]-=1
        elif o==2 and new_s[0]+1<self.size[0]:#South
            new_s[0]+=1
        elif o==3 and new_s[1]+1<self.size[1]:#East
            new_s[1]+=1


        if new_s==self.state:
            return self.state, 0, False

        else:
            #if proceed:
                #self.state = new_s

            return new_s, self.tab[new_s[0],new_s[1]], new_s==self.end

    # Random reset of the environment, e.g, for the q-learning
    def rand_reset(self):
        self.start=sub2ind(self.size[1],np.random.randint(0,self.size[0]*self.size[1]))
        self.state=[self.start[0],self.start[1]]

    # Reset at a specific position, e.g, for comparaison of noisy-rational demonstrations
    def reset(self,start):
        self.start=start
        self.state=start


    def print_env(self):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if (np.array([i,j])==self.state).all():
                    print("@",end="   ")
                elif (np.array([i,j])==self.start).all():
                    print("-",end="   ")
                elif (np.array([i,j])==self.end).all():
                    print("0",end="   ")
                elif self.tab[i,j]==self.value_safe:#0:
                    print("x",end="   ")
                else:
                    print("!",end="   ")

            print("\n")



    ## Q-learning ##

    def update_q(self,a,s,s1,r,done):

        s = ind2sub(self.size[1],s)
        s1 = ind2sub(self.size[1],s1)

        if done:
            td = r - self.q_table[a,s]
        else:
            td = r + self.discount*np.max(self.q_table[:,s1]) - self.q_table[a,s]

        self.q_table[a,s] = self.q_table[a,s] + self.lr*td

        return td


    def q_learning(self,limit_step,nb_episode):

        self.q_table = np.zeros((4,self.size[0]*self.size[1]))
        n_step=[]
        err_=[]


        for e in tqdm(range(nb_episode+1)):
            k=0
            done=False
            self.rand_reset()

            while k<limit_step and not(done):

                if self.start==self.end:
                    pass

                epsi = np.random.rand(1)[0]
                if epsi < self.epsilon:
                    action = np.random.randint(0,4)
                else:
                    action = np.argmax(self.q_table[:,ind2sub(self.size[1],self.state)])

                [new_state, reward, done] = self.step(action)
                
                err = self.update_q(action,self.state,new_state,reward,done)

                self.state = new_state
                k+=1
                err_.append(abs(err))

            n_step.append(k)


        return n_step, err_

    def render(self):

        pygame.init()

        blockSize = 50 

        HEIGHT = self.size[1]*blockSize
        WIDTH = self.size[0]*blockSize

        win = pygame.display.set_mode((HEIGHT,WIDTH),pygame.RESIZABLE)
        pygame.display.set_caption("Grid")

        win.fill((255,255,255))

        #Draw grid
        
        for x in range(0, HEIGHT, blockSize):
            for y in range(0, WIDTH, blockSize):
                rect = pygame.Rect(x, y, blockSize, blockSize)
                pygame.draw.rect(win, [0,0,0], rect, 1)

        #Fill grid
        for x in range(self.size[0]):
            for y in range(self.size[1]):

                #print()

                if [x,y] == self.start:
                    rect = pygame.Rect(y*blockSize+2, x*blockSize+2, blockSize-4, blockSize-4)
                    pygame.draw.rect(win, [0,0,255], rect)
                elif self.tab[x,y] == 10:
                    rect = pygame.Rect(y*blockSize+2, x*blockSize+2, blockSize-4, blockSize-4)
                    pygame.draw.rect(win, [0,255,0], rect)
                elif self.tab[x,y] == -10:
                    rect = pygame.Rect(y*blockSize+2, x*blockSize+2, blockSize-4, blockSize-4)
                    pygame.draw.rect(win, [100,100,100], rect)

        pygame.draw.circle(win, [255,0,0], [self.state[1]*blockSize+blockSize//2,self.state[0]*blockSize+blockSize//2],blockSize//4)

        pygame.display.update()

        run = True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()


        return win, run 

    def display_qtable(self):

        fig, ax = plt.subplots(1)
        img = 255*np.ones((self.size[1]*2,self.size[0]*2,3))
        

        dl = 0.25
        width = 0.1


        for k in range(self.size[0]):
            for l in range(self.size[1]):

                # if [k,l] == [0,0]:
                #     print("IN")
                #     img[0:2,0:1,:] = [0,255,0]
                    
                # if self.tab[k,l] == -10:
                #     img[l:l+2,k:k+2,:] = [100,100,100]


                # if [k,l] == self.end:
                #     img[k*2+1:k*2+3,l*2-2:l*2,:] = [0,255,0]
                #     continue

                action = np.argmax(self.q_table[:,k*self.size[1]+l])
                if action == 0:
                    plt.arrow(2*l+1,2*k+1,0,-dl,width=width)
                elif action == 1:
                    plt.arrow(2*l+1,2*k+1,-dl,0,width=width)
                elif action == 2:
                    plt.arrow(2*l+1,2*k+1,0,dl,width=width)
                else:
                    plt.arrow(2*l+1,2*k+1,dl,0,width=width)

        ax.set_aspect('equal')
        ax.set_xlim(0,self.size[1]*2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        extent = (0, self.size[1]*2, self.size[0]*2, 0)
        ax.imshow(img,extent=extent)
        ax.grid()
        #ax.imshow(self.grid_img,extent=extent)

        plt.show()


        




