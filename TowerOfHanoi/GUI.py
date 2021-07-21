aimport pygame
import time

HEIGHT = 40
WIDTH4 = 200
WIDTH3 = 150
WIDTH2 = 100
WIDTH1 = 50


def drawEnv(win,state):

	win.fill((255,255,255))

	pygame.draw.rect(win,(128,128,128),(100,100,10,300)) #left pole
	pygame.draw.rect(win,(128,128,128),(300,100,10,300)) #middle pole
	pygame.draw.rect(win,(128,128,128),(500,100,10,300)) #right pole



	width = [WIDTH1,WIDTH2,WIDTH3,WIDTH4]
	color = [(255,0,0),(0,255,0),(0,0,255),(255,0,255)]

	position_height = [0]*len(state)
	state_reverse = state[::-1]
	for i,s in enumerate(state_reverse[1:]):
		if s in state_reverse[:i+1]:
			index_under = state_reverse[:i+1][::-1].index(s)+i
			position_height[i+1] = position_height[index_under] + 1

	position_height = position_height[::-1]


	for s,w,c,p in zip(state[::-1],width[::-1],color[::-1],position_height[::-1]):
		if s==0:
			pygame.draw.rect(win,c,(100-w//2 + 5,280-p*20,w,20))
		if s==1:
			pygame.draw.rect(win,c,(300-w//2 + 5,280-p*20,w,20))
		if s==2:
			pygame.draw.rect(win,c,(500-w//2 + 5,280-p*20,w,20))



	pygame.display.update()


pygame.init()

win = pygame.display.set_mode((600,300))
pygame.display.set_caption("Tower of Hanoi")


sequence = [[0,0,0],[2,0,0],[2,1,0],[1,1,0],[1,1,2],[0,1,2],[0,2,2],[2,2,2]]

for s in sequence:
	drawEnv(win,s)
	time.sleep(1)


run = True
while run:
	pygame.time.delay(100)
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False
pygame.quit()
