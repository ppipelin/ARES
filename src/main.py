import cv2 

import pygame
import numpy as np
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

from primitives import *

import math
import time
from rendering import *
from pose_estimation import *


def main():
	
	pygame.init()
	
	pygame.display.set_caption('ARES')

	video= load_video('data/video.mp4')
	[N, H, W, C] =  video.shape
	window = pygame.display.set_mode((W,H), DOUBLEBUF | OPENGL)
	FPS = 30.0
	TPF = 1.0/FPS
	n = 0	
	
	[textID, y, x] = init_background_texture(H, W)
	
	while True:
		begin_t = time.time()
		
		frame = video[n,:,:,:]
		clear(frame, H, W, y, x, textID)
		
		# 1/ Do the pose estimation
		
		# 2/ Render an object
		
		pygame.display.flip()
		n = (n + 1) % N
		time.sleep(max(0,TPF - time.time() + begin_t))
					
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.display.quit()
				pygame.quit()
				exit()
	

	
	pygame.quit()


# From a video file path, returns a numpy array [nb frames, height , width, channels] 
def load_video(path):
	cap = cv2.VideoCapture(path)
			
	video = []
	
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			video.append(frame)
		else:
			break
	return np.array(video)
	



main()
