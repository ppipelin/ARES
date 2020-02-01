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

#http://chev.me/arucogen/
#https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/
def main():
	
	video= load_video('data/video_ArUco_2.mp4')
	[N, H, W, C] =  video.shape
	
	print('Pygame initilization...')
	pygame.init()
	pygame.display.set_caption('ARES')
	window = pygame.display.set_mode((W,H), DOUBLEBUF | OPENGL)
	
	FPS = 30.0
	TPF = 1.0/FPS
	n = 0
	
	print('Background video texture initialization...')
	[textID, y, x] = init_background_texture(H, W)

	
	print('Detector creation and feature detection on model...')
	detector = cv2.xfeatures2d.SURF_create()
	#detector = cv2.xfeatures2d.SIFT_create()
	#detector= cv2.ORB_create(10000)
	detector.setHessianThreshold(2000)
	#detector.setUpright(True)
	
	model = cv2.imread('data/ArUco.png')
	H_model, W_model, C_model = model.shape
	
	# cv2.drawKeypoints(model,kp_model,model)
	# cv2.imshow('model', model)
	# cv2.waitKey(0)
	kp_model, des_model = detector.detectAndCompute(model, None)

	#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
	
	min_match = 15;
	
	print('Ready')
	while True:
		begin_t = time.time()
		frame = video[n,:,:,:]

		
		# 1/ Do the pose estimation
		
		kp_frame, des_frame = detector.detectAndCompute(frame, None)
		
		matches = bf.match(des_model, des_frame)
		matches = sorted(matches, key=lambda x: x.distance)
		
		if len(matches) > min_match:
			#cv2.drawKeypoints(frame,kp_frame,frame) 	# par ref
			#cap = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:min_match], 0, flags=2)
			#cv2.imshow('frame', cap[...,::-1]) # rgb->bgr, cv2.imshow takes bgr images...
			src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
			dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
			Homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			pts = np.float32([[0, 0], [0, H_model - 1], [W_model - 1, H_model - 1], [W_model - 1, 0]]).reshape(-1, 1, 2)
			dst = cv2.perspectiveTransform(pts, Homography)
			cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
		
		clear(frame, H, W, y, x, textID)
		#cv2.waitKey(0)
		# 2/ Render an object
		#=render_cube(H, W)
		pygame.display.flip()
		n = (n + 1) % N
		end_t = time.time()
		delta = end_t - begin_t
		print(str(delta) + '/'+ str(TPF)+ '(' +str(delta/TPF)+')')
		time.sleep(max(0,TPF - delta))
		
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.display.quit()
				pygame.quit()
				exit()
	
	
	pygame.quit()


# From a video file path, returns a numpy array [nb frames, height , width, channels] 
def load_video(path):
	cap = cv2.VideoCapture(path)
	print('loading video from path : ' +  path +'...')	
	video = []
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			video.append(frame[...,::-1]) #bgr to rgb
		else:
			break
	return np.array(video)
	



main()
