import cv2
import argparse
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
def main(data_folder, descriptor_choice, extra_desc_param, do_calibration):
	
	video_path = data_folder + 'video_plateau.mp4'
	print('loading video from path : ' +  video_path +'...')	
	video= load_video(video_path)
	[N, H, W, C] =  video.shape
	
	# """"""precisely estimated calibration"""""""""
	F = 800
	u0 = W/2
	v0 = H/2
	K = np.matrix([[F, 0, u0], [0, F, v0], [0, 0, 1]])

	angle = 0;#-np.pi/4
	ciTw = np.matrix([[1,0,0,u0],[0, np.cos(angle), -np.sin(angle), v0],[0, np.sin(angle), np.cos(angle), -F/8], [0,0,0,1]])
	

	
	print('Pygame initilization...')
	pygame.init()
	pygame.display.set_caption('ARES')
	window = pygame.display.set_mode((W,H), DOUBLEBUF | OPENGL)
	
	FPS = 30.0
	TPF = 1.0/FPS
	n = 90
	
	print('Background video texture initialization...')
	[textID, y, x] = init_background_texture(H, W)

	
	print('Detector creation and feature detection on model...')
	detector = cv2.xfeatures2d.SURF_create() if descriptor_choice == 'surf' else cv2.xfeatures2d.SIFT_create() if descriptor_choice == 'sift' else cv2.ORB_create(int(opt.extra_desc_param)) if descriptor_choice == 'orb' else None
	
	if detector is None:
		raise Exception("The provided descriptor_choice (" + descriptor_choice + ") is not valid")
	print("detector", type(detector))

	if descriptor_choice == 'surf':
		detector.setHessianThreshold(int(opt.extra_desc_param))
	
	model = cv2.imread(data_folder + 'plateau.png')
	H_model, W_model, C_model = model.shape
	model = cv2.cvtColor(model,cv2.COLOR_BGR2GRAY)
	
	kp_model, des_model = detector.detectAndCompute(model, None)
	print('Keypoints/Descriptors : '+str(len(kp_model)))

	model = cv2.drawKeypoints(model,kp_model,model) 
	cv2.imshow('model', model)
	#cv2.waitKey(0)
	flann_params = dict(algorithm = 6, table_number = 6, key_size = 12, multi_probe_level = 1)
	
	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if descriptor_choice == 'orb' else cv2.BFMatcher(cv2.NORM_L1, crossCheck=True) if descriptor_choice == 'sift' or descriptor_choice == 'surf' else None
	if matcher is None:
		raise Exception("The provided descriptor_choice (" + descriptor_choice + ") is not valid")

	
	#matcher = cv2.FlannBasedMatcher(flann_params, {})
	min_match = 15; #render anything only if nb_matches > min_match
	
	print('Ready')
	while True:
		begin_t = time.time()
		frame = video[n,:,:,:]
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		clear(frame, H, W, y, x, textID)
		
		# 1/ Do the pose estimation
		
		kp_frame, des_frame = detector.detectAndCompute(gray, None)
		matches = matcher.match(des_model, des_frame)
		#matches = matches_ratio_test(matcher, des_model, des_frame, 0.75)
		#print(len(matches))
		
		matches = sorted(matches, key=lambda x: x.distance)
		
		if len(matches) > min_match:
			#cv2.drawKeypoints(frame,kp_frame,frame) 	# par ref
			#cap = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:min_match], 0, flags=2)
			#cv2.imshow('frame', cap[...,::-1]) # rgb->bgr, cv2.imshow takes bgr images...
			#cv2.waitKey(0)
			src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
			dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
			Homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			#Homography = -Homography
			# print('Homography')
			# print(Homography)
			if Homography is not None:
				#pts = np.float32([[0, 0], [0, H_model - 1], [W_model - 1, H_model - 1], [W_model - 1, 0]]).reshape(-1, 1, 2)
				#dst = cv2.perspectiveTransform(pts, Homography)
				#cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
				cTci = compute_cTci(K, Homography)
				cTw = np.dot(cTci,ciTw)
				# print('cTw')
				# print(cTw)
				render_cube(cTw, K, H, W, n * TPF)
		
		
		
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
	video = []
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			video.append(frame[...,::-1]) #bgr to rgb
		else:
			break
	return np.array(video)
	
#https://medium.com/@ahmetozlu93/marker-less-augmented-reality-by-opencv-and-opengl-531b2af0a130
def matches_ratio_test(matcher, des_1, des_2, min_ratio = 0.75):
	matches = matcher.knnMatch(des_1, des_2, 2)
	two_matches = filter(lambda x: len(x) ==2, matches)
	better_matches = filter(lambda x: x[0].distance < x[1].distance*min_ratio, two_matches)
	return list(map(lambda x : x[0], better_matches))

#thanks python for offering basic functions
def normalize(v):
	norm = np.linalg.norm(v)
	if norm == 0: 
		return v
	return v / norm

def compute_cTci(K,homography):

	# Compute rotation along the x and y axis as well as the translation
	# rot_and_transl = np.dot(np.linalg.inv(K), homography)
	# print(rot_and_transl)
	# c1 = rot_and_transl[:, 0]
	# c2 = rot_and_transl[:, 1]
	# citc = rot_and_transl[:, 2]
	# c3 = np.cross(c1,c2, axis = 0)
	# cRtci = np.column_stack((c1, c2, c3, citc));
	# print('cRtci')
	# print(cRtci)
	# return np.vstack([cRtci, [0,0,0,1]])

	#Compute rotation along the x and y axis as well as the translation
	#homography = homography * (-1)
	rot_and_transl = np.dot(np.linalg.inv(K), homography)
	col_1 = rot_and_transl[:, 0]
	col_2 = rot_and_transl[:, 1]
	col_3 = rot_and_transl[:, 2]
	
	# print('rot_and_transl')
	# print(rot_and_transl)
	# print('Kinv')
	# print(np.linalg.inv(K))
	#normalise vectors
	l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
	rot_1 = col_1 / l
	rot_2 = col_2 / l
	translation = col_3 / l
	#compute the orthonormal basis
	c = rot_1 + rot_2

	p = np.cross(rot_1, rot_2, axis = 0)
	d = np.cross(c, p, axis = 0)
	rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
	rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
	rot_3 = np.cross(rot_1, rot_2, axis = 0)

	#finally, compute the 3D projection matrix from the model to the current frame
	cTci = np.column_stack((rot_1, rot_2, rot_3, translation))
	cTci = np.vstack([cTci, [0,0,0,1]])
	# print('cTci')
	# print(cTci)
	return cTci

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--data_folder", type=str, required=False, default="data/")
	parser.add_argument("-desc", "--descriptor", type=str, required=False, choices=['sift', 'surf', 'orb'], default='surf')
	parser.add_argument('-e','--extra_desc_param', type=int, required=False, default=2500)
	parser.add_argument('-c', '--calibration', dest='do_calibration', action='store_true')
	parser.set_defaults(do_calibration=False)
	opt = parser.parse_args()
	
	print("#" * 100)
	print("Launching main.py with the following parameters : ")
	print("data_folder			", opt.data_folder)
	print("descriptor			", opt.descriptor)
	print("extra_desc_param		", opt.extra_desc_param)
	print("do_calibration			", opt.do_calibration)
	print("#" * 100)

	main(opt.data_folder, opt.descriptor, opt.extra_desc_param, opt.do_calibration)
