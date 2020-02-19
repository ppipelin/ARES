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
from model import *

#http://chev.me/arucogen/
#https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/
def main(data_folder, descriptor_choice, extra_desc_param, do_calibration, shader_folder, save, model_name, video_name, unmute):
	
	video_path = data_folder + 'video_' + video_name + '.mp4'
	print('loading video from path : ' +  video_path +'...')	
	video= load_video(video_path)
	[N, H, W, C] =  video.shape
	# N = N//16
	if(save is not 'nosave'):
		video_to_save = np.empty((0, H, W, 3), dtype=np.uint8)

	# """"""precisely estimated calibration"""""""""
	F = 545#800#270
	u0 = W/2#440.35
	v0 = H/2#229.73#
	coords0 = np.array((u0,v0))
	K = np.matrix([[F, 0, u0], [0, F, v0], [0, 0, 1]])
	Kinv = np.linalg.inv(K)
	dist = np.array([[ 0.12740911, -0.37299138, -0.00393397,  0.000759 ,   0.43353899]])
	#angle = 0;#-np.pi/4
	#ciTw = np.matrix([[1,0,0,u0],[0, np.cos(angle), -np.sin(angle), v0],[0, np.sin(angle), np.cos(angle), -F/8], [0,0,0,1]])
	


	
	print('Pygame initilization...')
	pygame.init()
	pygame.display.set_caption('ARES')
	window = pygame.display.set_mode((W,H), DOUBLEBUF | OPENGL)
	
	init_shaders(shader_folder)

	FPS = 30.0
	TPF = 1.0/FPS
	n = 0

	print('Model loading...')
	model_path = 'data/models/' + model_name + '/'
	model = Model()
	model.load_from_obj(model_path+'model.obj')
	
	print('Background video texture initialization...')
	[textID, y, x] = init_background_texture(H, W)

	
	print('Detector creation and feature detection on model...')
	detector = cv2.xfeatures2d.SURF_create() if descriptor_choice == 'surf' else cv2.xfeatures2d.SIFT_create() if descriptor_choice == 'sift' else cv2.ORB_create(int(opt.extra_desc_param)) if descriptor_choice == 'orb' else None
	
	if detector is None:
		raise Exception("The provided descriptor_choice (" + descriptor_choice + ") is not valid")
	print("detector", type(detector))

	if descriptor_choice == 'surf':
		detector.setHessianThreshold(int(opt.extra_desc_param))
	
	marker = cv2.imread(data_folder + video_name + '.png')
	H_marker, W_marker, C_marker = marker.shape
	size_marker = min(H_marker, W_marker)
	size_scale = 12.5/size_marker # the width of the target measures 12.5 cm => will be 12.5 unit wide
	marker = cv2.cvtColor(marker,cv2.COLOR_BGR2GRAY)
	
	kp_marker, des_marker = detector.detectAndCompute(marker, None)
	print('Keypoints/Descriptors : '+str(len(kp_marker)))

	marker = cv2.drawKeypoints(marker,kp_marker,marker) 
	#cv2.imshow('marker', marker)
	#cv2.waitKey(0)
	flann_params = dict(algorithm = 6, table_number = 6, key_size = 12, multi_probe_level = 1)
	
	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if descriptor_choice == 'orb' else cv2.BFMatcher(cv2.NORM_L1, crossCheck=True) if descriptor_choice == 'sift' or descriptor_choice == 'surf' else None
	if matcher is None:
		raise Exception("The provided descriptor_choice (" + descriptor_choice + ") is not valid")

	
	#matcher = cv2.FlannBasedMatcher(flann_params, {})
	min_matches = 15 #render anything only if nb_matches > min_match

	# iframe = cv2.cvtColor(video[n,:,:,:], cv2.COLOR_RGB2GRAY)
	# ok_ciTw, ciTw, kp_iframe, des_iframe, imatches = compute_ciTw(K, dist, detector, matcher, iframe, kp_marker, des_marker, min(H_marker, W_marker), min_matches)
	# cv2.drawKeypoints(iframe,kp_iframe,iframe) 	# par ref
	# cap = cv2.drawMatches(marker, kp_marker, iframe, kp_iframe, imatches[:min_matches], 0, flags=2)
	# cv2.imshow('frame', cap[...,::-1]) # rgb->bgr, cv2.imshow takes bgr images...
	# cv2.waitKey(0)

	print('Ready')
	while True:
		if unmute:
			print('#' * 100)
			print('frame ' ,n)
		begin_t = time.time()
		t = n * TPF

		frame = video[n,:,:,:]
		clear(frame, H, W, y, x, textID)
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		
		ok_cTw, cTw, kp_frame, des_frame, matches = compute_ciTw(K, dist, detector, matcher, frame, kp_marker, des_marker,size_scale, min_matches)

		if ok_cTw:
			set_P_from_camera(K, H, W)
			set_V_from_camera(cTw, t)
			set_M(1, size_scale, H_marker, W_marker)

			render_model(model, n * TPF)
		# # 1/ Do the pose estimation
		# beg = time.time()
		# kp_frame, des_frame = detector.detectAndCompute(gray, None)
		# print("detection/description time : "+str(-beg + time.time()))
		# beg = time.time()
		# matches = matcher.match(des_iframe, des_frame)
		# print("matching              time : "+str(-beg + time.time()))
		# #matches = matches_ratio_test(matcher, des_marker, des_frame, 0.75)
		
		# #cv2.imshow('frame',gray)
		# matches = sorted(matches, key=lambda x: x.distance)
		# print("matches : " + str(len(matches)))
		# if len(matches) > min_matches:
		# 	# cv2.drawKeypoints(frame,kp_frame,frame) 	# par ref
		# 	# cap = cv2.drawMatches(iframe, kp_iframe, frame, kp_frame, matches[:min_matches], 0, flags=2)
		# 	# cv2.imshow('frame', cap[...,::-1]) # rgb->bgr, cv2.imshow takes bgr images...
			
		# 	src_pts = np.float32([kp_iframe[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
		# 	dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

		# 	E, mask = cv2.findEssentialMat(src_pts, dst_pts)
		# 	retval, R, t, mask  = cv2.recoverPose(E, src_pts, dst_pts, K)
		# 	print(R)
		# 	print(t)
		# 	# src_pts = np.float32([(np.array(kp_iframe[m.queryIdx].pt) - coords0)/F for m in matches]).reshape(-1, 1, 2)
		# 	# dst_pts = np.float32([(np.array(kp_frame[m.trainIdx].pt) - coords0)/F for m in matches]).reshape(-1, 1, 2)
		# 	beg = time.time()
		# 	Homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		# 	print("estimating homography time : "+str(-beg + time.time()))
		# 	#Homography = -Homography
		# 	# print('Homography')
		# 	# print(Homography)
		# 	if Homography is not None:
		# 		#pts = np.float32([[0, 0], [0, H_marker - 1], [W_marker - 1, H_marker - 1], [W_marker - 1, 0]]).reshape(-1, 1, 2)
		# 		#dst = cv2.perspectiveTransform(pts, Homography)
		# 		#cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
		# 		#cTci = compute_cTci(np.dot(Kinv,Homography), F, u0, v0)
		# 		#cTci = compute_cTci(Homography, F, u0, v0)
		# 		ok_cTci, cTci = compute_cTci_OpenCV(Homography, K)
		# 		#cTci = compute_cTci_simple(Homography, F, u0, v0)
		# 		if(ok_cTci):
		# 			cTw = np.dot(cTci, ciTw)
		# 			print('cTw')
		# 			print(cTw)			#cTw = ciTw
		# 			print('ciTw')
		# 			print(ciTw)
		# 			print('cTci')
		# 			print(cTci)
		# 			render_cube(cTw, K, H, W, n * TPF)
		# 		#render_model(model, cTw, K, H, W, n*TPF)
		
		
		
		#cv2.waitKey(0)
		# 2/ Render an object
		#=render_cube(H, W)
		pygame.display.flip()

		if(save is not 'nosave'):
			string_image = pygame.image.tostring(window, 'RGB')
			temp_surf = pygame.image.fromstring(string_image,(W, H),'RGB')
			tmp_arr = pygame.surfarray.array3d(temp_surf)
			tmp_arr = np.swapaxes(tmp_arr,0,1)
			video_to_save = np.append(video_to_save, [tmp_arr], axis=0)
			if((n + 1) % N is 0):
				break
		n = (n + 1) % N
		end_t = time.time()
		delta = end_t - begin_t

		if unmute:
			print(str(delta) + '/'+ str(TPF)+ '(' +str(delta/TPF)+')')
		
		time.sleep(max(0,TPF - delta))
		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
				switch_shader_type()		
			elif event.type == pygame.QUIT:
				pygame.display.quit()
				pygame.quit()
				exit()
	pygame.quit()
	if(save is not 'nosave'):
		if(save is None):
			save_file = descriptor_choice + '.mp4'
		else:
			save_file = save
		out = cv2.VideoWriter('results/'+str(save_file), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (W,H))
		for i in range(video_to_save.shape[0]):
			out.write(cv2.cvtColor(np.uint8(video_to_save[i]), cv2.COLOR_BGR2RGB))

def compute_ciTw(K, dist, detector, matcher, gray, kp_marker, des_marker, size_marker, min_matches):
	kp_firstframe, des_firstframe = detector.detectAndCompute(gray, None)
	matches = matcher.match(des_marker, des_firstframe)
	matches = sorted(matches, key=lambda x: x.distance)
	if len(matches) < min_matches:
		return False, None, kp_firstframe, des_firstframe, matches
	src_pts = np.float32([ np.array(kp_marker[m.queryIdx].pt + (0,))*size_marker for m in matches]).reshape(-1, 1, 3)
	dst_pts = np.float32([kp_firstframe[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
	retval, rvec, tvec, inliers	= cv2.solvePnPRansac(src_pts, dst_pts, K,  dist, None, None, False, 100, 5.0)
	if retval == False:
		return False, None, kp_firstframe, des_firstframe, matches
	rmat, jacobian = cv2.Rodrigues(rvec)
	ciTw = np.column_stack((rmat[:,0], rmat[:,1], rmat[:,2], tvec))
	ciTw = np.vstack([ciTw, [0,0,0,1]])
	return True, ciTw, kp_firstframe, des_firstframe, matches


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
	return np.array(video, dtype=np.uint8)
	
#https://medium.com/@ahmetozlu93/marker-less-augmented-reality-by-opencv-and-opengl-531b2af0a130
def matches_ratio_test(matcher, des_1, des_2, min_ratio = 0.75):
	matches = matcher.knnMatch(des_1, des_2, 2)
	two_matches = filter(lambda x: len(x) ==2, matches)
	better_matches = filter(lambda x: x[0].distance < x[1].distance*min_ratio, two_matches)
	return list(map(lambda x : x[0], better_matches))


def compute_cTci(Kinv_homography, F, u0, v0):
	#Compute rotation along the x and y axis as well as the translation

	rot_and_transl = Kinv_homography
	col_1 = rot_and_transl[:, 0]
	col_2 = rot_and_transl[:, 1]
	col_3 = rot_and_transl[:, 2]


	
	print('rot_and_transl')
	print(rot_and_transl)
	# print('Kinv')
	# print(np.linalg.inv(K))
	#normalise vectors
	l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
	rot_1 = col_1 / l
	rot_2 = col_2 / l
	translation = col_3# / l
	#compute the orthonormal basis
	c = rot_1 + rot_2

	p = np.cross(rot_1, rot_2, axis = 0)
	d = np.cross(c, p, axis = 0)
	rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
	rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
	rot_3 = np.cross(rot_1, rot_2, axis = 0)
	#finally, compute the 3D projection matrix from the marker to the current frame
	# translation[0] = (translation[0] + u0) 
	# translation[1] = (translation[1] + v0)
	# translation[2] = (translation[2] - F)

	# translation = - translation / F
	cTci = np.column_stack((rot_1, rot_2, rot_3, translation))
	cTci = np.vstack([cTci, [0,0,0,1]])
	# print('cTci')
	# print(cTci)
	return cTci

def compute_cTci_simple(Kinv_homography, F, u0, v0):
	#Compute rotation along the x and y axis as well as the translation

	rot_and_transl = Kinv_homography
	c1 = rot_and_transl[:, 0]
	c2 = rot_and_transl[:, 1]
	t = rot_and_transl[:, 2]
	c3 = np.cross(c1, c2, axis = 0)
	cTci = np.column_stack((c1, c2, c3, t))
	cTci = np.vstack([cTci, [0,0,0,1]])
	return cTci

def compute_cTci_OpenCV(H, K):
	retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
	# if retval is not 1:
	# 	print('fail : ', retval)
	# 	exit()
	if retval is 0:
		return False, None
	print(retval)
	rmat = rotations[0]
	tvec = translations[0]
	cTci = np.column_stack((rmat[:,0], rmat[:,1], rmat[:,2], tvec))
	cTci = np.vstack([cTci, [0,0,0,1]])
	return True, cTci

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--data_folder", type=str, required=False, default="data/", help="folder containing the data")
	parser.add_argument("-desc", "--descriptor", type=str, required=False, choices=['sift', 'surf', 'orb'], default='surf', help="descriptor choice")
	parser.add_argument('-e','--extra_desc_param', type=int, required=False, default=2500, help="extra descriptor parameter (number of desc)")
	parser.add_argument('-c', '--calibration', dest='do_calibration', action='store_true', help="choose to do the calibration")
	parser.add_argument('-sf', '--shader_folder', type=str, required=False, default = 'src/', help="select the shader folder (and so the shader)")
	parser.add_argument('-s', '--save', type=str, required=False, default='nosave', nargs='?', help="save the AR video? (and where)")
	parser.add_argument('-m', '--model', type=str, required=False, default='cube', help="choose the model to render")
	parser.add_argument('-v', '--video', type=str, required=False, default='book', help="select video (inside the data folder")
	parser.add_argument('-u', '--unmute', action='store_true', required=False, default=False, help='disable perf prints')

	parser.set_defaults(do_calibration=False)
	opt = parser.parse_args()
	
	print("#" * 100)
	print("Launching main.py with the following parameters : ")
	print("data_folder			", opt.data_folder)
	print("video				", opt.video)
	print("descriptor			", opt.descriptor)
	print("extra_desc_param		", opt.extra_desc_param)
	print("do_calibration			", opt.do_calibration)
	print("save			", opt.save)
	print("model		", opt.model)
	print("#" * 100)

	main(opt.data_folder, opt.descriptor, opt.extra_desc_param, opt.do_calibration, opt.shader_folder, opt.save, opt.model, opt.video, opt.unmute)
