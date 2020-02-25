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
from filtering import *

#http://chev.me/arucogen/
#https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/
def main(data_folder, descriptor_choice, extra_desc_param, do_calibration, shader_folder, save, model_name, video_name, unmute, dsp):
	
	#cv2.waitKey(0)

	video_path = data_folder + 'video_' + video_name + '.mp4'
	print('loading video from path : ' +  video_path +'...')	
	video= load_video(video_path)
	[N, H, W, C] =  video.shape
	mean_video = np.mean(video, axis=0, dtype=np.float32)
	mean_video = mean_video.astype(np.uint8)
	# N = N//16
	if(save is not 'nosave'):
		video_to_save = np.empty((0, H, W, 3), dtype=np.uint8)

	# """"""precisely estimated calibration"""""""""
	F = 545#800#270
	u0 = W/2#440.35
	v0 = H/2#229.73#
	coords0 = np.array((u0,v0))
	K = np.matrix([[F, 0, u0], [0, F, v0], [0, 0, 1]])
	dist = np.array([[ 0.12740911, -0.37299138, -0.00393397,  0.000759 ,   0.43353899]])
	
	renderer = Renderer(H, W, dsp)

	
	print('Pygame initilization...')
	pygame.init()
	pygame.display.set_caption('ARES')
	window = pygame.display.set_mode((W,H), DOUBLEBUF | OPENGL)
	
	light_direction = (0, 1, 0)

	renderer.init_shader_data(shader_folder, light_direction)

	FPS = 30.0
	TPF = 1.0/FPS
	KF = KalmanFilter(TPF)
	n = 0

	print('Model loading...')
	model_path = 'data/models/' + model_name + '/'
	model = Model()
	model.load_from_obj(model_path+'model.obj')
	
	print('Background video texture initialization...')
	renderer.init_textures()
	renderer.set_mean_texture(mean_video)

	
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
	size_marker_cm = 12.5 if video_name is "book" else 40
	size_scale = size_marker_cm/size_marker # the width of the target measures 12.5 cm => will be 12.5 unit wide
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
	min_inliers = 25
	filtering_activated = True
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
			print('frame ' , n)
		begin_t = time.time()
		t = n * TPF

		frame = video[n,:,:,:]
		renderer.clear(frame)
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		
		rmat, tvec, nb_inliers = compute_cTw_solvePnP(K, dist, detector, matcher, frame, kp_marker, des_marker,size_scale, min_matches)
		#rmat, tvec, nb_inliers = compute_cTw_findHomography(K, dist, detector, matcher, frame, kp_marker, des_marker,size_scale, min_matches)


		if unmute:
			print(nb_inliers, " inliers (mininum to update Kalman filter : ", min_inliers,")")
		if nb_inliers > min_inliers:
			KF.fill(rmat, tvec)

		if filtering_activated == True:
			cTw = KF.predict()
		elif nb_inliers > 0:
			cTw = np.column_stack((rmat[:,0], rmat[:,1], rmat[:,2], tvec))
			cTw = np.vstack([cTw, [0,0,0,1]])
		
		renderer.set_P_from_camera(K)
		renderer.set_V_from_camera(cTw)
		renderer.set_M(1, size_scale, H_marker, W_marker)

		
		renderer.render_model(model, t)

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
				renderer.switch_shader_type()		
			elif event.type == pygame.KEYDOWN and event.key == pygame.K_k:
				filtering_activated = not filtering_activated
				print("filtering :" , filtering_activated)
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
	parser.add_argument('--dsp', action='store_true', required=False, default=False, help='debug Shaders')

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

	main(opt.data_folder, opt.descriptor, opt.extra_desc_param, opt.do_calibration, opt.shader_folder, opt.save, opt.model, opt.video, opt.unmute, opt.dsp)
