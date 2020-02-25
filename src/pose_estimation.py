import cv2
import numpy as np
import math

def compute_cTw_solvePnP(K, dist, detector, matcher, gray, kp_marker, des_marker, size_marker, min_matches):
	kp_firstframe, des_firstframe = detector.detectAndCompute(gray, None)
	matches = matcher.match(des_marker, des_firstframe)
	matches = sorted(matches, key=lambda x: x.distance)
	if len(matches) < min_matches:
		return None, None,0
	src_pts = np.float32([ np.array(kp_marker[m.queryIdx].pt + (0,))*size_marker for m in matches]).reshape(-1, 1, 3)
	dst_pts = np.float32([kp_firstframe[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
	retval, rvec, tvec, inliers	= cv2.solvePnPRansac(src_pts, dst_pts, K,  dist, None, None, False, 100, 5.0)
	if retval == False:
		return None, None,0
	rmat, jacobian = cv2.Rodrigues(rvec)
	return rmat, tvec, len(inliers)

def compute_cTw_findHomography(K, dist, detector, matcher, gray, kp_marker, des_marker, size_marker, min_matches):
	kp_firstframe, des_firstframe = detector.detectAndCompute(gray, None)
	matches = matcher.match(des_marker, des_firstframe)
	matches = sorted(matches, key=lambda x: x.distance)
	if len(matches) < min_matches:
		return None, None,0
	src_pts = np.float32([np.array(kp_marker[m.queryIdx].pt)*size_marker for m in matches]).reshape(-1, 1, 2)
	dst_pts = np.float32([kp_firstframe[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
	Homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	rmat, tvec = cTw_from_Homography(Homography, K)
	return rmat, tvec, 50

def cTw_from_Homography(Homography, K):
	#Compute rotation along the x and y axis as well as the translation
	rot_and_transl = np.dot(np.linalg.inv(K), Homography)

	col_1 = rot_and_transl[:, 0]
	col_2 = rot_and_transl[:, 1]
	col_3 = rot_and_transl[:, 2]

	# print('Kinv')
	# print(np.linalg.inv(K))
	#normalise vectors
	l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
	rot_1 = col_1 / l
	rot_2 = col_2 / l
	tvec = col_3# / l
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
	rmat = np.column_stack((rot_1, rot_2, rot_3))
	print('rmat')
	print(rmat)
	print('tvec')
	print(tvec)
	return rmat, tvec
