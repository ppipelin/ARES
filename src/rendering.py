from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GLUT as GLUT
import math
import numpy as np
from primitives import *
from model import *

# Allocate a big texture on the OpenGL context, and returns its ID and the max uv coords of the corresonding given height/width
def init_background_texture(H, W):

	textID = glGenTextures(1)
	glBindTexture(GL_TEXTURE_2D, textID) 
	Hpow2 = int(math.pow(2, math.ceil(math.log(H)/math.log(2))))
	Wpow2 = int(math.pow(2, math.ceil(math.log(W)/math.log(2))))
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, Wpow2, Hpow2, 0,  GL_RGB, GL_UNSIGNED_BYTE, np.zeros((Wpow2, Hpow2, 3), dtype =np.uint8))
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER , GL_NEAREST)
	return [textID, H/Hpow2, W/Wpow2]


# 1/ Clears the buffers (color and depth) and DEACTIVATES depth tests (need to reactivate later!!)
# 2/ Binds the texture and blit the given image to it
# 3/ Sets a orthogrphic projection over the area W/H
# 4/ draw a quad in the camera field (0,0,W,H) with x and y as the maximum uv texture coordinates
# => sets the background with the given image
def clear(image, H, W, y, x, textID):
	#/1
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST)

	#/2
	glEnable(GL_TEXTURE_2D)
	glBindTexture(GL_TEXTURE_2D, textID) 
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, image)
	
	#/3
	glMatrixMode (GL_PROJECTION)
	glLoadIdentity()
	gluOrtho2D(0, W, H, 0)
	glMatrixMode(GL_MODELVIEW)
	#glLoadIdentity()
	#/4
	glBegin(GL_QUADS)
	glTexCoord2f(0, 0)
	glVertex2f(0, 0)
	glTexCoord2f(x, 0)
	glVertex2f(W, 0)
	glTexCoord2f(x, y)
	glVertex2f(W, H)
	glTexCoord2f(0, y)
	glVertex2f(0, H)
	glEnd()

def set_projection_from_camera(K, H, W):
	"""  Set view from a camera calibration matrix. """

	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()

	fx = K[0,0]
	fy = K[1,1]
	fovy = 2*np.arctan(0.5*H/fy)*180/np.pi
	aspect = (W*fy)/(H*fx)
	# print('fovy '+ str(fovy))
	# print('aspect '+ str(aspect))
	# define the near and far clipping planes
	near = 0.1
	far = 100.0
	
	# set perspective
	gluPerspective(fovy,aspect,near,far)
	glViewport(0,0,W,H)

# def set_modelview_from_camera(cTw):
	## """  Set the model view matrix from camera pose. """
	# Rt = cTw[:-1, :]

	##rotate teapot 90 deg around x-axis so that z-axis is up
	# Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]])

	##set rotation to best approximation
	# R = Rt[:,:3]
	# U,S,V = np.linalg.svd(R)
	# R = np.dot(U,V)
	# R[0,:] = -R[0,:] # change sign of x-axis
	
	##set translation
	# t = np.squeeze(Rt[:,3])

	##setup 4*4 model view matrix
	# M = np.eye(4)
	# M[:3,:3] = np.dot(R,Rx)
	# M[:3,3] = t
	
	# print('M')
	# print(M)
	##transpose and flatten to get column order
	# M = M.T
	# m = M.flatten() 

	##replace model view with the new matrix
	# glMatrixMode(GL_MODELVIEW)
	# glLoadIdentity()
	# glLoadMatrixf(m)

def set_modelview_from_camera(cTw):
	"""  Set the model view matrix from camera pose. """
	
	cv_to_gl = np.eye(4)
	cv_to_gl[1,1] = -cv_to_gl[1,1] # Invert the y axis
	cv_to_gl[2,2] = -cv_to_gl[2,2] # Invert the z axis
	viewMatrix = np.dot(cv_to_gl, cTw)
	viewMatrix[0,3] *= 0.01 # cm to m
	viewMatrix[1,3] *= 0.01 # cm to m
	viewMatrix[2,3] *= 0.01 # cm to m

	viewMatrix = viewMatrix.T
	viewMatrix = viewMatrix.flatten() 

	# replace model view with the new matrix
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	glLoadMatrixf(viewMatrix)
  
def render_cube(cTw, K, H,W,t):
	glEnable(GL_DEPTH_TEST)
	glBindTexture(GL_TEXTURE_2D, 0) 
	
	set_projection_from_camera(K, H, W)
	set_modelview_from_camera(cTw)
	
	glEnable(GL_BLEND)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	glEnable(GL_CULL_FACE)
	glCullFace(GL_FRONT)
	Cube(t)
	glDisable(GL_CULL_FACE)
	
	glLoadIdentity()
	glDisable(GL_BLEND)

	glColor(255.0, 255.0, 255.0, 255.0)

def render_model(model, cTw, K, H,W,t):
	glEnable(GL_DEPTH_TEST)
	glBindTexture(GL_TEXTURE_2D, 0) 
	
	set_projection_from_camera(K, H, W)
	set_modelview_from_camera(cTw)
	
	glEnable(GL_BLEND)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	glEnable(GL_CULL_FACE)
	glCullFace(GL_FRONT)
	glScale(0.1,0.1,0.1)
	glRotate(-90,1,0,0)
	#glEnable(GL_LIGHTING)
	#glEnable(GL_LIGHT0)
	#glLightfv( GL_LIGHT0, GL_POSITION, (0,0,10,1) )
	glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
	glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.0,0.0,0.0])
	glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0])
	glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)
	model.render()
	glDisable(GL_CULL_FACE)
	
	glLoadIdentity()
	glDisable(GL_BLEND)

	glColor(255.0, 255.0, 255.0, 255.0)