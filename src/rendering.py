from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GLUT as GLUT
import math
import numpy as np
from primitives import *
from model import *
import glm

SP = {}


def init_shaders(shader_folder):
	global SP
	vert_filepath = shader_folder + 'shader_vert.glsl'
	frag_filepath = shader_folder + 'shader_frag.glsl'
	vert_code = open(vert_filepath, 'r').read()
	frag_code = open(frag_filepath, 'r').read()

	vert_ID = glCreateShader(GL_VERTEX_SHADER)
	glShaderSource(vert_ID, vert_code)
	glCompileShader(vert_ID)
	if not glGetShaderiv(vert_ID, GL_COMPILE_STATUS):
		error = glGetShaderInfoLog(vert_ID)
		print(error)
		raise Exception('Failed to compile the vertex shader!', error)

	frag_ID = glCreateShader(GL_FRAGMENT_SHADER)
	glShaderSource(frag_ID, frag_code)
	glCompileShader(frag_ID)
	if not glGetShaderiv(frag_ID, GL_COMPILE_STATUS):
		error = glGetShaderInfoLog(frag_ID)
		print(error)
		raise Exception('Failed to compile the fragment shader!', error)

	program_ID = glCreateProgram()
	glAttachShader(program_ID, vert_ID)
	glAttachShader(program_ID, frag_ID)
	glLinkProgram(program_ID)

	if not glGetProgramiv(program_ID, GL_LINK_STATUS):
		error = glGetProgramInfoLog(program_ID)
		print(error)
		raise Exception('Failed to link the shader program!', error)

	SP = {
		'vert_ID': vert_ID, 
		'frag_ID': frag_ID, 
		'PID': program_ID,
		}

	addAttribute('in_position')
	addAttribute('in_normal')
	addAttribute('in_uv')

	addUniform('uni_mat_V')
	addUniform('uni_mat_P')
	addUniform('uni_mat_M')

	addUniform('uni_WlightDirection')
	addUniform('uni_lightColor')

	addUniform('uni_mode')

	addUniform('uni_diffuse')
	addUniform('uni_glossy')
	addUniform('uni_ambiant')
	
	print('shader program: ', SP)
	print('init shader: done!')

	glUseProgram(program_ID)
	glUniform3f(SP['uni_ambiant_ID'], 0.2, 0.2, 0.2)
	glUniform3f(SP['uni_lightColor_ID'], 1, 1, 1)
	glUniform3f(SP['uni_WlightDirection_ID'], 0, 0, 1)
	glUniform3f(SP['uni_diffuse_ID'], 1, 1, 1)
	glUseProgram(0)


def addAttribute(attrib_name):
	global SP
	SP[attrib_name+'_ID'] = glGetAttribLocation(SP['PID'], attrib_name)
	if SP[attrib_name+'_ID'] == -1:
		print(Warning('Failed to get the ID of the Attribute "'+attrib_name+'"'))
	return SP[attrib_name+'_ID']

def addUniform(uni_name):
	global SP
	SP[uni_name+'_ID'] = glGetUniformLocation(SP['PID'], uni_name)
	if SP[uni_name+'_ID'] == -1:
		print(Warning('Failed to get the ID of the Uniform "'+uni_name+'"'))
	return SP[uni_name+'_ID']

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
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
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
	assert(SP['uni_mat_P_ID'] != -1)
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
	mP = glm.perspective(fovy, aspect, near, far)
	glUniformMatrix4fv(SP['uni_mat_P_ID'], 1, False, glm.value_ptr(mP))

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

def nparray_to_glm_mat(array):
	res = glm.mat4()
	for i in range(4):
		for j in range(4):
			tmp = array[i, j]
			res[i][j] = tmp
	return res
	
	

def set_modelview_from_camera(cTw):
	"""  Set the model view matrix from camera pose. """
	assert(SP['uni_mat_V_ID'] != -1)

	cv_to_gl = np.eye(4)
	cv_to_gl[1,1] = -cv_to_gl[1,1] # Invert the y axis
	cv_to_gl[2,2] = -cv_to_gl[2,2] # Invert the z axis
	viewMatrix = np.dot(cv_to_gl, cTw)
	# viewMatrix[0,3] *= 0.01 # cm to m
	# viewMatrix[1,3] *= 0.01 # cm to m
	# viewMatrix[2,3] *= 0.01 # cm to m

	viewMatrix = viewMatrix.T

	V = nparray_to_glm_mat(viewMatrix)

	# replace model view with the new matrix
	glUniformMatrix4fv(SP['uni_mat_V_ID'], 1, False, glm.value_ptr(V))
	
  
def render_cube(cTw, K, H,W,t):
	glUseProgram(SP['PID'])
	glEnable(GL_DEPTH_TEST)
	glBindTexture(GL_TEXTURE_2D, 0) 
	
	set_projection_from_camera(K, H, W)
	set_modelview_from_camera(cTw)
	
	glEnable(GL_BLEND)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	#glEnable(GL_CULL_FACE)
	#glCullFace(GL_BACK)
	Cube(t)
	#glDisable(GL_CULL_FACE)
	
	glLoadIdentity()
	glDisable(GL_BLEND)

	glColor(255.0, 255.0, 255.0, 255.0)

def render_model(model, cTw, K, H,W,t):
	glUseProgram(SP['PID'])
	glEnable(GL_DEPTH_TEST)
	glBindTexture(GL_TEXTURE_2D, 0) 
	
	set_projection_from_camera(K, H, W)
	set_modelview_from_camera(cTw)
	
	glEnable(GL_BLEND)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	# glEnable(GL_CULL_FACE)
	# glCullFace(GL_FRONT)
	# glScale(0.1,0.1,0.1)
	# glRotate(-90,1,0,0)
	model.render(SP)
	#glDisable(GL_CULL_FACE)
	
	# glLoadIdentity()
	glDisable(GL_BLEND)

	glColor(255.0, 255.0, 255.0, 255.0)
	glUseProgram(0)