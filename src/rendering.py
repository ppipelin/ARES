from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GLUT as GLUT
import math
import numpy as np
from primitives import *
from model import *
import glm

class Renderer:


	def __init__(self):
		self.SP = {}


	def init_shaders(self, shader_folder):
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

		self.SP = {
			'vert_ID': vert_ID, 
			'frag_ID': frag_ID, 
			'PID': program_ID,
			}

		self.addAttribute('in_position')
		self.addAttribute('in_normal')
		self.addAttribute('in_uv')

		self.addUniform('uni_mat_V')
		self.addUniform('uni_mat_P')
		self.addUniform('uni_mat_M')

		self.addUniform('uni_WlightDirection')
		self.addUniform('uni_lightColor')

		self.addUniform('uni_mode')

		self.addUniform('uni_diffuse')
		self.addUniform('uni_glossy')
		self.addUniform('uni_ambiant')
		
		print('shader program: ', self.SP)
		print('init shader: done!')

		glUseProgram(program_ID)
		glUniform3f(self.SP['uni_ambiant_ID'], 0.2, 0.2, 0.2)
		glUniform3f(self.SP['uni_lightColor_ID'], 1, 0.8, 0.6)
		glUniform3f(self.SP['uni_WlightDirection_ID'], 0, 1, 0)

		glUniform3f(self.SP['uni_diffuse_ID'], 0.5, 0.5, 0.5)
		glUniform4f(self.SP['uni_glossy_ID'], 1, 1, 1, 50)
		glUniform1ui(self.SP['uni_mode_ID'], 0)
		self.SP['uni_mode'] = 0
		glUseProgram(0)


	def addAttribute(self, attrib_name):
		self.SP[attrib_name+'_ID'] = glGetAttribLocation(self.SP['PID'], attrib_name)
		if self.SP[attrib_name+'_ID'] == -1:
			print(Warning('Failed to get the ID of the Attribute "'+attrib_name+'"'))
		return self.SP[attrib_name+'_ID']

	def addUniform(self, uni_name):
		self.SP[uni_name+'_ID'] = glGetUniformLocation(self.SP['PID'], uni_name)
		if self.SP[uni_name+'_ID'] == -1:
			print(Warning('Failed to get the ID of the Uniform "'+uni_name+'"'))
		return self.SP[uni_name+'_ID']

	# Allocate a big texture on the OpenGL context, and returns its ID and the max uv coords of the corresonding given height/width
	def init_background_texture(self, H, W):

		textID = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, textID) 
		Hpow2 = int(math.pow(2, math.ceil(math.log(H)/math.log(2))))
		Wpow2 = int(math.pow(2, math.ceil(math.log(W)/math.log(2))))
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, Wpow2, Hpow2, 0,  GL_RGB, GL_UNSIGNED_BYTE, np.zeros((Wpow2, Hpow2, 3), dtype =np.uint8))
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER , GL_NEAREST)
		return [textID, H/Hpow2, W/Wpow2]

	def switch_shader_type(self):
		self.SP['uni_mode'] = (self.SP['uni_mode'] + 1) % 3
		glUseProgram(self.SP['PID'])
		glUniform1ui(self.SP['uni_mode_ID'], self.SP['uni_mode'])
		glUseProgram(0)

	# 1/ Clears the buffers (color and depth) and DEACTIVATES depth tests (need to reactivate later!!)
	# 2/ Binds the texture and blit the given image to it
	# 3/ Sets a orthogrphic projection over the area W/H
	# 4/ draw a quad in the camera field (0,0,W,H) with x and y as the maximum uv texture coordinates
	# => sets the background with the given image
	def clear(self, image, H, W, y, x, textID):
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

	def set_P_from_camera(self, K, H, W):
		glUseProgram(self.SP['PID'])
		"""  Set view from a camera calibration matrix. """
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()

		fx = K[0,0]
		fy = K[1,1]
		fovy = 2*np.arctan(0.5*H/fy)
		aspect = (W*fy)/(H*fx)
		# print('fovy '+ str(fovy))
		# print('aspect '+ str(aspect))
		# define the near and far clipping planes
		near = 0.1
		far = 100.0
		
		# set perspective
		mP = glm.perspective(fovy, aspect, near, far)
		glUniformMatrix4fv(self.SP['uni_mat_P_ID'], 1, False, glm.value_ptr(mP))

		glViewport(0,0,W,H)
		glUseProgram(0)

	def set_M(self, scale, size_scale, H_marker, W_marker):
		glUseProgram(self.SP['PID'])
		H = H_marker * size_scale
		W = W_marker * size_scale
		sM = glm.scale(glm.mat4(), glm.vec3(scale, -scale, scale)) 
		tM = glm.translate(glm.mat4(), glm.vec3(W/2, 0, -H/2))
		M = tM * sM
		glUniformMatrix4fv(self.SP['uni_mat_M_ID'], 1, False, glm.value_ptr(M))
		glUseProgram(0)

	def nparray_to_glm_mat(self, array):
		res = glm.mat4()
		for i in range(4):
			for j in range(4):
				tmp = array[i, j]
				res[i][j] = tmp
		return res
		
		

	def set_V_from_camera(self, cTw, t):
		glUseProgram(self.SP['PID'])
		"""  Set the model view matrix from camera pose. """

		cv_to_gl = np.zeros((4, 4))
		
		cv_to_gl[0, 0] = 1
		cv_to_gl[3, 3] = 1
		#invert the y and z axis
		cv_to_gl[1, 1] = -1
		cv_to_gl[2, 2] = -1

		viewMatrix = np.dot(cv_to_gl, cTw)
		# viewMatrix[0,3] *= 0.01 # cm to m
		# viewMatrix[1,3] *= 0.01 # cm to m
		# viewMatrix[2,3] *= 0.01 # cm to m

		viewMatrix = viewMatrix.T

		V = self.nparray_to_glm_mat(viewMatrix) * glm.rotate(glm.mat4(), math.pi/2, glm.vec3(1, 0, 0))
		

		#V = glm.translate(glm.mat4(), glm.vec3(0, 0, -10*2))# * glm.rotate(glm.mat4(), math.pi, glm.vec3(1, 0, 0))
		

		# replace model view with the new matrix
		glUniformMatrix4fv(self.SP['uni_mat_V_ID'], 1, False, glm.value_ptr(V))
		glUseProgram(0)

	def render_model(self, model, t):
		glUseProgram(self.SP['PID'])
		glEnable(GL_DEPTH_TEST)
		glBindTexture(GL_TEXTURE_2D, 0) 
		
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		# glEnable(GL_CULL_FACE)
		# glCullFace(GL_FRONT)
		# glScale(0.1,0.1,0.1)
		# glRotate(-90,1,0,0)
		model.render(self.SP)
		#glDisable(GL_CULL_FACE)
		
		# glLoadIdentity()
		glDisable(GL_BLEND)

		glColor(255.0, 255.0, 255.0, 255.0)
		glUseProgram(0)