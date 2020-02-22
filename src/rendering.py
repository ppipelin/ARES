from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GLUT as GLUT
import math
import numpy as np
from primitives import *
from model import *
import glm

def strid(name):
	return name + '_ID'

class Renderer:


	def __init__(self, use_debug=False):
		self.M = glm.identity(glm.mat4x4)
		self.V = glm.identity(glm.mat4x4)
		self.P = glm.identity(glm.mat4x4)

		self.update_M = True
		self.update_V = True
		self.update_P = True
	

	def init_shaders(self, folder, shaders, GL_PRIMITIVE_TARGET):
		program_ID = glCreateProgram()
		SP = {'PID': program_ID, 'target': GL_PRIMITIVE_TARGET}
		

		for shader_name, GL_SHADER_TYPE in shaders:
			shader_filepath = folder + shader_name
			shader_code = open(shader_filepath, 'r').read()
			shader_ID = glCreateShader(GL_SHADER_TYPE)
			glShaderSource(shader_ID, shader_code)
			glCompileShader(shader_ID)
			if not glGetShaderiv(shader_ID, GL_COMPILE_STATUS):
				error = glGetShaderInfoLog(shader_ID)
				print('Failed to compile the shader ' + folder + shader_name)
				print(error)
				raise Exception('Failed to compile a shader!')
			
			glAttachShader(program_ID, shader_ID)

		glLinkProgram(program_ID)
		if not glGetProgramiv(program_ID, GL_LINK_STATUS):
			error = glGetProgramInfoLog(program_ID)
			print(error)
			raise Exception('Failed to link the shader program!', error)

		
		
		print('shader program: ', SP)
		print('init shader: done!')

		return SP

		
	def init_attrib_uni(self, SP, attribs, unis):
		glUseProgram(SP['PID'])

		for attrib_name in attribs:
			id = self.addAttribute(attrib_name, SP)
			SP[strid(attrib_name)] = id

		for uni_name, utype, value in unis:
			id = self.addUniform(uni_name, SP)
			SP[uni_name] = {'value': value, 'utype': utype, 'ID': id}
			if id == -1:
				print('Warning, the uniform ' + uni_name + ' is not recognized by the SP ' + str(SP['PID']))
			else:
				if utype == '3f':
					glUniform3f(id, value[0], value[1], value[2])
				elif utype == '4f':
					glUniform4f(id, value[0], value[1], value[2], value[3])
				elif utype == '1ui':
					glUniform1ui(id, value)
				else:
					raise('The uniform type ' + utype + ' is not recognized! And the value is ' + str(value))

		P_ID = self.addUniform('P', SP)
		V_ID = self.addUniform('V', SP)
		M_ID = self.addUniform('M', SP)

		SP['P_ID'] = P_ID
		SP['V_ID'] = V_ID
		SP['M_ID'] = M_ID

		glUseProgram(0)

		return SP


	def addAttribute(self, attrib_name, SP):
		id = glGetAttribLocation(SP['PID'], attrib_name)
		if id == -1:
			print(Warning('Failed to get the ID of the Attribute "'+attrib_name+'" of the program ' + str(SP['PID'])))
		return id

	def addUniform(self, uni_name, SP):
		id = glGetUniformLocation(SP['PID'], uni_name)
		if id == -1:
			print(Warning('Failed to get the ID of the Uniform "'+uni_name+'" of the program ' + str(SP['PID'])))
		return id

	# Allocate a big texture on the OpenGL context, and returns its ID and the max uv coords of the corresonding given height/width
	def init_background_texture(self, H, W):

		textID = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, textID) 
		Hpow2 = int(math.pow(2, math.ceil(math.log(H)/math.log(2))))
		Wpow2 = int(math.pow(2, math.ceil(math.log(W)/math.log(2))))
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, Wpow2, Hpow2, 0,  GL_RGB, GL_UNSIGNED_BYTE, np.zeros((Wpow2, Hpow2, 3), dtype =np.uint8))
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER , GL_NEAREST)
		return [textID, H/Hpow2, W/Wpow2]

	def switch_shader_type(self, SP):
		SP['uni_mode']['value'] = (SP['uni_mode']['value'] + 1) % 3
		glUseProgram(SP['PID'])
		glUniform1ui(SP['uni_mode']['ID'], SP['uni_mode']['value'])
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
		"""  Set view from a camera calibration matrix. """

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
		self.P = glm.perspective(fovy, aspect, near, far)
		self.update_P = True

		glViewport(0,0,W,H)

	def set_M(self, scale, size_scale, H_marker, W_marker):
		H = H_marker * size_scale
		W = W_marker * size_scale
		sM = glm.scale(glm.mat4(), glm.vec3(scale, -scale, scale)) 
		tM = glm.translate(glm.mat4(), glm.vec3(W/2, 0, -H/2))
		self.M = tM * sM
		self.update_M = True

	def nparray_to_glm_mat(self, array):
		res = glm.mat4()
		for i in range(4):
			for j in range(4):
				tmp = array[i, j]
				res[i][j] = tmp
		return res
		
		

	def set_V_from_camera(self, cTw, t):
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

		self.V = self.nparray_to_glm_mat(viewMatrix) * glm.rotate(glm.mat4(), math.pi/2, glm.vec3(1, 0, 0))
		#V = glm.translate(glm.mat4(), glm.vec3(0, 0, -10*2))# * glm.rotate(glm.mat4(), math.pi, glm.vec3(1, 0, 0))

		self.update_V = True
		

	def render_model(self, model, t, SP):
		glUseProgram(SP['PID'])

		if SP['M_ID'] != -1 and self.update_M:
			glUniformMatrix4fv(SP['M_ID'], 1, GL_FALSE, glm.value_ptr(self.M))
			self.update_M = False
		if SP['V_ID'] != -1 and self.update_V:
			glUniformMatrix4fv(SP['V_ID'], 1, GL_FALSE, glm.value_ptr(self.V))
			self.update_V = False
		if SP['P_ID'] != -1 and self.update_P:
			glUniformMatrix4fv(SP['P_ID'], 1, GL_FALSE, glm.value_ptr(self.P))
			self.update_P = False

		glEnable(GL_DEPTH_TEST)
		glBindTexture(GL_TEXTURE_2D, 0) 
		
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