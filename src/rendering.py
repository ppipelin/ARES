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


	def __init__(self, H, W, use_debug=False):
		self.M = glm.identity(glm.mat4x4)
		self.V = glm.identity(glm.mat4x4)
		self.P = glm.identity(glm.mat4x4)

		self.update_M = True
		self.update_V = True
		self.update_P = True

		self.H = H
		self.W = W
		self.x = 0
		self.y = 0
		self.SP = []
		self.use_debug = use_debug



	def init_shader_data(self, shader_folder, light_direction):
		if self.use_debug:
			self.init_shaders(shader_folder, [
				('vert.glsl', GL_VERTEX_SHADER),
				('geo.glsl', GL_GEOMETRY_SHADER),
				('line_frag.glsl', GL_FRAGMENT_SHADER),
			], GL_TRIANGLES)
			self.init_attrib_uni(['in_position', 'in_normal', 'in_uv'], [
				('uni_WlightDirection', '3f', light_direction),
			])
		else:
			self.init_shaders(shader_folder, [
			('vert.glsl', GL_VERTEX_SHADER), 
			('frag.glsl', GL_FRAGMENT_SHADER),
			], GL_TRIANGLES)	
			self.init_attrib_uni(['in_position', 'in_normal', 'in_uv'], [
				('uni_WlightDirection', '3f', light_direction),
				('uni_lightColor', '3f', (1, 1, 1)),
				('uni_diffuse', '3f', (0.5, 0.5, 0.5)),
				('uni_ambiant', '3f', (0.2, 0.2, 0.2)),
				('uni_glossy', '4f', (1, 1, 1, 1000)),
				('uni_mode', '1ui', 0),
				('uni_resolution', '2i', (self.W, self.H)),
			])
		


	def init_shaders(self, folder, shaders, GL_PRIMITIVE_TARGET):
		program_ID = glCreateProgram()
		self.SP = {'PID': program_ID, 'target': GL_PRIMITIVE_TARGET}

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
		
		print('shader program: ', self.SP)
		print('init shader: done!')
	
	def init_attrib_uni(self, attribs, unis):
		glUseProgram(self.SP['PID'])

		for attrib_name in attribs:
			id = self.addAttribute(attrib_name)
			self.SP[strid(attrib_name)] = id

		for uni_name, utype, value in unis:
			id = self.addUniform(uni_name)
			self.SP[uni_name] = {'value': value, 'utype': utype, 'ID': id}
			if id == -1:
				print('Warning, the uniform ' + uni_name + ' is not recognized by the SP ' + str(self.SP['PID']))
			else:
				if utype == '2i':
					glUniform2i(id, value[0], value[1])
				elif utype == '3f':
					glUniform3f(id, value[0], value[1], value[2])
				elif utype == '4f':
					glUniform4f(id, value[0], value[1], value[2], value[3])
				elif utype == '1ui':
					glUniform1ui(id, value)
				else:
					raise('The uniform type ' + utype + ' is not recognized! And the value is ' + str(value))

		P_ID = self.addUniform('P')
		V_ID = self.addUniform('V')
		M_ID = self.addUniform('M')

		self.SP['P_ID'] = P_ID
		self.SP['V_ID'] = V_ID
		self.SP['M_ID'] = M_ID

		glUseProgram(0)


	def addAttribute(self, attrib_name):
		id = glGetAttribLocation(self.SP['PID'], attrib_name)
		if id == -1:
			print(Warning('Failed to get the ID of the Attribute "'+attrib_name+'" of the program ' + str(self.SP['PID'])))
		return id

	def addUniform(self, uni_name):
		id = glGetUniformLocation(self.SP['PID'], uni_name)
		if id == -1:
			print(Warning('Failed to get the ID of the Uniform "'+uni_name+'" of the program ' + str(self.SP['PID'])))
		return id

	# Allocate a big texture on the OpenGL context, and returns its ID and the max uv coords of the corresonding given height/width
	def init_textures(self):
		self.SP['background_texture_ID'] = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.SP['background_texture_ID']) 
		Hpow2 = int(math.pow(2, math.ceil(math.log(self.H)/math.log(2))))
		Wpow2 = int(math.pow(2, math.ceil(math.log(self.W)/math.log(2))))
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, Wpow2, Hpow2, 0,  GL_RGB, GL_UNSIGNED_BYTE, np.zeros((Wpow2, Hpow2, 3), dtype =np.uint8))
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER , GL_NEAREST)
		self.y = self.H/Hpow2
		self.x = self.W/Wpow2

		self.SP['mean_texture_ID'] = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.SP['mean_texture_ID']) 
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, Wpow2, Hpow2, 0,  GL_RGB, GL_UNSIGNED_BYTE, np.zeros((Wpow2, Hpow2, 3), dtype =np.uint8))
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER , GL_NEAREST)
	
	def set_mean_texture(self, mean):
		glUseProgram(self.SP['PID'])
		glActiveTexture(GL_TEXTURE1)
		glBindTexture(GL_TEXTURE_2D, self.SP['mean_texture_ID']) 
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.W, self.H, GL_RGB, GL_UNSIGNED_BYTE, mean)
		
		sampler_ID = glGetUniformLocation(self.SP['PID'], 'mean_texture_sampler')
		glUniform1i(sampler_ID, 1) #set mean_texture_sampler uniform to 0 (that we just activated)
		glUseProgram(0)

	def switch_shader_type(self):
		self.SP['uni_mode']['value'] = (self.SP['uni_mode']['value'] + 1) % 4
		glUseProgram(self.SP['PID'])
		glUniform1ui(self.SP['uni_mode']['ID'], self.SP['uni_mode']['value'])
		glUseProgram(0)

	# 1/ Clears the buffers (color and depth) and DEACTIVATES depth tests (need to reactivate later!!)
	# 2/ Binds the texture and blit the given image to it
	# 3/ Sets a orthogrphic projection over the area W/H
	# 4/ draw a quad in the camera field (0,0,W,H) with x and y as the maximum uv texture coordinates
	# => sets the background with the given image
	def clear(self, image):
		#/1 clear frame buffer and Z buffer
		
		glDisable(GL_DEPTH_TEST)

		#/2 send the image texture to the GPU
		glActiveTexture(GL_TEXTURE0)
		glEnable(GL_TEXTURE_2D)
		glBindTexture(GL_TEXTURE_2D, self.SP['background_texture_ID']) 
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.W, self.H, GL_RGB, GL_UNSIGNED_BYTE, image)
		
		#/3 Load a matrix for the rendering
		glMatrixMode (GL_PROJECTION)
		glLoadIdentity()
		gluOrtho2D(0, self.W, self.H, 0)
		glMatrixMode(GL_MODELVIEW)
		#glLoadIdentity()
		#/4 render a quad with the previously binded texture
		glBegin(GL_QUADS)
		glTexCoord2f(0, 0)
		glVertex2f(0, 0)
		glTexCoord2f(self.x, 0)
		glVertex2f(self.W, 0)
		glTexCoord2f(self.x, self.y)
		glVertex2f(self.W, self.H)
		glTexCoord2f(0, self.y)
		glVertex2f(0, self.H)
		glEnd()
		glClear(GL_DEPTH_BUFFER_BIT)

	def set_P_from_camera(self, K):
		"""  Set view from a camera calibration matrix. """

		fx = K[0,0]
		fy = K[1,1]
		fovy = 2*np.arctan(0.5*self.H/fy)
		aspect = (self.W*fy)/(self.H*fx)
		# define the near and far clipping planes
		near = 0.1
		far = 100.0
		
		# set perspective
		self.P = glm.perspective(fovy, aspect, near, far)
		self.update_P = True

		glViewport(0,0,self.W,self.H)

	def set_M(self, scale, size_scale, H_marker, W_marker):
		H_tmp = H_marker * size_scale
		W_tmp = W_marker * size_scale
		sM = glm.scale(glm.mat4(), glm.vec3(scale, -scale, scale)) 
		tM = glm.translate(glm.mat4(), glm.vec3(W_tmp/2, 0, -H_tmp/2))
		self.M = tM * sM
		self.update_M = True

	def nparray_to_glm_mat(self, array):
		res = glm.mat4()
		for i in range(4):
			for j in range(4):
				tmp = array[i, j]
				res[i][j] = tmp
		return res
		
		

	def set_V_from_camera(self, cTw):
		"""  Set the model view matrix from camera pose. """

		cv_to_gl = np.zeros((4, 4))
		
		cv_to_gl[0, 0] = 1
		cv_to_gl[3, 3] = 1
		#invert the y and z axis
		cv_to_gl[1, 1] = -1
		cv_to_gl[2, 2] = -1

		viewMatrix = np.dot(cv_to_gl, cTw)

		viewMatrix = viewMatrix.T

		self.V = self.nparray_to_glm_mat(viewMatrix) * glm.rotate(glm.mat4(), math.pi/2, glm.vec3(1, 0, 0))
		#V = glm.translate(glm.mat4(), glm.vec3(0, 0, -10*2))# * glm.rotate(glm.mat4(), math.pi, glm.vec3(1, 0, 0))

		self.update_V = True
		

	def render_model(self, model, t):
		glUseProgram(self.SP['PID'])

		if self.SP['M_ID'] != -1:
			glUniformMatrix4fv(self.SP['M_ID'], 1, GL_FALSE, glm.value_ptr(self.M))
			self.update_M = False
		if self.SP['V_ID'] != -1:
			glUniformMatrix4fv(self.SP['V_ID'], 1, GL_FALSE, glm.value_ptr(self.V))
			self.update_V = False
		if self.SP['P_ID'] != -1:
			glUniformMatrix4fv(self.SP['P_ID'], 1, GL_FALSE, glm.value_ptr(self.P))
			self.update_P = False

		glEnable(GL_DEPTH_TEST)
		# glDepthFunc(GL_LESS)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		# glEnable(GL_CULL_FACE)
		# glCullFace(GL_FRONT)

		model.render(self.SP)
		#glDisable(GL_CULL_FACE)

		glDisable(GL_BLEND)
		glDisable(GL_DEPTH_TEST)

		

	def reset_program(self):
		glUseProgram(0)