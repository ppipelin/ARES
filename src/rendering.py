from OpenGL.GL import *
from OpenGL.GLU import *
import math
import numpy as np

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