import numpy as np
from OpenGL.GL import *
from PIL import Image
import glm
import math


class Model:
    def __init__(self):
        self.vert_coords = []
        self.text_coords = []
        self.norm_coords = []

        self.vertex_index = []
        self.texture_index = []
        self.normal_index = []
        self.VBO = 0
        self.model = []
        self.texture = 0
        self.texture_offset = 0
        self.normal_offset = 0

        #self.M = glm.scale(glm.mat4(), glm.vec3(1, 1, 1))

    def render(self, SP):
        
        #glUniformMatrix4fv(SP['uni_mat_M_ID'], 1, False, glm.value_ptr(self.M))

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)

        # positions
        in_position_ID = SP['in_position_ID']
        if in_position_ID != -1:
            glVertexAttribPointer(in_position_ID, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
            glEnableVertexAttribArray(in_position_ID)
        # normals
        in_normal_ID = SP['in_normal_ID']
        if in_normal_ID != -1:
            glVertexAttribPointer(in_normal_ID, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(self.normal_offset))
            glEnableVertexAttribArray(in_normal_ID)

        # texture uv 
        in_uv_ID = SP['in_uv_ID']
        if in_uv_ID != -1:
            glVertexAttribPointer(in_uv_ID, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(self.uv_offset))
            glEnableVertexAttribArray(in_uv_ID)

        if False and self.textured:
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glEnable(GL_TEXTURE_2D)
        glDrawArrays(GL_TRIANGLES, 0, len(self.vertex_index))


    def load_from_obj(self, obj_file, tex_file=''):
        scale = 0.5
        scaleM =  glm.scale(glm.mat4(), glm.vec3(scale, -scale, scale))
        rotateM = glm.rotate(glm.mat4(), -math.pi/2, glm.vec3(1, 0, 0))
        #self.M = scaleM 

        self.textured = not (tex_file == '')
        for line in open(obj_file, 'r'):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue

            if values[0] == 'v':
                self.vert_coords.append(values[1:4])
            if values[0] == 'vt':
                self.text_coords.append(values[1:3])
            if values[0] == 'vn':
                self.norm_coords.append(values[1:4])

            if values[0] == 'f':
                face_i = []
                text_i = []
                norm_i = []
                for v in values[1:4]:
                    w = v.split('/')
                    face_i.append(int(w[0])-1)
                    text_i.append(int(w[1])-1)
                    norm_i.append(int(w[2])-1)
                self.vertex_index.append(face_i)
                self.texture_index.append(text_i)
                self.normal_index.append(norm_i)

        self.vertex_index = [y for x in self.vertex_index for y in x]
        self.texture_index = [y for x in self.texture_index for y in x]
        self.normal_index = [y for x in self.normal_index for y in x]


        for i in self.vertex_index:
            self.model.extend(self.vert_coords[i])

        for i in self.normal_index:
            self.model.extend(self.norm_coords[i])

        for i in self.texture_index:
            self.model.extend(self.text_coords[i])

        self.model = np.array(self.model, dtype='float32')

        # 3 because vec3 and 4 because float
        self.normal_offset = len(self.vertex_index) * 3 * self.model.itemsize
        self.uv_offset = self.normal_offset + len(self.normal_index) * 3 * self.model.itemsize

        # VBO

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.model.itemsize * len(self.model), self.model, GL_STATIC_DRAW)
        
        if False and self.textured:
            # Texture
            image = Image.open(tex_file)   
            flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = np.array(list(flipped_image.getdata()), np.uint8)
            self.texture_offset = len(self.vertex_index)*12

            self.texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
            # Set the texture wrapping parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            # Set texture filtering parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            

        # textID = glGenTextures(1)
        # glBindTexture(GL_TEXTURE_2D, textID) 
        # Hpow2 = int(math.pow(2, math.ceil(math.log(H)/math.log(2))))
        # Wpow2 = int(math.pow(2, math.ceil(math.log(W)/math.log(2))))
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, Wpow2, Hpow2, 0,  GL_RGB, GL_UNSIGNED_BYTE, np.zeros((Wpow2, Hpow2, 3), dtype =np.uint8))
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER , GL_NEAREST)
        
        