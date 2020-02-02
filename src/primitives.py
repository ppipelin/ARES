from OpenGL.GL import *
from OpenGL.GLU import *


verticies = [
    (0.1, -0.1, -0.1),
    (0.1, 0.1, -0.1),
    (-0.1, 0.1, -0.1),
    (-0.1, -0.1, -0.1),
    (0.1, -0.1, 0.1),
    (0.1, 0.1, 0.1),
    (-0.1, -0.1, 0.1),
    (-0.1, 0.1, 0.1),
]

colors = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
]

edges = [
    (0, 1, 0),
    (0, 3, 1),
    (0, 4, 2),
    (2, 1, 0),
    (2, 3, 1),
    (2, 7, 2),
    (6, 3, 0),
    (6, 4, 1),
    (6, 7, 2),
    (5, 1, 0),
    (5, 4, 1),
    (5, 7, 2),
]

surfaces = [
    (0,1,2,3),
    (3,2,7,6),
    (6,7,5,4),
    (4,5,1,0),
    (1,5,7,2),
    (4,0,3,6)
]


def Cube():
    glBegin(GL_QUADS)
    x = 0
    for surface in surfaces:
        glColor3fv(colors[x])
        x = (x + 1) % len(colors)
        for vertex in surface:
            
            glVertex3fv(verticies[vertex])
    glEnd()

    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()