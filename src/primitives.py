from OpenGL.GL import *
from OpenGL.GLU import *


verticies = [
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1),
]

colors = [
    (255.0, 0.0, 0.0),
    (0.0, 255.0, 0.0),
    (0.0, 0.0, 255.0),
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


def Cube():
    glBegin(GL_LINES)
    for edge in edges:
        glColor(colors[edge[2]][0], colors[edge[2]][1], colors[edge[2]][2], 255.0)
        for vertex in (edge[0], edge[1]):
            glVertex3fv(verticies[vertex])
    glEnd()