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
    for surface in surfaces:
        x = 2
        for vertex in surface:
            x = (x + 1) % len(colors)
            glColor3fv(colors[x])
            glVertex3fv(verticies[vertex])
    glEnd()

    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()