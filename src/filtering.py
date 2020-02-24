import cv2
import math
import numpy as np

class KalmanFilter:
    def __init__(self, dt):
        self.KF = cv2.KalmanFilter(18, 6, 0, cv2.CV_64F)

        self.KF.processNoiseCov = 1e-5 * np.identity(18, np.float64)
        self.KF.measurementNoiseCov = 1e-4 * np.identity(6, np.float64)
        self.KF.errorCovPost = np.identity(18, np.float64)

        print(self.KF.processNoiseCov)
        print(self.KF.measurementNoiseCov)
        print(self.KF.errorCovPost)

        measurementMatrix = self.KF.measurementMatrix.copy()
        transitionMatrix = self.KF.transitionMatrix.copy()

        transitionMatrix[0, 3] = dt
        transitionMatrix[1, 4] = dt
        transitionMatrix[2, 5] = dt
        transitionMatrix[3, 6] = dt
        transitionMatrix[4, 7] = dt
        transitionMatrix[5, 8] = dt
        transitionMatrix[0, 6] = 0.5 * dt ** 2
        transitionMatrix[1, 7] = 0.5 * dt ** 2
        transitionMatrix[2, 8] = 0.5 * dt ** 2

        transitionMatrix[9, 12] = dt
        transitionMatrix[10, 13] = dt
        transitionMatrix[11, 14] = dt
        transitionMatrix[12, 15] = dt
        transitionMatrix[13, 16] = dt
        transitionMatrix[14, 17] = dt
        transitionMatrix[9, 15] = 0.5 * dt ** 2
        transitionMatrix[10, 16] = 0.5 * dt ** 2
        transitionMatrix[11, 17] = 0.5 * dt ** 2

        measurementMatrix[0, 0] = 1 # x
        measurementMatrix[1, 1] = 1 # y
        measurementMatrix[2, 2] = 1 # z
        measurementMatrix[3, 9] = 1 # roll
        measurementMatrix[4, 10] = 1 # pitch
        measurementMatrix[5, 11] = 1 # yaw

        self.KF.transitionMatrix = transitionMatrix
        self.KF.measurementMatrix = measurementMatrix

        self.measurements = np.zeros(6)

    def fill(self, rmat, tvec):
        measured_eulers = rotationMatrixToEulerAngles(rmat)
        self.measurements[0] = tvec[0]
        self.measurements[1] = tvec[1]
        self.measurements[2] = tvec[2]
        self.measurements[3] = measured_eulers[0]
        self.measurements[4] = measured_eulers[1]
        self.measurements[5] = measured_eulers[2]
    
    def predict(self):
        prediction = self.KF.predict()
        estimated = self.KF.correct(self.measurements)
        tvec = np.zeros(3)
        tvec[0] = estimated[0]
        tvec[1] = estimated[1]
        tvec[2] = estimated[2]
        euler = np.zeros(3)
        euler[0] = estimated[9]
        euler[1] = estimated[10]
        euler[2] = estimated[11]
        rmat = eulerAnglesToRotationMatrix(euler)
        cTw = np.column_stack((rmat[:,0], rmat[:,1], rmat[:,2], tvec))
        cTw = np.vstack([cTw, [0,0,0,1]])
        return cTw

def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])                 
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])          
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])                          
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def rotationMatrixToEulerAngles(R) :     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])