import numpy as np
import cv2
import glob

dim_board = (7,7)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((dim_board[0]*dim_board[1],3), np.float32)
objp[:,:2] = np.mgrid[0:dim_board[0],0:dim_board[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# fname = 'data/mire_chessboard_real.jpg'
fname = 'data/chess_big.png'
img = cv2.imread(fname)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imwrite("test.jpg",gray)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, dim_board,None)
# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, dim_board, corners2,ret)
    cv2.imshow('img',img)
    # cv2.waitKey(5000)
    # cv2.waitKey(1000)
# print(gray.shape)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print("Calib matrix:\n",  mtx)
print(dist)
cv2.destroyAllWindows()
