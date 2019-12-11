import numpy as np 
import matplotlib.image as mpimg 
import glob 
import cv2 

image_files_path = glob.glob("calibration/calibration*.jpg")

def calib():
    """
    calculate the matrix and the distorition coefficient for calib
    Note: sometime, this function is not necessary
    """

    img_points = []  # 2D points in image plane
    obj_points = []  # 3D points in real world space
    

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for curr_file in image_files_path:
        img = mpimg.imread(curr_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret == True:
            img_points.append(corners)
            obj_points.append(objp)
        else:
            continue
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    return mtx, dist


def undistort(img, mtx, dist):
    """
    undistort image
    """
    return cv2.undistort(img, mtx, dist, None, mtx)