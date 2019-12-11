import cv2
import numpy as np
from utils import *
from compute_threshold import *

s_ltop_x, s_ltop_y, s_rtop_x, s_rtop_y = 299, 40, 374, 40
s_lbot_x, s_lbot_y, s_rbot_x, s_rbot_y = 218, 92, 472, 92

def render():
    global s_ltop_x, s_ltop_y, s_rtop_x, s_rtop_y, s_lbot_x, s_lbot_y, s_rbot_x, s_rbot_y
    img_ = img.copy()
    c_rows, c_cols = img.shape[:2]
    s_LTop2, s_RTop2 = [s_ltop_x, s_ltop_y], [s_rtop_x, s_rtop_y]
    s_LBot2, s_RBot2 = [s_lbot_x, s_lbot_y], [s_rbot_x, s_rbot_y]
    cv2.circle(img_, (s_RTop2[0], s_RTop2[1]), 5, (0,255,0), -1)
    cv2.circle(img_, (s_LTop2[0], s_LTop2[1]), 5, (0,255,255), -1)
    cv2.circle(img_, (s_RBot2[0], s_RBot2[1]), 5, (0,0,255), -1)
    cv2.circle(img_, (s_LBot2[0], s_LBot2[1]), 5, (255, 0,0), -1)
    print('in this\n')
    print(s_ltop_x)
    
    src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
    dst = np.float32([(60, 540), (60, 0), (360, 0), (360, 540)])

    warp_img, M, Minv = get_perspective_transform(img_, src, dst, (420, 540))

    cv2.imshow('img', img_)
    cv2.imshow('wrapimg', warp_img)


        
def onchange_s_ltop_x(pos):
    global s_ltop_x
    s_ltop_x = pos
    render()


def onchange_s_ltop_y(pos):
    global s_ltop_y
    s_ltop_y = pos
    render()


def onchange_s_rtop_x(pos):
    global s_rtop_x
    s_rtop_x = pos
    render()


def onchange_s_rtop_y(pos):
    global s_rtop_y
    s_rtop_y = pos
    render()


def onchange_s_lbot_x(pos):
    global s_lbot_x
    s_lbot_x = pos
    render()


def onchange_s_lbot_y(pos):
    global s_lbot_y
    s_lbot_y = pos
    render()


def onchange_s_rbot_x(pos):
    global s_rbot_x
    s_rbot_x = pos
    render()


def onchange_s_rbot_y(pos):
    global s_rbot_y
    s_rbot_y = pos
    render()

# Create a black image, a window
img = cv2.imread('check_3.png')
print(img.shape)
# Resize image to half of origin
img = cv2.resize(img, None, fx=1/2, fy=1/2 , interpolation=cv2.INTER_AREA)
# rows, cols = img.shape[:2]
# th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)

# combined_gradient = combine_gradients(img, th_sobelx, th_sobely, th_mag, th_dir)
# img = combined_gradient
# print(img.shape)
cv2.namedWindow('wrapimg')
cv2.namedWindow('img')



cv2.createTrackbar('s_ltop_x', 'wrapimg', s_ltop_x, 1080, onchange_s_ltop_x)
cv2.createTrackbar('s_ltop_y', 'wrapimg', s_ltop_y, 1080, onchange_s_ltop_y)
cv2.createTrackbar('s_rtop_x', 'wrapimg', s_rtop_x, 1080, onchange_s_rtop_x)
cv2.createTrackbar('s_rtop_y', 'wrapimg', s_rtop_y, 1080, onchange_s_rtop_y)
cv2.createTrackbar('s_lbot_x', 'wrapimg', s_lbot_x, 1080, onchange_s_lbot_x)
cv2.createTrackbar('s_lbot_y', 'wrapimg', s_lbot_y, 1080, onchange_s_lbot_y)
cv2.createTrackbar('s_rbot_x', 'wrapimg', s_rbot_x, 1080, onchange_s_rbot_x)
cv2.createTrackbar('s_rbot_y', 'wrapimg', s_rbot_y, 1080, onchange_s_rbot_y)

render()
cv2.waitKey(0)
cv2.destroyAllWindows('wrapimg')