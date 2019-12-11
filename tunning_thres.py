import cv2
from calibration import calib, undistort
from compute_threshold import combine_gradients, get_hls, combine_hls_grandient
from utils import *
import numpy as np
import argparse

class LaneDetector:
    def __init__(self, image, thesh_sobelx_X=35, thesh_sobelx_Y=100, thesh_sobely_X=30, thesh_sobely_Y=255, thesh_mag_X=30, thesh_mag_Y=255, thesh_dir_X=0.7, thesh_dir_Y=1.3, th_h_X=10, th_h_Y=100, th_l_X=0, th_l_Y=60, th_s_X=85, th_s_Y=255):

        self.left_line = Line()
        self.right_line = Line()

        self.mtx, self.dist = calib()

        self.image = image

        # HLS thresh 
        self.thesh_h_X = th_h_X
        self.thesh_h_Y = th_h_Y
        self.thesh_l_X = th_l_X
        self.thesh_l_Y = th_l_Y
        self.thesh_s_X = th_s_X
        self.thesh_s_Y = th_s_Y

        # Sobel and mag, dir thresh
        self.thesh_sobelx_X = thesh_sobelx_X
        self.thesh_sobelx_Y = thesh_sobelx_Y
        self.thesh_sobely_X = thesh_sobely_X
        self.thesh_sobely_Y = thesh_sobely_Y
        self.thesh_mag_X = thesh_mag_X
        self.thesh_mag_Y = thesh_mag_Y
        self.thesh_dir_X = thesh_dir_X
        self.thesh_dir_Y = thesh_dir_Y


        def thesh_sobelx_X(self):
            return self.thesh_sobelx_X

        def thesh_sobelx_Y(self):
            return self.thesh_sobelx_Y

        def thesh_sobely_X(self):
            return self.thesh_sobely_X 

        def thesh_sobely_Y(self):
            return self.thesh_sobely_Y

        def thesh_mag_X(self):
            return self.thesh_mag_X


        def onchangethesh_h_X(pos):
            self.thesh_h_X = pos
            self.renders()

        def onchangethesh_h_Y(pos):
            self.thesh_h_Y = pos
            self.renders()

        def onchangethesh_l_X(pos):
            self.thesh_l_X = pos
            self.renders()

        def onchangethesh_l_Y(pos):
            self.thesh_l_Y = pos
            self.renders()

        def onchangethesh_s_X(pos):
            self.thesh_s_X = pos
            self.renders()

        def onchangethesh_s_Y(pos):
            self.thesh_s_Y = pos
            self.renders()
        
        
        def onchangethesh_sobelx_X(pos):
            self.thesh_sobelx_X = pos
            self.renders()

        def onchangethesh_sobelx_Y(pos):
            self.thesh_sobelx_Y = pos
            self.renders()

        def onchangethesh_sobely_X(pos):
            self.thesh_sobely_X = pos
            self.renders()

        def onchangethesh_sobely_Y(pos):
            self.thesh_sobely_Y = pos
            self.renders()

        def onchangethesh_mag_X(pos):
            self.thesh_mag_X = pos
            self.renders()

        def onchangethesh_mag_Y(pos):
            self.thesh_mag_Y = pos
            self.renders()

        def onchangethesh_dir_X(pos):
            self.thesh_mag_X = pos
            self.renders()

        def onchangethesh_dir_Y(pos):
            self.thesh_mag_Y = pos
            self.renders()

        

        cv2.namedWindow('Find_threshold_window')

        cv2.createTrackbar('thesh_h_X',      'Find_threshold_window', self.thesh_h_X,          255, onchangethesh_h_X)
        cv2.createTrackbar('thesh_h_Y',      'Find_threshold_window', self.thesh_h_Y,          255, onchangethesh_h_Y)
        cv2.createTrackbar('thesh_l_X',      'Find_threshold_window', self.thesh_l_X,          255, onchangethesh_l_X)
        cv2.createTrackbar('thesh_l_Y',      'Find_threshold_window', self.thesh_l_Y,          255, onchangethesh_l_Y)
        cv2.createTrackbar('thesh_s_X',      'Find_threshold_window', self.thesh_s_X,          255, onchangethesh_s_X)
        cv2.createTrackbar('thesh_s_Y',      'Find_threshold_window', self.thesh_s_Y,          255, onchangethesh_s_Y)

        cv2.createTrackbar('thesh_sobelx_X', 'Find_threshold_window', self.thesh_sobelx_X, 255, onchangethesh_sobelx_X)
        cv2.createTrackbar('thesh_sobelx_Y', 'Find_threshold_window', self.thesh_sobelx_X, 255, onchangethesh_sobelx_Y)
        cv2.createTrackbar('thesh_sobely_X', 'Find_threshold_window', self.thesh_sobely_X, 255, onchangethesh_sobely_X)
        cv2.createTrackbar('thesh_sobely_Y', 'Find_threshold_window', self.thesh_sobely_Y, 255, onchangethesh_sobely_Y)
        cv2.createTrackbar('thesh_mag_Y',    'Find_threshold_window', self.thesh_mag_Y,     255, onchangethesh_mag_Y)
        

        self.renders()

        cv2.waitKey(0)

        cv2.destroyWindow('Find_threshold_window')


    def renders(self):
        # Correcting for Distortion
        undist_img = undistort(self.image, self.mtx, self.dist)
        # resize video
        undist_img = cv2.resize(undist_img, None, fx=1/2, fy=1/2 , interpolation=cv2.INTER_AREA)
        rows, cols = undist_img.shape[:2]

        combined_gradient = combine_gradients(undist_img, (self.thesh_sobelx_X, self.thesh_sobelx_Y), (self.thesh_sobely_X,self.thesh_sobely_Y), (self.thesh_mag_X, self.thesh_mag_Y), (self.thesh_dir_X, self.thesh_dir_Y))

        combined_hls = get_hls(undist_img, (self.thesh_h_X,self.thesh_h_Y) , (self.thesh_l_X, self.thesh_l_Y), (self.thesh_s_X, self.thesh_s_Y))

        combined_result = combine_hls_grandient(combined_gradient, combined_hls)

        c_rows, c_cols = combined_result.shape[:2]
        s_LTop2, s_RTop2 = [299, 40], [374, 40]
        s_LBot2, s_RBot2 = [218, 92], [472, 92]

        src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
        dst = np.float32([(60, 540), (60, 0), (360, 0), (360, 540)])

        warp_img, M, Minv = get_perspective_transform(combined_result, src, dst, (420, 540))
        cv2.imwrite('combine.png', combined_gradient)
        cv2.imwrite('wrap.png', warp_img)

        searching_img = get_lane_lines_img(warp_img, self.left_line, self.right_line)

        w_comb_result, w_color_result = draw_lane_line(searching_img, self.left_line, self.right_line)

        color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
        lane_color = np.zeros_like(undist_img)
        lane_color[180:, :] = color_result
        

        result = cv2.addWeighted(undist_img, 1, lane_color, 0.3, 0)
        cv2.imwrite('results.png', result)

        cv2.imshow('LaneLines', result)
        cv2.imshow('bird_view', w_color_result)
        cv2.imshow('search_img', searching_img)
        cv2.imshow('gradient', warp_img)
        print(combined_hls.shape) # For debuging
        cv2.imshow('hls', combined_hls)
        cv2.imshow('combine', combined_result)


def main():
    
    parser = argparse.ArgumentParser(description='Visualize Lane Finding Params.')
    parser.add_argument('filename')

    args = parser.parse_args()

    img = cv2.imread(args.filename)

    lane_detect = LaneDetector(img)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
