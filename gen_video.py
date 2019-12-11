import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibration import calib, undistort
from compute_threshold import combine_gradients, get_hls, combine_hls_grandient
from utils import *

input_name = 'video1.avi'

save_img = True

left_line = Line()
right_line = Line()

# params can tunning
th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
th_h, th_l, th_s = (255, 255), (0, 141), (0, 255)

# camera matrix & distortion coefficient
mtx, dist = calib()


def pipeline(frame):
    # cv2.imwrite('./out_img/origin.png', frame)
    undist_img = undistort(frame, mtx, dist)

    undist_img = cv2.resize(undist_img, None, fx=1/2, fy=1/2 , interpolation=cv2.INTER_AREA)
    # cv2.imwrite('./out_img/0_undist_img.png', undist_img)
    rows, cols = undist_img.shape[:2]

    combined_gradient = combine_gradients(undist_img, th_sobelx, th_sobely, th_mag, th_dir)
    # cv2.imwrite('./out_img/1_combine_gradient.png', combined_gradient)

    combined_hls = get_hls(undist_img, th_h, th_l, th_s)
    # cv2.imwrite('./out_img/2_combine_hls.png', combined_hls)

    combined_result = combine_hls_grandient(combined_gradient, combined_hls)
    # cv2.imwrite('./out_img/3_combine_result.png', combined_result)

    c_rows, c_cols = combined_gradient.shape[:2]
    s_LTop2, s_RTop2 = [299, 40], [374, 40]
    s_LBot2, s_RBot2 = [218, 92], [472, 92]

    src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
    dst = np.float32([(60, 540), (60, 0), (360, 0), (360, 540)])
    

    warp_img, M, Minv = get_perspective_transform(combined_result, src, dst, (420, 540))
    # cv2.imwrite('./out_img/bird_eye_view.png', warp_img)

    searching_img = get_image_lane_lines(warp_img, left_line, right_line)
    # cv2.imwrite('./out_img/searching.png', searching_img)

    w_comb_result, w_color_result = draw_lane_line(searching_img, left_line, right_line)
    # cv2.imwrite('./out_img/draw_lane.png', w_color_result)

    color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
    # cv2.imwrite('./out_img/wrapped.png', color_result)

    lane_color = np.zeros_like(undist_img)
    lane_color[180:, :] = color_result

    result = cv2.addWeighted(undist_img, 1, lane_color, 0.3, 0)
    # cv2.imwrite('./out_img/result.png', result)

        
    return result


if __name__ == '__main__':

    cap = cv2.VideoCapture(input_name)
    if cap.isOpened() == False:
        print('Error')
        
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            out = pipeline(frame)
            cv2.imshow("out", out)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
        else:
            break
        # break
        
    cap.release()
    cv2.destroyAllWindows()


