import numpy as np 
import cv2 
from PIL import Image
import matplotlib.image as mpimg

def get_perspective_transform(img, src, dst, size):

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

    return warp_img, M, Minv


class Line:
    def __init__(self):
        self.detected = False
        self.window_margin = 30
        self.previous_x = []
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None
        self.start_x = None
        self.end_x = None
        self.all_x = None
        self.all_y = None 

def smooth_lines(lines, prev_n_line=3):
    lines = np.squeeze(lines)
    avg_line = np.zeros((540))

    for i, line in enumerate(reversed(lines)):
        if i == prev_n_line:
            break
        avg_line += line

    avg_line /= prev_n_line

    return avg_line


def measure_curvature_lane(left_lane, right_lane):
    """
    This function measures curvature of the left lane and right lane
    """

    ploty = left_lane.all_y
    left_x, right_x = left_lane.all_x, right_lane.all_x

    left_x = left_x[::-1]     # reverse to top to bottom in y
    right_x = right_x[::-1]   # reverse to top to bottom in y

    y_eval = np.max(ploty)

    lane_width = abs(left_lane.start_x - right_lane.start_x)
    ym_per_pix = 30/720
    xm_per_pix = 3.7*(720/1200) / lane_width

    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_x * xm_per_pix, 2)

    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])

    right_lane.radius_of_curvature = right_curverad
    left_lane.radius_of_curvature = left_curverad


def line_search_begin(bin_img, right_lane, left_lane):
    """
    Use in case need for searching without information about previous lane
    """

    hist = np.sum(bin_img[int(bin_img.shape[0] / 2):, :], axis=0)
    out_img = np.dstack((bin_img, bin_img, bin_img)) * 255

    mid_point = np.int(hist.shape[0] / 2)

    left_X_base = np.argmax(hist[:mid_point])
    right_X_base = np.argmax(hist[mid_point:]) + mid_point

    num_windows = 27

    window_height = np.int(bin_img.shape[0] / num_windows)
    nonzero = bin_img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    current_left_X = left_X_base
    current_right_X = right_X_base

    min_num_pixel = 50

    win_left_lane = []
    win_right_lane = []
    
    margin = left_lane.window_margin

    for window in range(num_windows):
        win_y_low = bin_img.shape[0] - (window + 1) * window_height
        win_y_high = bin_img.shape[0] - window * window_height
        win_left_x_min = current_left_X - margin
        win_left_x_max = current_left_X + margin
        win_right_x_min = current_right_X - margin
        win_right_x_max = current_right_X + margin

        cv2.rectangle(out_img, (win_left_x_min, win_y_low), (win_left_x_max, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_right_x_min, win_y_low), (win_right_x_max, win_y_high), (0, 255, 0), 2)


        left_window_inds = ((nonzero_y <= win_y_high) & (nonzero_y >= win_y_low) &  (nonzero_x >= win_left_x_min) & (
            nonzero_x <= win_left_x_max)).nonzero()[0]
        right_window_inds = ((nonzero_y <= win_y_high) & (nonzero_x >= win_right_x_min) & (nonzero_y >= win_y_low) & (
            nonzero_x <= win_right_x_max)).nonzero()[0]
  
        win_left_lane.append(left_window_inds)
        win_right_lane.append(right_window_inds)


        if len(left_window_inds) > min_num_pixel:
            current_left_X = np.int(np.mean(nonzero_x[left_window_inds]))
        if len(right_window_inds) > min_num_pixel:
            current_right_X = np.int(np.mean(nonzero_x[right_window_inds]))

   
    win_left_lane = np.concatenate(win_left_lane)
    win_right_lane = np.concatenate(win_right_lane)


    left_x= nonzero_x[win_left_lane]
    lefty =  nonzero_y[win_left_lane]
    right_x = nonzero_x[win_right_lane]
    righty = nonzero_y[win_right_lane]

    out_img[lefty, left_x] = [255, 0, 0]
    out_img[righty, right_x] = [0, 0, 255]

    # Fit a polynomial to each lane
    left_fit = np.polyfit(lefty, left_x, 2)
    right_fit = np.polyfit(righty, right_x, 2)

    left_lane.current_fit = left_fit
    right_lane.current_fit = right_fit

    ploty = np.linspace(0, bin_img.shape[0] - 1, bin_img.shape[0])

    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_lane.previous_x.append(left_plotx)
    right_lane.previous_x.append(right_plotx)

    if len(left_lane.previous_x) > 10:
        left_avg_line = smooth_lines(left_lane.previous_x, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        left_lane.current_fit = left_avg_fit
        left_lane.all_x, left_lane.all_y = left_fit_plotx, ploty
    else:
        left_lane.current_fit = left_fit
        left_lane.all_x, left_lane.all_y = left_plotx, ploty

    if len(right_lane.previous_x) > 10:
        right_avg_line = smooth_lines(right_lane.previous_x, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        right_lane.current_fit = right_avg_fit
        right_lane.all_x, right_lane.all_y = right_fit_plotx, ploty
    else:
        right_lane.current_fit = right_fit
        right_lane.all_x, right_lane.all_y = right_plotx, ploty

    left_lane.start_x, right_lane.start_x = left_lane.all_x[len(left_lane.all_x)-1], right_lane.all_x[len(right_lane.all_x)-1]
    left_lane.end_x, right_lane.end_x = left_lane.all_x[0], right_lane.all_x[0]

    left_lane.detected, right_lane.detected = True, True
    
    measure_curvature_lane(left_lane, right_lane)
    
    return out_img



def line_search(bin_img, left_line, right_line):
    """
    Use in case that known the information about previous lane
    """

    out_img = np.dstack((bin_img, bin_img, bin_img)) * 255
    
    nonzero = bin_img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    window_margin = left_line.window_margin

    left_line_fit = left_line.current_fit
    right_line_fit = right_line.current_fit
    left_x_min = left_line_fit[0] * nonzero_y ** 2 + left_line_fit[1] * nonzero_y + left_line_fit[2] - window_margin
    left_x_max = left_line_fit[0] * nonzero_y ** 2 + left_line_fit[1] * nonzero_y + left_line_fit[2] + window_margin
    right_x_min = right_line_fit[0] * nonzero_y ** 2 + right_line_fit[1] * nonzero_y + right_line_fit[2] - window_margin
    right_x_max = right_line_fit[0] * nonzero_y ** 2 + right_line_fit[1] * nonzero_y + right_line_fit[2] + window_margin


    left_inds = ((nonzero_x >= left_x_min) & (nonzero_x <= left_x_max)).nonzero()[0]
    right_inds = ((nonzero_x >= right_x_min) & (nonzero_x <= right_x_max)).nonzero()[0]


    left_x, lefty = nonzero_x[left_inds], nonzero_y[left_inds]
    right_x, righty = nonzero_x[right_inds], nonzero_y[right_inds]

    out_img[lefty, left_x] = [255, 0, 0]
    out_img[righty, right_x] = [0, 0, 255]

    left_fit = np.polyfit(lefty, left_x, 2)
    right_fit = np.polyfit(righty, right_x, 2)

    ploty = np.linspace(0, bin_img.shape[0] - 1, bin_img.shape[0])

    # Fit line
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_x_avg = np.average(left_plotx)
    right_x_avg = np.average(right_plotx)

    left_line.previous_x.append(left_plotx)
    right_line.previous_x.append(right_plotx)

    if len(left_line.previous_x) > 10:  # take previously detected lane lines 
        left_avg_line = smooth_lines(left_line.previous_x, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        left_line.current_fit = left_avg_fit
        left_line.all_x, left_line.all_y = left_fit_plotx, ploty
    else:
        left_line.current_fit = left_fit
        left_line.all_x, left_line.all_y = left_plotx, ploty

    if len(right_line.previous_x) > 10: 
        right_avg_line = smooth_lines(right_line.previous_x, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.all_x, right_line.all_y = right_fit_plotx, ploty
    else:
        right_line.current_fit = right_fit
        right_line.all_x, right_line.all_y = right_plotx, ploty

    # Calculate stddev of the distance between X positions of pixels of left and right lines for checking restart
    stddev = np.std(right_line.all_x - left_line.all_x)

    if (stddev > 100):
        left_line.detected = False

    left_line.start_x, right_line.start_x = left_line.all_x[len(left_line.all_x) - 1], right_line.all_x[len(right_line.all_x) - 1]
    left_line.end_x, right_line.end_x = left_line.all_x[0], right_line.all_x[0]

    measure_curvature_lane(left_line, right_line)
    
    return out_img


def get_image_lane_lines(binary_img, left_line, right_line):
    """
    Get the lane line. Select option to choose which searching is used
    """
    
    if left_line.detected == False:
        return line_search_begin(binary_img, left_line, right_line)
    else:
        return line_search(binary_img, left_line, right_line)


def draw_lane_line(img, left_line, right_line, left_color=(255, 0, 0), right_color=(0, 0, 255),  road_color=(0, 255, 255)):
    """ 
    Draw the lane
    """

    # Create an empty image to draw on
    window_img = np.zeros_like(img)

    window_margin = left_line.window_margin
    left_plotx, right_plotx = left_line.all_x, right_line.all_x
    ploty = left_line.all_y

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_plotx - window_margin/5, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_plotx + window_margin/5, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_plotx - window_margin/5, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_plotx + window_margin/5, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), left_color)
    cv2.fillPoly(window_img, np.int_([right_line_pts]), right_color)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_plotx+window_margin/5, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx-window_margin/5, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), road_color)
    result = cv2.addWeighted(img, 1, window_img, 0.3, 0)

    return result, window_img
