import numpy as np 
import cv2

def abs_sobel_threshold(img, ora='x', threshold=(20, 100)):
    """
    Apply sobel function in x or y and then takes abs value
    and apply a threshold
    """

    if ora == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    else:
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    
    abs_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))   #Scale value of sobel

    # create and apply binary threshold
    binary = np.zeros_like(abs_sobel)
    binary[(abs_sobel <= threshold[1]) & (abs_sobel >= threshold[0])] = 255

    return binary

def binary_threshold(img, threshold=(29, 255)):
    """
    Apply binary threshold for single channel of image
    """
    binary = np.zeros_like(img)
    binary[(img <= threshold[1]) & (img >= threshold[0])] = 255
    return binary

def get_hls(img, th_h, th_l, th_s):
    """
    Taking an image, converts it to HLS and apply threshold for extract
    features
    """

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    row, col = img[:2]
    
    H = hls[180:, :, 0]
    L = hls[180:, :, 1]
    S = hls[180:, :, 2]

    h_channel = binary_threshold(H, th_h)
    l_channel = binary_threshold(L, th_l)
    s_channel = binary_threshold(S, th_s)

    # combine channel after threshold
    combine = np.zeros_like(h_channel).astype(np.uint8)
    combine[((s_channel > 1) & (l_channel == 0)) | ((s_channel == 0) & (h_channel > 1) & (l_channel > 1))] = 255

    return combine

def direction_threshold(img, sobel_kernel=3, threshold=(0.7, 1.3)):
    """
    Apply sobel x and y, then computes the direction of the gradients,
    after that, applies a threshold
    """

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Compute the direction of the gradient
    gradient_dir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))

    # create binary threshold of direction
    binary = np.zeros_like(gradient_dir)
    binary[(gradient_dir <= threshold[1]) & (gradient_dir >= threshold[0])] = 255

    return binary.astype(np.uint8)


def magnitude_threshold(img, sobel_kernel=3, threshold=(0, 255)):
    """
    Applies a binary threshold for magnitude
    """

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    gradient_magnitude = ((gradient_magnitude/np.max(gradient_magnitude))*255).astype(np.uint8)

    # create binary threshold of magnitude
    binary = np.zeros_like(gradient_magnitude)
    binary[(binary <= threshold[1]) & (binary >= threshold[0])] = 255

    return binary

def combine_gradients(img, threshold_x, threshold_y, threshold_mag, threshold_dir):
    """
    Combine all binary thresholds to single
    """
    row, col = img[:2]

    tmp = img.copy()
    tmp = tmp[180:, :, 2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ROI = gray[180:, :]
    gray = ROI
    # gray = gray[360:, :]
    sobel_x = abs_sobel_threshold(gray, 'x', threshold_x)
    sobel_y = abs_sobel_threshold(gray, 'y', threshold_y)
    magnitude = magnitude_threshold(gray, 3, threshold_mag)
    direction = direction_threshold(gray, 15, threshold_dir)

    # Combine gradients to single image
    combine = np.zeros_like(sobel_y).astype(np.uint8)
    combine[((sobel_x > 1) & (magnitude > 1) & (direction > 1)) | ((sobel_y > 1) & (sobel_x > 1))] = 255
    cv2.imwrite('combine.png', combine)

    return combine


def combine_hls_grandient(grad, hls):
    """
    Combine hls and gradient to single image
    """

    combine = np.zeros_like(grad).astype(np.uint8)
    combine[grad > 1] = 100
    combine[hls > 1 ] = 255

    return combine