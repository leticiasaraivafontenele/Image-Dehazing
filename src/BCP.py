import cv2
import numpy as np


def bright_channel_prior(img_rgb, sky_mask, window_size=15, k=0.1):

    max_rgb = np.max(img_rgb, axis=2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    bright_channel = cv2.dilate(max_rgb, kernel)

    num_pixels = bright_channel.size
    num_top = max(int(num_pixels * 0.001), 1)
    
    bright_channel_sky = bright_channel.copy()
    bright_channel_sky[~sky_mask] = -np.inf
    
    if np.all(np.isinf(bright_channel_sky)):
        indices = np.unravel_index(
            np.argsort(bright_channel.ravel())[-num_top:], 
            bright_channel.shape
        )
    else:
        indices = np.unravel_index(
            np.argsort(bright_channel_sky.ravel())[-num_top:], 
            bright_channel.shape
        )
    
    A_bright = np.mean(img_rgb[indices], axis=0)

    b = np.abs(bright_channel - np.max(A_bright))
    mean_b = np.mean(b)
    
    A_bright_max = np.max(A_bright)
    A_bright_safe = np.maximum(A_bright_max, 1e-6)
    denominator = max(255 - A_bright_safe, 1e-6)
    
    t_bright = np.where(
        b < mean_b,
        (b + k) / denominator,
        b / denominator     
    )
    
    t_bright = np.minimum(t_bright, 1.0)
    t_bright = np.maximum(t_bright, 0.0)
    
    return t_bright, A_bright