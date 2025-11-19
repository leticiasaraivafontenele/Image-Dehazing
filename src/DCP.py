import numpy as np
from scipy.ndimage import minimum_filter
import cv2

def dark_channel_prior(img_rgb, sky_mask, window_size=15, omega=0.95):

    I_norm = img_rgb / 255.0
    
    dark_channel = np.min(I_norm, axis=2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(dark_channel, kernel)
    
    num_pixels = dark_channel.size
    num_top = max(int(num_pixels * 0.001), 1)
    
    dark_channel_non_sky = dark_channel.copy()
    dark_channel_non_sky[sky_mask] = -np.inf
    
    indices = np.unravel_index(
        np.argsort(dark_channel_non_sky.ravel())[-num_top:], 
        dark_channel.shape
    )
    A_dark = np.mean(I_norm[indices], axis=0)
    
    A_dark_safe = np.maximum(A_dark, 1e-6)
    I_normalized = I_norm / A_dark_safe

    min_channel = np.min(I_normalized, axis=2)
    
    min_channel_filtered = cv2.erode(min_channel.astype(np.float32), kernel)
    
    t_dark = 1 - omega * min_channel_filtered
    t_dark = np.clip(t_dark, 0.1, 1)
    
    return t_dark, A_dark
