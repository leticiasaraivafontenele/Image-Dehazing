import numpy as np
import cv2

def dark_channel_prior(img_rgb, sky_mask, window_size=15, omega=0.95, refine=True, guided_radius=60, guided_eps=0.001):

    dark_channel = np.min(img_rgb, axis=2)
    
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
    
    A_dark = np.mean(img_rgb[indices], axis=0)
    
    A_dark_safe = np.maximum(A_dark, 1e-6)
    
    I_normalized = np.zeros_like(img_rgb)
    for c in range(3):
        I_normalized[:, :, c] = img_rgb[:, :, c] / A_dark_safe[c]
    
    min_channels = np.min(I_normalized, axis=2)
    min_neighborhood = cv2.erode(min_channels.astype(np.float32), kernel)
    
    t_dark = 1 - omega * min_neighborhood
    t_dark = np.clip(t_dark, 0.1, 1.0)
    
    if refine:
        print("  Aplicando Guided Filter no t_DCP...")
        img_gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        img_gray = img_gray.astype(np.float64) / 255.0
        
        from guided_filter import guided_filter
        t_dark = guided_filter(img_gray, t_dark.astype(np.float64), radius=guided_radius, eps=guided_eps)
        t_dark = np.clip(t_dark, 0.1, 1.0)
    
    return t_dark, A_dark