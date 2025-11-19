import cv2
import numpy as np

def otsu_segmentation(img_rgb):

    if img_rgb.dtype == np.float32:
        img_bgr = img_rgb.astype(np.uint8)
    else:
        img_bgr = img_rgb
    
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    NUM = img_gray.size
    P = hist / NUM 
    
    sigma2_list = []
    for t in range(256):
        w1 = np.sum(P[:t+1])  
        w2 = np.sum(P[t+1:]) 
        
        if w1 == 0 or w2 == 0:
            sigma2_list.append(0)
            continue
        
        u1 = np.sum(np.arange(0, t+1) * P[:t+1]) / w1
        u2 = np.sum(np.arange(t+1, 256) * P[t+1:]) / w2
        
        u = w1 * u1 + w2 * u2
        
        sigma2 = w1 * (u1 - u)**2 + w2 * (u2 - u)**2
        sigma2_list.append(sigma2)
    
    t_opt = np.argmax(sigma2_list)

    sky_mask = img_gray > t_opt
    
    return sky_mask, t_opt
