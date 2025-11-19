import numpy as np
from guided_filter import guided_filter

def recover_image(img_rgb, t_fused, A_fused):

    t_fused = np.maximum(t_fused, 0.1)
    t_fused_filter = guided_filter(t_fused, t_fused, radius=60, eps=0.0001)
    
    t_fused_3d = np.repeat(t_fused_filter[:, :, np.newaxis], 3, axis=2)
    
    J = (img_rgb - A_fused) / t_fused_3d + A_fused
    J = np.clip(J, 0, 255)
    
    return J.astype(np.uint8)