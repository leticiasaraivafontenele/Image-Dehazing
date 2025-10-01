import cv2
import numpy as np
import os

input_path = "imgs/0046_0.9_0.16.jpg"

img_rgb = cv2.imread(input_path)
img_rgb = img_rgb.astype(np.float32)

window_size = 15 
k = 0.1 

#cor do fog
max_rgb = np.max(img_rgb, axis=2)
kernel = np.ones((window_size, window_size), np.uint8)
bright_channel = cv2.dilate(max_rgb, kernel)

num_pixels = bright_channel.size
num_top = max(int(num_pixels * 0.001), 1)
indices = np.unravel_index(np.argsort(bright_channel.ravel())[-num_top:], bright_channel.shape)

A_bright = np.mean(img_rgb[indices], axis=0)

#compara as cores img com o fog
t_bright = np.zeros_like(max_rgb)
for c in range(3):
    I_c = img_rgb[:, :, c]
    A_c = A_bright[c]
    b = np.abs(bright_channel - A_c)
    mean_b = np.mean(b)
    
    t_bright_c = np.where(b < mean_b, (b + k) / (255 - A_c), b / (255 - A_c))
    t_bright_c = np.clip(t_bright_c, 0, 1)
    
    t_bright += t_bright_c / 3


out_dir = "BCP_map"
os.makedirs(out_dir, exist_ok=True)

base_name = os.path.basename(input_path)
img_name, img_ext = os.path.splitext(base_name)
name_result = f"{img_name}_bcp{img_ext}"

output_path = os.path.join(out_dir, name_result)
t_bright_to_save = (t_bright * 255).astype(np.uint8)

cv2.imwrite(output_path, t_bright_to_save)
