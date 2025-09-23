import cv2
import numpy as np

img_rgb = cv2.imread("0046_0.9_0.16.jpg")    

img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([img], [0], None, [256], [0,256]).flatten()
NUM = img.size
P = hist / NUM 
sigma2_list = []

for t in range(256):
    w1 = np.sum(P[:t+1])          # Probability sum for class 1 (intensities 0 to t)
    w2 = np.sum(P[t+1:])          # Probability sum for class 2 (intensities t+1 to 255)

    if w1 == 0 or w2 == 0:
        sigma2_list.append(0)
        continue

    u1 = np.sum(np.arange(0, t+1) * P[:t+1]) / w1  # Mean for class 1 (0 to t)
    u2 = np.sum(np.arange(t+1, 256) * P[t+1:]) / w2  # Mean for class 2 (t+1 to 255)

    u = w1 * u1 + w2 * u2    # Total mean

    sigma2 = w1 * (u1 - u)**2 + w2 * (u2 - u)**2
    sigma2_list.append(sigma2)
    



t_opt = np.argmax(sigma2_list)

img_binary = img > t_opt  # True/False
img_binary = img_binary.astype(np.uint8) * 255  # 0 ou 255

cv2.imwrite("resultado_otsu.jpg", img_binary)