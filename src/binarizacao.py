import cv2
import numpy as np
import sys
import os

if len(sys.argv) < 2:
    print("Adicione o path da imagem.")
    sys.exit(1) 

img_path = sys.argv[1]
img_rgb = cv2.imread(img_path)    

if img_rgb is None:
    print(f"Erro ao carreagar a imagem '{img_path}'")
    sys.exit(1)

img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

histogram = cv2.calcHist([img], [0], None, [256], [0,256]).flatten()
pixels = img.size
P = histogram / pixels 
sigma2_list = []

for t in range(256):
    p_dark = np.sum(P[:t+1])
    p_bright = np.sum(P[t+1:])

    if p_dark == 0 or p_bright == 0:
        sigma2_list.append(0)
        continue

    m_dark = np.sum(np.arange(0, t+1) * P[:t+1]) / p_dark
    m_bright = np.sum(np.arange(t+1, 256) * P[t+1:]) / p_bright

    m_total = p_dark * m_dark + p_bright * m_bright

    sigma2 = p_dark * (m_dark - m_total)**2 + p_bright * (m_bright - m_total)**2
    sigma2_list.append(sigma2)
    

t_opt = np.argmax(sigma2_list)

img_binary = img > t_opt 
img_binary = img_binary.astype(np.uint8) * 255


out_dir = "img_bin"
os.makedirs(out_dir, exist_ok=True)

base_name = os.path.basename(img_path)
img_name, img_ext = os.path.splitext(base_name)

name_result = f"{img_name}_bin{img_ext}"

out_path = os.path.join(out_dir, name_result)

cv2.imwrite(out_path, img_binary)
