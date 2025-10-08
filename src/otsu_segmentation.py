import cv2
import numpy as np

def otsu_segmentation(img_rgb):
    """
    Segmenta a imagem em regiões de céu e não-céu usando OTSU.
    Equações (3)-(7) do artigo.
    
    Args:
        img_rgb: Imagem RGB de entrada (uint8 ou float32)
    
    Returns:
        sky_mask: Máscara binária (True para céu, False para não-céu)
        threshold: Valor do limiar ótimo encontrado
    """
    # Converter para BGR se necessário
    if img_rgb.dtype == np.float32:
        img_bgr = img_rgb.astype(np.uint8)
    else:
        img_bgr = img_rgb
    
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    NUM = img_gray.size
    P = hist / NUM  # Equação (3)
    
    sigma2_list = []
    for t in range(256):
        # Equação (4): Probabilidades das regiões
        w1 = np.sum(P[:t+1])  # não-céu (0 a t)
        w2 = np.sum(P[t+1:])  # céu (t+1 a 255)
        
        if w1 == 0 or w2 == 0:
            sigma2_list.append(0)
            continue
        
        # Equação (5): Médias das regiões
        u1 = np.sum(np.arange(0, t+1) * P[:t+1]) / w1
        u2 = np.sum(np.arange(t+1, 256) * P[t+1:]) / w2
        
        # Equação (6): Expectativa geral
        u = w1 * u1 + w2 * u2
        
        # Equação (7): Variância entre clusters
        sigma2 = w1 * (u1 - u)**2 + w2 * (u2 - u)**2
        sigma2_list.append(sigma2)
    
    t_opt = np.argmax(sigma2_list)
    
    # Céu: valores > threshold (valor 1)
    # Não-céu: valores <= threshold (valor 0)
    sky_mask = img_gray > t_opt
    
    return sky_mask, t_opt
