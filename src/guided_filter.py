import cv2
import numpy as np

def guided_filter(I, p, radius=60, eps=0.0001):
    """
    Implementação do Guided Filter para refinar mapas de transmissão.
    Baseado em: He et al. "Guided Image Filtering" (ECCV 2010)
    
    Args:
        I: Imagem guia (grayscale, float64, normalizada 0-1)
        p: Imagem a ser filtrada (float64, 0-1)
        radius: Raio da janela
        eps: Parâmetro de regularização
    
    Returns:
        q: Imagem filtrada
    """
    # Garantir tipos corretos
    I = I.astype(np.float64)
    p = p.astype(np.float64)
    
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (radius, radius))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
    
    q = mean_a * I + mean_b
    return q