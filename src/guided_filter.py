import cv2

def guided_filter(I, p, radius=60, eps=0.0001):
    """
    Implementação do Guided Filter para refinar mapas de transmissão.
    Baseado em: He et al. "Guided Image Filtering" (ECCV 2010)
    
    Args:
        I: Imagem guia (grayscale)
        p: Imagem a ser filtrada
        radius: Raio da janela
        eps: Parâmetro de regularização
    
    Returns:
        q: Imagem filtrada
    """
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