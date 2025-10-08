import cv2
import numpy as np


def bright_channel_prior(img_rgb, sky_mask, window_size=15, k=0.1):
    """
    Calcula o Bright Channel Prior para regiões do céu.
    Equações (9), (10), (11) e (12) do artigo.
    
    Args:
        img_rgb: Imagem RGB (float32, 0-255)
        sky_mask: Máscara do céu (True = céu)
        window_size: Tamanho da janela Ω(q)
        k: Fator de ajuste (0.05 a 0.15)
    
    Returns:
        t_bright: Mapa de transmissão BCP
        A_bright: Luz atmosférica BCP (vetor RGB)
    """
    # Equação (9): J^bright(q) = max{max[I^c(q)]}, J^bright(q) → 255
    # Bright channel: máximo entre os canais RGB
    max_rgb = np.max(img_rgb, axis=2)
    
    # Aplicar filtro máximo na vizinhança Ω(q)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    bright_channel = cv2.dilate(max_rgb, kernel)
    
    # Estimar luz atmosférica A_bright
    # Selecionar 0.1% pixels mais brilhantes no bright channel
    num_pixels = bright_channel.size
    num_top = max(int(num_pixels * 0.001), 1)
    
    # Apenas considerar região do céu para estimativa
    bright_channel_sky = bright_channel.copy()
    bright_channel_sky[~sky_mask] = -np.inf
    
    if np.all(np.isinf(bright_channel_sky)):
        # Se não há céu, usar toda a imagem
        indices = np.unravel_index(
            np.argsort(bright_channel.ravel())[-num_top:], 
            bright_channel.shape
        )
    else:
        indices = np.unravel_index(
            np.argsort(bright_channel_sky.ravel())[-num_top:], 
            bright_channel.shape
        )
    
    A_bright = np.mean(img_rgb[indices], axis=0)
    
    # Equação (10): t_bright(q) = (max[max I^c(y)] - A^c_bright) / (255 - A^c_bright)
    # Mas primeiro calcular b conforme descrito no artigo
    b = np.abs(bright_channel - np.max(A_bright))
    mean_b = np.mean(b)
    
    # Equação (11): Aplicar ajuste com k
    A_bright_max = np.max(A_bright)
    A_bright_safe = np.maximum(A_bright_max, 1e-6)
    denominator = max(255 - A_bright_safe, 1e-6)
    
    t_bright = np.where(
        b < mean_b,
        (b + k) / denominator,  # if b < mean(b)
        b / denominator          # if b >= mean(b)
    )
    
    # Equação (12): t_bright(q) = min(t_bright(q), 1)
    t_bright = np.minimum(t_bright, 1.0)
    t_bright = np.maximum(t_bright, 0.0)
    
    return t_bright, A_bright