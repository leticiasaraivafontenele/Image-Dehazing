import numpy as np
from scipy.ndimage import minimum_filter
import cv2

def dark_channel_prior(img_rgb, sky_mask, window_size=15, omega=0.95):
    """
    Calcula o Dark Channel Prior para regiões não-céu.
    Equação (8) do artigo.
    
    Args:
        img_rgb: Imagem RGB (float32, 0-255)
        sky_mask: Máscara do céu (True = céu)
        window_size: Tamanho da janela Ω(q)
        omega: Parâmetro para manter neblina (0.95)
    
    Returns:
        t_dark: Mapa de transmissão DCP
        A_dark: Luz atmosférica DCP (vetor RGB)
    """
    # Normalizar para 0-1 temporariamente
    I_norm = img_rgb / 255.0
    
    # Dark channel: mínimo entre os canais RGB
    dark_channel = np.min(I_norm, axis=2)
    
    # Aplicar filtro mínimo na vizinhança Ω(q)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(dark_channel, kernel)
    
    # Estimar luz atmosférica A_DCP
    # Selecionar 0.1% pixels mais brilhantes no dark channel
    num_pixels = dark_channel.size
    num_top = max(int(num_pixels * 0.001), 1)
    
    # Apenas considerar região não-céu para estimativa
    dark_channel_non_sky = dark_channel.copy()
    dark_channel_non_sky[sky_mask] = -np.inf
    
    indices = np.unravel_index(
        np.argsort(dark_channel_non_sky.ravel())[-num_top:], 
        dark_channel.shape
    )
    A_dark = np.mean(I_norm[indices], axis=0)
    
    # Equação (8): t_DCP(q) = 1 - ω * min(I^c(y)/A^c)
    A_dark_safe = np.maximum(A_dark, 1e-6)
    I_normalized = I_norm / A_dark_safe
    #estranho
    
    # Mínimo entre canais
    min_channel = np.min(I_normalized, axis=2)
    
    # Mínimo na vizinhança
    min_channel_filtered = cv2.erode(min_channel.astype(np.float32), kernel)
    
    # Calcular transmissão
    t_dark = 1 - omega * min_channel_filtered
    t_dark = np.clip(t_dark, 0.1, 1)
    
    return t_dark, A_dark
