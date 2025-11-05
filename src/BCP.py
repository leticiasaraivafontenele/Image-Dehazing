import cv2
import numpy as np

def bright_channel_prior(img_rgb, sky_mask, window_size=15, k=0.1, refine=True, guided_radius=60, guided_eps=0.001):
    """
    Calcula o Bright Channel Prior para regiões do céu.
    Equações (9), (10), (11) e (12) do artigo.
    
    Args:
        img_rgb: Imagem RGB (float32, 0-255)
        sky_mask: Máscara do céu (True = céu)
        window_size: Tamanho da janela Ω(q)
        k: Fator de ajuste (0.05 a 0.15)
        refine: Se True, aplica guided filter
        guided_radius: Raio do guided filter
        guided_eps: Epsilon do guided filter
    
    Returns:
        t_bright: Mapa de transmissão BCP (refinado se refine=True)
        A_bright: Luz atmosférica BCP (vetor RGB)
    """
    
    # --- Equação (9): J^bright(q) = max{max[I^c(q)]} ---
    max_rgb = np.max(img_rgb, axis=2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    bright_channel = cv2.dilate(max_rgb, kernel)
    
    # --- Estimar luz atmosférica A_bright ---
    num_pixels = bright_channel.size
    num_top = max(int(num_pixels * 0.001), 1)
    
    bright_channel_sky = bright_channel.copy()
    bright_channel_sky[~sky_mask] = -np.inf
    
    if np.all(np.isinf(bright_channel_sky)):
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
    
    # --- Equação (10): Transmissão inicial ---
    A_c_bright = np.max(A_bright)
    b = np.abs(bright_channel - A_c_bright)
    mean_b = np.mean(b)
    
    # --- Equação (11): Aplicar ajuste com k ---
    A_c_safe = max(A_c_bright, 1e-6)
    denominator = max(255.0 - A_c_safe, 1e-6)
    
    t_bright = np.where(
        b < mean_b,
        (b + k) / denominator,
        b / denominator
    )
    
    # --- Equação (12): Limitar transmissão ---
    t_bright = np.clip(t_bright, 0.0, 1.0)
    
    # --- Refinamento com Guided Filter ---
    if refine:
        print("  Aplicando Guided Filter no t_BCP...")
        # Preparar imagem guia (grayscale normalizada)
        img_gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        img_gray = img_gray.astype(np.float64) / 255.0
        
        # Aplicar guided filter
        from guided_filter import guided_filter
        t_bright = guided_filter(img_gray, t_bright.astype(np.float64), radius=guided_radius, eps=guided_eps)
        t_bright = np.clip(t_bright, 0.0, 1.0)
    
    return t_bright, A_bright