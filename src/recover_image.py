import numpy as np
from guided_filter import guided_filter

def recover_image(img_rgb, t_fused, A_fused):
    """
    Recupera a imagem sem neblina usando a equação de formação atmosférica.
    Equação (16): J(q) = (I(q) - A) / max(t(q), 0.1) + A
    
    Args:
        img_rgb: Imagem com neblina (float32, 0-255)
        t_fused: Mapa de transmissão fundido
        A_fused: Luz atmosférica fundida (vetor RGB)
    
    Returns:
        J: Imagem recuperada sem neblina
    """
    # Garantir limite mínimo de 0.1 para transmissão
    t_fused = np.maximum(t_fused, 0.1)
    t_fused_filter = guided_filter(t_fused, t_fused, radius=60, eps=0.0001)
    
    # Expandir t_fused para 3 canais
    t_fused_3d = np.repeat(t_fused_filter[:, :, np.newaxis], 3, axis=2)
    
    # Equação (16): J(q) = (I(q) - A) / max(t(q), 0.1) + A
    J = (img_rgb - A_fused) / t_fused_3d + A_fused
    J = np.clip(J, 0, 255)
    
    return J.astype(np.uint8)