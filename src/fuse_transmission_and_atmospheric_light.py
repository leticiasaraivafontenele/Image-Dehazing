import numpy as np
import matplotlib.pyplot as plt

def fuse_transmission_and_atmospheric_light(t_dark, t_bright, A_dark, A_bright, sky_mask):
    """
    Funde os mapas de transmissão e luzes atmosféricas do DCP e BCP.
    Implementa as Equações (13), (14) e (15) do artigo.
    
    Args:
        t_dark: Mapa de transmissão DCP
        t_bright: Mapa de transmissão BCP
        A_dark: Luz atmosférica DCP (vetor RGB)
        A_bright: Luz atmosférica BCP (vetor RGB)
        sky_mask: Máscara binária (True para céu)
    
    Returns:
        t_fused: Mapa de transmissão fundido
        A_fused: Luz atmosférica fundida (vetor RGB)
    """
    H, W = sky_mask.shape
    total_pixels = H * W
    
    # Calcular Z: número de pixels no céu (valores > threshold)
    Z = np.sum(sky_mask)
    non_sky_pixels = total_pixels - Z
    
    # Equação (13): λ = Z / (H × W)
    lambda_weight = Z / total_pixels
    
    print(f"\n=== Parâmetros de Fusão ===")
    print(f"Dimensões: {H} x {W} = {total_pixels} pixels")
    print(f"Pixels do céu (Z): {Z} ({100*lambda_weight:.2f}%)")
    print(f"Pixels não-céu: {non_sky_pixels} ({100*(1-lambda_weight):.2f}%)")
    print(f"λ (lambda): {lambda_weight:.4f}")
    
    # Calcular σ (sigma) - parâmetro adaptativo
    # σ = (1/10^e) - (z/(H×W - Z)) - 0.05
    e = np.e
    base_sigma = (1 / (10 ** e))
    
    sigma = np.zeros_like(t_dark)
    
    # Para região do céu
    if Z > 0:
        sigma_sky = base_sigma - (Z / max(non_sky_pixels, 1)) - 0.05
        sigma[sky_mask] = sigma_sky
        print(f"σ (céu): {sigma_sky:.6f}")
    
    # Para região não-céu
    if non_sky_pixels > 0:
        sigma_non_sky = base_sigma - (Z / max(non_sky_pixels, 1)) - 0.05
        sigma[~sky_mask] = sigma_non_sky
        print(f"σ (não-céu): {sigma_non_sky:.6f}")
    
    # Limitar sigma ao intervalo [-0.05, 0.05]
    sigma = np.clip(sigma, -0.05, 0.05)
    
    # Equação (14): t(q) = λ·t_bright(q) + (1-λ)·t_dark(q) - σ
    t_fused = lambda_weight * t_bright + (1 - lambda_weight) * t_dark - sigma
    t_fused = np.clip(t_fused, 0.1, 1.0)
    
    # Equação (15): A = λ·A_DCP + (1-λ)·A_bright
    A_fused = lambda_weight * A_dark + (1 - lambda_weight) * A_bright
    
    print(f"\nLuz Atmosférica:")
    print(f"  A_dark (DCP): {A_dark}")
    print(f"  A_bright (BCP): {A_bright}")
    print(f"  A_fused: {A_fused}")
    
    return t_fused, A_fused