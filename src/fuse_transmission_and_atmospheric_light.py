import numpy as np
import matplotlib.pyplot as plt

def fuse_transmission_and_atmospheric_light(t_dark, t_bright, A_dark, A_bright, sky_mask):

    H, W = sky_mask.shape
    total_pixels = H * W
    
    Z = np.sum(sky_mask)
    non_sky_pixels = total_pixels - Z
    
    lambda_weight = Z / total_pixels
    
    print(f"\n=== Parâmetros de Fusão ===")
    print(f"Dimensões: {H} x {W} = {total_pixels} pixels")
    print(f"Pixels do céu (Z): {Z} ({100*lambda_weight:.2f}%)")
    print(f"Pixels não-céu: {non_sky_pixels} ({100*(1-lambda_weight):.2f}%)")
    print(f"λ (lambda): {lambda_weight:.4f}")
    
    e = np.e
    
    sigma = np.zeros_like(t_dark)
    
    if Z > 0:
        sigma_sky = (1 / 10 ) *( e**( - (Z / max(non_sky_pixels, 1)))) - 0.05
        sigma[sky_mask] = sigma_sky
        print(f"σ (céu): {sigma_sky:.6f}")
    
    if non_sky_pixels > 0:
        sigma_non_sky = (1 / 10 ) *( e**( - (Z / max(non_sky_pixels, 1)))) - 0.05
        sigma[~sky_mask] = sigma_non_sky
        print(f"σ (não-céu): {sigma_non_sky:.6f}")
    
    t_fused = lambda_weight * t_bright + (1 - lambda_weight) * t_dark - sigma
    
    A_fused = lambda_weight * A_dark + (1 - lambda_weight) * A_bright
    
    print(f"\nLuz Atmosférica:")
    print(f"  A_dark (DCP): {A_dark}")
    print(f"  A_bright (BCP): {A_bright}")
    print(f"  A_fused: {A_fused}")
    
    return t_fused, A_fused