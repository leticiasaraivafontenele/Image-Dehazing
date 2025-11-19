import cv2
import numpy as np
import matplotlib.pyplot as plt
from otsu_segmentation import otsu_segmentation
from BCP import bright_channel_prior
from DCP import dark_channel_prior
from fuse_transmission_and_atmospheric_light import fuse_transmission_and_atmospheric_light
from recover_image import recover_image

def single_image_dehazing(img_path, window_size=15, k=0.1, omega=0.95):

    print("=== Single Image Dehazing - Algorithm 1 ===\n")
    
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Não foi possível carregar a imagem: {img_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    print(f"Imagem carregada: {img_rgb.shape}")
    
    print("\n[Passo 1] Segmentação OTSU...")
    sky_mask, threshold = otsu_segmentation(img_bgr)
    print(f"Threshold ótimo: {threshold}")
    
    print("\n[Passo 2.1] Dark Channel Prior (não-céu)...")
    t_dark, A_dark = dark_channel_prior(img_rgb, sky_mask, window_size, omega)
    
    print("\n[Passo 2.2] Bright Channel Prior (céu)...")
    t_bright, A_bright = bright_channel_prior(img_rgb, sky_mask, window_size, k)
    
    print("\n[Passo 3] Fusão dos mapas...")
    t_fused, A_fused = fuse_transmission_and_atmospheric_light(
        t_dark, t_bright, A_dark, A_bright, sky_mask
    )
    
    print("\n[Passo 4] Recuperando imagem sem neblina...")
    J = recover_image(img_rgb, t_fused, A_fused)
    
    resultados = {
        'original': img_rgb,
        'sky_mask': sky_mask,
        't_dark': t_dark,
        't_bright': t_bright,
        't_fused': t_fused,
        'dehazed': J
    }
    
    print("\n=== Processamento Concluído ===\n")
    return J, resultados