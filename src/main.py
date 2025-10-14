import cv2
import numpy as np
import matplotlib.pyplot as plt
from otsu_segmentation import otsu_segmentation
from BCP import bright_channel_prior
from DCP import dark_channel_prior
from fuse_transmission_and_atmospheric_light import fuse_transmission_and_atmospheric_light
from recover_image import recover_image

def single_image_dehazing(img_path, window_size=15, k=0.1, omega=0.95):
    """
    Pipeline completo do Algorithm 1 do artigo.
    
    Args:
        img_path: Caminho da imagem com neblina
        window_size: Tamanho da janela (padrão 15)
        k: Fator de ajuste BCP (0.05-0.15)
        omega: Parâmetro DCP (padrão 0.95)
    
    Returns:
        J: Imagem sem neblina
        resultados: Dicionário com resultados intermediários
    """
    print("=== Single Image Dehazing - Algorithm 1 ===\n")
    
    # Carregar imagem
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Não foi possível carregar a imagem: {img_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    print(f"Imagem carregada: {img_rgb.shape}")
    
    # Passo 1: Segmentação OTSU
    print("\n[Passo 1] Segmentação OTSU...")
    sky_mask, threshold = otsu_segmentation(img_bgr)
    print(f"Threshold ótimo: {threshold}")
    
    # Passo 2.1: DCP para região não-céu
    print("\n[Passo 2.1] Dark Channel Prior (não-céu)...")
    t_dark, A_dark = dark_channel_prior(img_rgb, sky_mask, window_size, omega)
    
    # Passo 2.2: BCP para região do céu
    print("\n[Passo 2.2] Bright Channel Prior (céu)...")
    t_bright, A_bright = bright_channel_prior(img_rgb, sky_mask, window_size, k)
    
    # Passo 3: Fusão
    print("\n[Passo 3] Fusão dos mapas...")
    t_fused, A_fused = fuse_transmission_and_atmospheric_light(
        t_dark, t_bright, A_dark, A_bright, sky_mask
    )
    
    # Passo 4: Recuperar imagem
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


def visualize_results(resultados, save_path='resultado_completo.jpg'):
    """Visualiza todos os resultados intermediários"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(resultados['original'].astype(np.uint8))
    axes[0, 0].set_title('(a) Imagem Original com Neblina', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(resultados['sky_mask'], cmap='gray')
    axes[0, 1].set_title('(b) Segmentação OTSU\n(Branco=Céu, Preto=Não-céu)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    im1 = axes[0, 2].imshow(resultados['t_dark'], cmap='jet', vmin=0, vmax=1)
    axes[0, 2].set_title('(c) t_DCP (Dark Channel Prior)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    im2 = axes[1, 0].imshow(resultados['t_bright'], cmap='jet', vmin=0, vmax=1)
    axes[1, 0].set_title('(d) t_BCP (Bright Channel Prior)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im3 = axes[1, 1].imshow(resultados['t_fused'], cmap='jet', vmin=0, vmax=1)
    axes[1, 1].set_title('(e) t_fused (Mapa Fundido)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    axes[1, 2].imshow(resultados['dehazed'])
    axes[1, 2].set_title('(f) Imagem Recuperada (Sem Neblina)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Resultados salvos em: {save_path}")
    plt.show()


# ============= Exemplo de Uso =============
if __name__ == "__main__":
    import sys
    
    # Permitir passar o caminho da imagem como argumento
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "/home/leticia/faculdade/pdi/SOTS/outdoor/hazy/0047_0.9_0.12.jpg"
    
    print(f"Processando: {img_path}\n")
    
    # Processar imagem
    J, resultados = single_image_dehazing(
        img_path=img_path,
        window_size=15,
        k=0.1,
        omega=0.95
    )
    
    # Visualizar resultados
    visualize_results(resultados)
    
    # Salvar imagem final
    cv2.imwrite('imagem_sem_neblina_final.jpg', cv2.cvtColor(J, cv2.COLOR_RGB2BGR))
    print("Imagem final salva: imagem_sem_neblina_final.jpg")