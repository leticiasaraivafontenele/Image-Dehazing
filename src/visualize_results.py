import numpy as np
import matplotlib.pyplot as plt

def visualize_results(resultados, save_path='resultado_completo.jpg'):
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