import numpy as np
import cv2
from scipy import ndimage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ParticleSwarmOptimizer:
    """Particle Swarm Optimization para encontrar threshold ótimo do Otsu"""
    
    def __init__(self, n_particles=500, max_iter=400, w=0.729, c1=1.49445, c2=1.49445):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive parameter
        self.c2 = c2  # social parameter
    
    def fitness_function(self, threshold, hist):
        """Função fitness baseada na variância entre clusters do Otsu"""
        threshold = int(np.clip(threshold, 0, 255))
        
        # Calcular probabilidades
        total_pixels = np.sum(hist)
        if total_pixels == 0:
            return 0
        
        prob = hist / total_pixels
        
        # Pesos das classes
        w1 = np.sum(prob[:threshold + 1])
        w2 = np.sum(prob[threshold + 1:])
        
        if w1 == 0 or w2 == 0:
            return 0
        
        # Médias das classes
        u1 = np.sum(np.arange(threshold + 1) * prob[:threshold + 1]) / w1
        u2 = np.sum(np.arange(threshold + 1, 256) * prob[threshold + 1:]) / w2
        
        # Variância entre clusters
        variance = w1 * w2 * (u1 - u2) ** 2
        
        return variance
    
    def optimize(self, hist):
        """Otimização PSO para encontrar threshold ótimo"""
        # Inicialização das partículas
        particles = np.random.uniform(0, 255, self.n_particles)
        velocities = np.random.uniform(-10, 10, self.n_particles)
        
        # Melhores posições pessoais e globais
        p_best = particles.copy()
        p_best_fitness = np.array([self.fitness_function(p, hist) for p in particles])
        
        g_best = particles[np.argmax(p_best_fitness)]
        g_best_fitness = np.max(p_best_fitness)
        
        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                # Atualizar velocidade
                r1, r2 = np.random.random(), np.random.random()
                velocities[i] = (self.w * velocities[i] + 
                               self.c1 * r1 * (p_best[i] - particles[i]) +
                               self.c2 * r2 * (g_best - particles[i]))
                
                # Atualizar posição
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], 0, 255)
                
                # Avaliar fitness
                fitness = self.fitness_function(particles[i], hist)
                
                # Atualizar melhor pessoal
                if fitness > p_best_fitness[i]:
                    p_best[i] = particles[i]
                    p_best_fitness[i] = fitness
                    
                    # Atualizar melhor global
                    if fitness > g_best_fitness:
                        g_best = particles[i]
                        g_best_fitness = fitness
        
        return int(g_best)

class ImageDehazer:
    """Implementação do algoritmo de dehazing baseado em BCP e DCP melhorados"""
    
    def __init__(self, omega=0.95, k_adjustment=0.1, patch_size=15):
        self.omega = omega
        self.k_adjustment = k_adjustment
        self.patch_size = patch_size
        self.pso = ParticleSwarmOptimizer()
    
    def segment_sky_regions(self, image: np.ndarray) -> np.ndarray:
        """Segmenta regiões de céu usando Otsu com PSO"""
        # Converter para escala de cinza
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calcular histograma
        hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
        
        # Encontrar threshold ótimo usando PSO
        optimal_threshold = self.pso.optimize(hist)
        
        # Criar máscara binária (1 para céu, 0 para não-céu)
        sky_mask = (gray > optimal_threshold).astype(np.uint8)
        
        return sky_mask
    
    def dark_channel(self, image: np.ndarray) -> np.ndarray:
        """Calcula o dark channel da imagem"""
        # Mínimo entre os canais RGB
        min_rgb = np.min(image, axis=2)
        
        # Erosão com elemento estruturante quadrado
        kernel = np.ones((self.patch_size, self.patch_size))
        dark_channel = ndimage.minimum_filter(min_rgb, size=self.patch_size)
        
        return dark_channel
    
    def bright_channel(self, image: np.ndarray) -> np.ndarray:
        """Calcula o bright channel da imagem"""
        # Máximo entre os canais RGB
        max_rgb = np.max(image, axis=2)
        
        # Dilatação com elemento estruturante quadrado
        bright_channel = ndimage.maximum_filter(max_rgb, size=self.patch_size)
        
        return bright_channel
    
    def estimate_atmospheric_light_dcp(self, image: np.ndarray, dark_ch: np.ndarray) -> np.ndarray:
        """Estima luz atmosférica usando DCP"""
        h, w = dark_ch.shape
        flat_dark = dark_ch.flatten()
        flat_image = image.reshape(-1, 3)
        
        # Selecionar 0.1% dos pixels mais brilhantes no dark channel
        num_pixels = int(0.001 * len(flat_dark))
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        
        # Média dos pixels correspondentes na imagem original
        atmospheric_light = np.mean(flat_image[indices], axis=0)
        
        return atmospheric_light
    
    def estimate_atmospheric_light_bcp(self, image: np.ndarray, bright_ch: np.ndarray) -> np.ndarray:
        """Estima luz atmosférica usando BCP"""
        h, w = bright_ch.shape
        flat_bright = bright_ch.flatten()
        flat_image = image.reshape(-1, 3)
        
        # Selecionar 0.1% dos pixels mais brilhantes no bright channel
        num_pixels = int(0.001 * len(flat_bright))
        indices = np.argpartition(flat_bright, -num_pixels)[-num_pixels:]
        
        # Média dos pixels correspondentes na imagem original
        atmospheric_light = np.mean(flat_image[indices], axis=0)
        
        return atmospheric_light
    
    def estimate_transmission_dcp(self, image: np.ndarray, atmospheric_light: np.ndarray) -> np.ndarray:
        """Estima transmission map usando DCP"""
        # Normalizar pela luz atmosférica
        normalized = image / atmospheric_light
        
        # Dark channel da imagem normalizada
        dark_ch = self.dark_channel(normalized)
        
        # Transmission map
        transmission = 1 - self.omega * dark_ch
        
        return transmission
    
    def estimate_transmission_bcp(self, image: np.ndarray, atmospheric_light: np.ndarray) -> np.ndarray:
        """Estima transmission map usando BCP melhorado"""
        # Bright channel da imagem
        bright_ch = self.bright_channel(image)
        
        # Cálculo melhorado conforme o artigo
        max_bright = np.max(bright_ch)
        b = np.abs(bright_ch - np.max(atmospheric_light))
        mean_b = np.mean(b)
        
        # Aplicar melhoria proposta
        b_improved = np.where(b < mean_b, b + self.k_adjustment, b)
        
        # Transmission map
        transmission = b_improved / (255 - np.max(atmospheric_light))
        
        # Limitar a 1
        transmission = np.minimum(transmission, 1.0)
        
        return transmission
    
    def guided_filter(self, guide: np.ndarray, src: np.ndarray, radius: int = 60, eps: float = 0.001) -> np.ndarray:
        """Filtro guiado para refinar transmission map"""
        if len(guide.shape) == 3:
            guide = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
        
        # Converter para float
        guide = guide.astype(np.float64) / 255.0
        src = src.astype(np.float64)
        
        # Médias usando filtro box
        mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
        mean_src = cv2.boxFilter(src, -1, (radius, radius))
        mean_guide_src = cv2.boxFilter(guide * src, -1, (radius, radius))
        
        # Covariância e variância
        cov_guide_src = mean_guide_src - mean_guide * mean_src
        var_guide = cv2.boxFilter(guide * guide, -1, (radius, radius)) - mean_guide * mean_guide
        
        # Coeficientes lineares
        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide
        
        # Médias dos coeficientes
        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))
        
        # Resultado filtrado
        result = mean_a * guide + mean_b
        
        return result
    
    def fuse_parameters(self, t_dcp: np.ndarray, t_bcp: np.ndarray, 
                       a_dcp: np.ndarray, a_bcp: np.ndarray, 
                       sky_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fusão dos parâmetros de céu e não-céu"""
        # Calcular peso lambda
        Z = np.sum(sky_mask)
        H, W = sky_mask.shape
        lambda_weight = Z / (H * W)
        
        # Parâmetro sigma adaptativo
        if (H * W - Z) > 0:
            sigma = (1/10) * np.exp(-Z / (H * W - Z)) - 0.05
        else:
            sigma = -0.05
        
        # Fusão do transmission map
        t_fused = lambda_weight * t_bcp + (1 - lambda_weight) * t_dcp - sigma
        
        # Fusão da luz atmosférica
        a_fused = lambda_weight * a_bcp + (1 - lambda_weight) * a_dcp
        
        return t_fused, a_fused
    
    def recover_image(self, image: np.ndarray, transmission: np.ndarray, 
                     atmospheric_light: np.ndarray, t_min: float = 0.1) -> np.ndarray:
        """Recupera imagem clara usando o modelo físico"""
        # Limitar transmission map
        transmission = np.maximum(transmission, t_min)
        
        # Expandir dimensões se necessário
        if len(transmission.shape) == 2:
            transmission = transmission[:, :, np.newaxis]
        
        # Recuperar imagem
        recovered = (image - atmospheric_light) / transmission + atmospheric_light
        
        # Limitar valores entre 0 e 255
        recovered = np.clip(recovered, 0, 255).astype(np.uint8)
        
        return recovered
    
    def dehaze(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Algoritmo principal de dehazing"""
        # Converter para float
        image_float = image.astype(np.float64)
        
        print("Etapa 1: Segmentando regiões de céu...")
        # 1. Segmentar regiões de céu
        sky_mask = self.segment_sky_regions(image)
        
        print("Etapa 2: Estimando parâmetros para região não-céu (DCP)...")
        # 2. Estimar parâmetros para região não-céu usando DCP
        dark_ch = self.dark_channel(image_float)
        a_dcp = self.estimate_atmospheric_light_dcp(image_float, dark_ch)
        t_dcp = self.estimate_transmission_dcp(image_float, a_dcp)
        
        print("Etapa 3: Estimando parâmetros para região de céu (BCP)...")
        # 3. Estimar parâmetros para região de céu usando BCP melhorado
        bright_ch = self.bright_channel(image_float)
        a_bcp = self.estimate_atmospheric_light_bcp(image_float, bright_ch)
        t_bcp = self.estimate_transmission_bcp(image_float, a_bcp)
        
        print("Etapa 4: Fundindo parâmetros...")
        # 4. Fundir parâmetros
        t_fused, a_fused = self.fuse_parameters(t_dcp, t_bcp, a_dcp, a_bcp, sky_mask)
        
        print("Etapa 5: Refinando transmission map...")
        # 5. Refinar transmission map com filtro guiado
        t_refined = self.guided_filter(image, t_fused)
        
        print("Etapa 6: Recuperando imagem clara...")
        # 6. Recuperar imagem clara
        dehazed = self.recover_image(image_float, t_refined, a_fused)
        
        # Informações de depuração
        debug_info = {
            'sky_mask': sky_mask,
            'dark_channel': dark_ch,
            'bright_channel': bright_ch,
            'transmission_dcp': t_dcp,
            'transmission_bcp': t_bcp,
            'transmission_fused': t_fused,
            'transmission_refined': t_refined,
            'atmospheric_light_dcp': a_dcp,
            'atmospheric_light_bcp': a_bcp,
            'atmospheric_light_fused': a_fused
        }
        
        return dehazed, debug_info

def visualize_results(original: np.ndarray, dehazed: np.ndarray, debug_info: dict):
    """Visualiza resultados do dehazing"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Imagem original e resultado
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Imagem Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(dehazed)
    axes[0, 1].set_title('Imagem Dehazed')
    axes[0, 1].axis('off')
    
    # Máscara de segmentação
    axes[0, 2].imshow(debug_info['sky_mask'], cmap='gray')
    axes[0, 2].set_title('Máscara de Céu')
    axes[0, 2].axis('off')
    
    # Dark channel
    axes[0, 3].imshow(debug_info['dark_channel'], cmap='gray')
    axes[0, 3].set_title('Dark Channel')
    axes[0, 3].axis('off')
    
    # Bright channel
    axes[1, 0].imshow(debug_info['bright_channel'], cmap='gray')
    axes[1, 0].set_title('Bright Channel')
    axes[1, 0].axis('off')
    
    # Transmission maps
    axes[1, 1].imshow(debug_info['transmission_dcp'], cmap='gray')
    axes[1, 1].set_title('Transmission DCP')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(debug_info['transmission_bcp'], cmap='gray')
    axes[1, 2].set_title('Transmission BCP')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(debug_info['transmission_fused'], cmap='gray')
    axes[1, 3].set_title('Transmission Fusionado')
    axes[1, 3].axis('off')
    
    # Transmission refinado
    axes[2, 0].imshow(debug_info['transmission_refined'], cmap='gray')
    axes[2, 0].set_title('Transmission Refinado')
    axes[2, 0].axis('off')
    
    # Comparação lado a lado
    comparison = np.hstack([original, dehazed])
    axes[2, 1].imshow(comparison)
    axes[2, 1].set_title('Comparação: Original | Dehazed')
    axes[2, 1].axis('off')
    
    # Informações de luz atmosférica
    axes[2, 2].text(0.1, 0.8, f'Luz Atmosférica DCP:\nR:{debug_info["atmospheric_light_dcp"][0]:.2f}\nG:{debug_info["atmospheric_light_dcp"][1]:.2f}\nB:{debug_info["atmospheric_light_dcp"][2]:.2f}', 
                    transform=axes[2, 2].transAxes, fontsize=10, verticalalignment='top')
    axes[2, 2].text(0.1, 0.4, f'Luz Atmosférica BCP:\nR:{debug_info["atmospheric_light_bcp"][0]:.2f}\nG:{debug_info["atmospheric_light_bcp"][1]:.2f}\nB:{debug_info["atmospheric_light_bcp"][2]:.2f}', 
                    transform=axes[2, 2].transAxes, fontsize=10, verticalalignment='top')
    axes[2, 2].set_title('Parâmetros de Luz')
    axes[2, 2].axis('off')
    
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

# Exemplo de uso
def example_usage():
    """Exemplo de como usar o algoritmo"""
    
    # Criar uma imagem sintética com névoa para teste
    def create_synthetic_hazy_image():
        # Criar imagem base
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Céu (região superior)
        img[0:100, :] = [200, 220, 255]  # Céu azul claro
        
        # Paisagem (região inferior)
        img[100:200, :] = [100, 150, 100]  # Verde
        img[200:, :] = [80, 120, 80]  # Verde mais escuro
        
        # Adicionar alguns objetos
        cv2.rectangle(img, (50, 120), (150, 200), (139, 69, 19), -1)  # Prédio marrom
        cv2.rectangle(img, (250, 130), (350, 180), (139, 69, 19), -1)  # Outro prédio
        
        # Simular névoa
        haze = np.ones_like(img) * 180  # Cor da névoa
        alpha = 0.4  # Intensidade da névoa
        hazy_img = cv2.addWeighted(img, 1-alpha, haze, alpha, 0)
        
        return hazy_img.astype(np.uint8)
    
    # Criar imagem com névoa
    hazy_image = create_synthetic_hazy_image()
    
    # Inicializar dehazer
    dehazer = ImageDehazer(omega=0.95, k_adjustment=0.1, patch_size=15)
    
    # Processar imagem
    print("Iniciando processo de dehazing...")
    dehazed_image, debug_info = dehazer.dehaze(hazy_image)
    
    # Visualizar resultados
    visualize_results(hazy_image, dehazed_image, debug_info)
    
    return hazy_image, dehazed_image, debug_info

if __name__ == "__main__":
    # Executar exemplo
    original, result, info = example_usage()
    print("Dehazing concluído com sucesso!")