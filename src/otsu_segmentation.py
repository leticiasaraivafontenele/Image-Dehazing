import cv2
import numpy as np

def fitness_otsu(t, P):
    """Calcula σ² (fitness) para um threshold t"""
    t = int(np.clip(t, 0, 254))
    
    # Calcular w1 e w2
    w1 = np.sum(P[0:t+1])
    w2 = np.sum(P[t+1:256])
    
    if w1 == 0 or w2 == 0:
        return 0
    
    # Calcular u1 e u2
    u1 = np.sum(np.arange(t+1) * P[0:t+1]) / w1
    u2 = np.sum(np.arange(t+1, 256) * P[t+1:256]) / w2
    
    # Calcular média global
    u = np.sum(np.arange(256) * P)
    
    # Calcular σ²
    sigma2 = w1 * (u1 - u)**2 + w2 * (u2 - u)**2
    
    return sigma2

def otsu_pso(image_path, n_particles=500, n_iterations=400):
    # Carregar imagem e calcular P(x)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    NUM = img.size
    count, _ = np.histogram(img, bins=256, range=(0, 256))
    P = count / NUM
    
    # Inicializar partículas aleatoriamente entre 0 e 254
    particles = np.random.uniform(0, 254, n_particles)
    velocities = np.random.uniform(-10, 10, n_particles)
    
    # Melhor posição de cada partícula
    best_positions = particles.copy()
    best_fitness = np.array([fitness_otsu(p, P) for p in particles])
    
    # Melhor global
    global_best_idx = np.argmax(best_fitness)
    global_best_position = best_positions[global_best_idx]
    global_best_fitness = best_fitness[global_best_idx]
    
    # Parâmetros do PSO
    w = 0.7  # inércia
    c1 = 1.5  # coeficiente cognitivo
    c2 = 1.5  # coeficiente social
    
    # Iterações do PSO
    for _ in range(n_iterations):
        for i in range(n_particles):
            # Atualizar velocidade
            r1, r2 = np.random.random(), np.random.random()
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (best_positions[i] - particles[i]) +
                           c2 * r2 * (global_best_position - particles[i]))
            
            # Atualizar posição
            particles[i] = np.clip(particles[i] + velocities[i], 0, 254)
            
            # Avaliar fitness
            current_fitness = fitness_otsu(particles[i], P)
            
            # Atualizar melhor pessoal
            if current_fitness > best_fitness[i]:
                best_fitness[i] = current_fitness
                best_positions[i] = particles[i]
                
                # Atualizar melhor global
                if current_fitness > global_best_fitness:
                    global_best_fitness = current_fitness
                    global_best_position = particles[i]
    
    return int(global_best_position), img

def limpar_segmentacao_ceu(imagem_binaria):
    """
    Remove pixels brancos isolados e mantém apenas a maior região conectada (céu).
    
    Args:
        imagem_binaria: Imagem já segmentada (preto e branco)
        
    Returns:
        Imagem limpa com apenas a região do céu
    """
    # Encontrar todos os componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        imagem_binaria, connectivity=8
    )
    
    # stats: [x, y, width, height, area]
    # Label 0 é o fundo (preto), começamos do 1
    
    # Encontrar a maior componente (excluindo o fundo)
    if num_labels <= 1:
        return imagem_binaria  # Nenhuma região branca encontrada
    
    areas = stats[1:, cv2.CC_STAT_AREA]  # Áreas de todas as componentes (exceto fundo)
    maior_componente = np.argmax(areas) + 1  # +1 porque pulamos o label 0
    
    # Criar imagem limpa mantendo apenas a maior componente
    imagem_limpa = np.zeros_like(imagem_binaria)
    imagem_limpa[labels == maior_componente] = 255
    
    return imagem_limpa


def otsu_segmentation(image_path, threshold=None):
    """
    Pipeline completo: carregar → segmentar → limpar
    """
    # Carregar imagem

    img = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
    
    # Se não forneceu threshold, usa Otsu
    if threshold is None:
        threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = threshold  # cv2.threshold retorna (valor, imagem)
    
    # Aplicar threshold
    _, segmentada = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    # Limpar ruído
    segmentada_limpa = limpar_segmentacao_ceu(segmentada)
    
    return segmentada_limpa, threshold


