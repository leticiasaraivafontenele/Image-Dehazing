import cv2
import numpy as np

def fitness_otsu(t, P):
    t = int(np.clip(t, 0, 254))
    
    w1 = np.sum(P[0:t+1])
    w2 = np.sum(P[t+1:256])
    
    if w1 == 0 or w2 == 0:
        return 0
    
    u1 = np.sum(np.arange(t+1) * P[0:t+1]) / w1
    u2 = np.sum(np.arange(t+1, 256) * P[t+1:256]) / w2
    
    u = np.sum(np.arange(256) * P)
    
    sigma2 = w1 * (u1 - u)*2 + w2 * (u2 - u)*2
    
    return sigma2

def otsu_pso(img, n_particles=500, n_iterations=400):

    NUM = img.size
    count, _ = np.histogram(img, bins=256, range=(0, 256))
    P = count / NUM
    
    particles = np.random.uniform(0, 254, n_particles)
    velocities = np.random.uniform(-10, 10, n_particles)
    
    best_positions = particles.copy()
    best_fitness = np.array([fitness_otsu(p, P) for p in particles])
    
    global_best_idx = np.argmax(best_fitness)
    global_best_position = best_positions[global_best_idx]
    global_best_fitness = best_fitness[global_best_idx]
    
    w = 0.7  
    c1 = 1.5
    c2 = 1.5
    
    for _ in range(n_iterations):
        for i in range(n_particles):
            r1, r2 = np.random.random(), np.random.random()
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (best_positions[i] - particles[i]) +
                           c2 * r2 * (global_best_position - particles[i]))
            
            particles[i] = np.clip(particles[i] + velocities[i], 0, 254)
            
            current_fitness = fitness_otsu(particles[i], P)
            
            if current_fitness > best_fitness[i]:
                best_fitness[i] = current_fitness
                best_positions[i] = particles[i]
                
                if current_fitness > global_best_fitness:
                    global_best_fitness = current_fitness
                    global_best_position = particles[i]
    
    return int(global_best_position), img

def clear_segmentation_sky(imagem_binaria):

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        imagem_binaria, connectivity=8
    )
    

    if num_labels <= 1:
        return imagem_binaria
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    maior_componente = np.argmax(areas) + 1
    
    imagem_limpa = np.zeros_like(imagem_binaria)
    imagem_limpa[labels == maior_componente] = 255
    
    return imagem_limpa


def otsu_segmentation(image_path, threshold=None):

    img = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)

    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    if threshold is None:
            threshold, _ = otsu_pso(img_gray)
    
    _, segmentada = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    segmentada_limpa = clear_segmentation_sky(segmentada)
    
    return segmentada_limpa, threshold