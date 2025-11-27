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
    
    u = w1 * u1 + w2 * u2
    
    sigma2 = w1 * (u1 - u)**2 + w2 * (u2 - u)**2
    
    return sigma2


def otsu_pso(img_gray, n_particles=30, n_iterations=100, verbose=False):

    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    NUM = img_gray.size
    P = hist / NUM
    
    particles = np.random.uniform(0, 254, n_particles)
    velocities = np.random.uniform(-10, 10, n_particles)
    
    best_positions = particles.copy()
    best_fitness = np.array([fitness_otsu(p, P) for p in particles])
    
    global_best_idx = np.argmax(best_fitness)
    global_best_position = best_positions[global_best_idx]
    global_best_fitness = best_fitness[global_best_idx]
    
    w_max, w_min = 0.9, 0.4 
    c1, c2 = 2.0, 2.0 
    
    for it in range(n_iterations):
        w = w_max - (w_max - w_min) * it / n_iterations 
        
        for i in range(n_particles):
            r1, r2 = np.random.random(), np.random.random()
            
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (best_positions[i] - particles[i]) +
                           c2 * r2 * (global_best_position - particles[i]))
            
            velocities[i] = np.clip(velocities[i], -20, 20)
            
            particles[i] = np.clip(particles[i] + velocities[i], 0, 254)
            
            current_fitness = fitness_otsu(particles[i], P)
            
            if current_fitness > best_fitness[i]:
                best_fitness[i] = current_fitness
                best_positions[i] = particles[i]
                
                if current_fitness > global_best_fitness:
                    global_best_fitness = current_fitness
                    global_best_position = particles[i]
        
        if it > 10 and it % 10 == 0:
            if abs(global_best_fitness - best_fitness.mean()) < 1e-6:
                if verbose:
                    print(f"✓ PSO convergiu na iteração {it}")
                break
    
    if verbose:
        print(f"✓ Threshold PSO: {int(global_best_position)} (σ²={global_best_fitness:.4f})")
    
    return int(global_best_position)


def clear_segmentation_sky(sky_mask):
    mask_uint8 = (sky_mask.astype(np.uint8) * 255)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )
    
    if num_labels <= 1:
        return sky_mask
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    maior_componente = np.argmax(areas) + 1
    
    sky_mask_clean = (labels == maior_componente)
    
    return sky_mask_clean


def otsu_segmentation(img_rgb, use_pso=True, postprocess=True, verbose=False):

    if img_rgb.dtype == np.float32 or img_rgb.dtype == np.float64:
        img_bgr = (img_rgb * 255).astype(np.uint8) if img_rgb.max() <= 1.0 else img_rgb.astype(np.uint8)
    else:
        img_bgr = img_rgb
    
    if len(img_bgr.shape) == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr

    if use_pso:
        t_opt = otsu_pso(img_gray, verbose=verbose)
    else:
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
        NUM = img_gray.size
        P = hist / NUM
        
        sigma2_list = []
        for t in range(256):
            sigma2_list.append(fitness_otsu(t, P))
        
        t_opt = np.argmax(sigma2_list)
        
        if verbose:
            print(f"✓ Threshold exaustivo: {t_opt} (σ²={sigma2_list[t_opt]:.4f})")
    
    sky_mask = img_gray > t_opt

    if postprocess:
        sky_mask = clear_segmentation_sky(sky_mask)
        if verbose:
            print("✓ Pós-processamento aplicado")
    
    return sky_mask, t_opt

