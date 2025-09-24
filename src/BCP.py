import cv2
import numpy as np

# --- Carregar a imagem RGB ---
img_rgb = cv2.imread("PDI\\0047_0.9_0.12.jpg")  # Substitua pelo caminho da sua imagem
img_rgb = img_rgb.astype(np.float32)

# --- Parâmetros ---
window_size = 15  # Tamanho da vizinhança Ω(q)
k = 0.1  # Fator de ajuste (entre 0.05 e 0.15)

# --- Passo 1: Bright Channel ---
# Para cada pixel, pega o máximo entre os canais R, G, B
max_rgb = np.max(img_rgb, axis=2)

# Aplicar o filtro máximo na vizinhança (dilatação) para simular o Ω(q)
kernel = np.ones((window_size, window_size), np.uint8)
bright_channel = cv2.dilate(max_rgb, kernel)

# --- Passo 2: Estimar a luz atmosférica ---
# Selecionar 0,1% pixels mais claros no bright channel
num_pixels = bright_channel.size
num_top = max(int(num_pixels * 0.001), 1)

# Índices dos pixels mais claros
indices = np.unravel_index(np.argsort(bright_channel.ravel())[-num_top:], bright_channel.shape)

# Luz atmosférica: média dos pixels correspondentes na imagem original
A_bright = np.mean(img_rgb[indices], axis=0)  # vetor [R, G, B]

# --- Passo 3: Mapa de transmissão inicial ---
t_bright = np.zeros_like(max_rgb)
for c in range(3):
    I_c = img_rgb[:, :, c]
    A_c = A_bright[c]
    b = np.abs(bright_channel - A_c)
    mean_b = np.mean(b)
    
    # Aplicar ajuste k
    t_bright_c = np.where(b < mean_b, (b + k) / (255 - A_c), b / (255 - A_c))
    
    # Limitar a transmissão máxima a 1
    t_bright_c = np.clip(t_bright_c, 0, 1)
    
    # Para simplificar, podemos pegar a média entre os canais
    t_bright += t_bright_c / 3

# --- Passo 4: Visualização ---
import matplotlib.pyplot as plt

plt.imshow(t_bright, cmap='gray')
plt.title("Mapa de Transmissão BCP (região do céu)")
plt.axis('off')
plt.show()