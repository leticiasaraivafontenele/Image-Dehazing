import cv2
from single_image_dehazing import single_image_dehazing
from visualize_results import visualize_results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = r"C:\\Users\\User\\Documents\\PDI\\SOTS\\outdoor\\hazy\\0003_0.8_0.2.jpg"
    
    print(f"Processando: {img_path}\n")
    
    J, resultados = single_image_dehazing(
        img_path=img_path,
        window_size=15,
        k=0.1,
        omega=0.95
    )
    
    visualize_results(resultados)
    
    cv2.imwrite('imagem_sem_neblina_final.jpg', cv2.cvtColor(J, cv2.COLOR_RGB2BGR))
    print("Imagem final salva: imagem_sem_neblina_final.jpg")