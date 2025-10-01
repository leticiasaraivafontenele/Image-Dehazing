import numpy as np
from scipy.ndimage import minimum_filter
import cv2

def guided_filter(I, p, r, eps):
    # Handle color guidance image
    if len(I.shape) == 3:
        # Convert to grayscale for guidance
        I = np.mean(I, axis=2)
    
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    
    q = mean_a * I + mean_b
    return q

def dark_channel(image, patch_size=15):
    # Get minimum across color channels
    min_channels = np.min(image, axis=2)
    
    # Apply minimum filter in local patch
    dark = minimum_filter(min_channels, size=patch_size)
    
    return dark

def estimate_atmospheric_light(image, dark_channel_img, top_percent=0.001):

    h, w = dark_channel_img.shape
    num_pixels = h * w
    num_brightest = int(max(num_pixels * top_percent, 1))
    
    # Get indices of brightest pixels in dark channel
    dark_flat = dark_channel_img.flatten()
    indices = np.argsort(dark_flat)[-num_brightest:]
    
    # Get corresponding pixels from original image
    image_flat = image.reshape(-1, 3)
    brightest_pixels = image_flat[indices]
    
    # Average the RGB values
    atmospheric_light = np.mean(brightest_pixels, axis=0)
    
    return atmospheric_light

def transmission_map(image, atmospheric_light, omega=0.95, patch_size=15):
    # Normalize image by atmospheric light
    normalized_image = np.zeros_like(image)
    for c in range(3):
        normalized_image[:, :, c] = image[:, :, c] / atmospheric_light[c]
    
    # Compute dark channel of normalized image
    dark_normalized = dark_channel(normalized_image, patch_size)
    
    # Calculate transmission map
    transmission = 1 - omega * dark_normalized
    
    return transmission

def compute_dcp_transmission(image_input, omega=0.95, patch_size=15, top_percent=0.001, 
                            refine=False, guided_r=60, guided_eps=0.0001):
    # Handle different input types
    if isinstance(image_input, str):
        # Read image from path
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError(f"Could not read image from {image_input}")
        # Convert BGR to RGB and normalize to [0, 1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    elif isinstance(image_input, np.ndarray):
        # Use numpy array directly
        image = image_input.astype(np.float64)
        # Normalize if needed (check if values are in [0, 255] range)
        if image.max() > 1.0:
            image = image / 255.0
    else:
        raise ValueError("image_input must be either a file path (string) or numpy array")
    
    # Step 1: Compute dark channel
    print("Computing dark channel...")
    dark = dark_channel(image, patch_size)
    
    # Step 2: Estimate atmospheric light
    print("Estimating atmospheric light...")
    A = estimate_atmospheric_light(image, dark, top_percent)
    print(f"Atmospheric light (RGB): {A}")
    
    # Step 3: Compute transmission map
    print("Computing transmission map...")
    t = transmission_map(image, A, omega, patch_size)
    
    # Step 4: Refine transmission map with guided filter
    if refine:
        print("Refining transmission map with guided filter...")
        t_refined = guided_filter(image, t, guided_r, guided_eps)
        return t_refined, A, image
    
    return t, A, image

# Example usage
if __name__ == "__main__":
    
    try:
        # METHOD 1: Pass image file path as string
        image_path = r"D:\dataset\SOTS\outdoor\hazy\0047_0.9_0.12.jpg"
        transmission, atmospheric_light, original_image = compute_dcp_transmission(
            image_path,  # <-- Pass your image path here
            omega=0.95, 
            patch_size=15
        )
        print(f"Transmission map shape: {transmission.shape}")
        print(f"Transmission range: [{transmission.min():.3f}, {transmission.max():.3f}]")
        
        # Optionally save the transmission map
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(transmission, cmap='gray')
        axes[1].set_title('Transmission Map')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('transmission_map.png', dpi=150, bbox_inches='tight')
        print("Saved transmission map visualization to 'transmission_map.png'")
        
    except FileNotFoundError:
        print("Example image not found. Please provide your own image path.")
        print("\nTo use this code:")
        print("1. Replace 'hazy_image.jpg' with your image path")
        print("2. Or use the functions directly in your code:")
        print("   t, A, img = compute_dcp_transmission('your_image.jpg')")