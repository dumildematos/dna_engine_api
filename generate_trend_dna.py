import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from fashion_clip.fashion_clip import FashionCLIP

def generate_trend_dna(trend_folder="data/trends/street_2026_spring"):
    """
    Encodes a collection of trend images into a single 'Trend DNA' vector.
    """
    print(f"📡 EXTRACTING TREND DNA FROM: {trend_folder}")
    
    # 1. Initialize FashionCLIP
    fclip = FashionCLIP("fashion-clip")
    
    image_paths = [os.path.join(trend_folder, f) for f in os.listdir(trend_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print("⚠️ No images found in trend folder.")
        return None

    # 2. Process images in batches
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]
    embeddings = fclip.encode_images(pil_images, batch_size=16)

    # 3. Calculate the Trend Centroid
    # This represents the 'average' aesthetic of the current trend
    trend_dna = np.mean(embeddings, axis=0)
    trend_dna /= np.linalg.norm(trend_dna)

    # 4. Save as a dynamic DNA file
    os.makedirs("data/centroids/trends", exist_ok=True)
    trend_name = os.path.basename(trend_folder)
    np.save(f"data/centroids/trends/{trend_name}.npy", trend_dna)
    
    print(f"✅ Trend DNA Saved: {trend_name}.npy")
    return trend_dna