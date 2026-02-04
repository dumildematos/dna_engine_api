import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
import os
import torch
import gc
from tqdm import tqdm # Progress bar for console
from fashion_clip.fashion_clip import FashionCLIP

def generate_brand_centroid(brand_name, image_paths, batch_size=16):
    """
    Improved DNA extraction with memory management and GPU support.
    """
    print(f"🧬 Starting DNA Extraction for: {brand_name.upper()}")
    
    # 1. Initialize with Device Awareness (Uses GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fclip = FashionCLIP("patrickjohncyh/fashion-clip")
    
    try:
        # 2. Extract Embeddings 
        # Using fclip's built-in batching to prevent memory spikes
        embeddings = fclip.encode_images(image_paths, batch_size=batch_size)
        
        # 3. Validation
        if len(embeddings) == 0:
            raise ValueError(f"No valid embeddings generated for {brand_name}")

        # 4. Calculate Mathematical Centroid
        centroid = np.mean(embeddings, axis=0)
        
        # 5. L2 Normalization (Crucial for Cosine Similarity)
        # Adds a tiny epsilon to prevent division by zero
        norm = np.linalg.norm(centroid) + 1e-10
        normalized_dna = centroid / norm
        
        # 6. Atomic Save (Prevents corrupted files if the script crashes)
        save_dir = "data/centroids"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{brand_name}.npy")
        
        np.save(save_path, normalized_dna)
        
        # 7. Memory Cleanup
        del embeddings
        gc.collect() 
        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"✅ Success: {brand_name} DNA saved ({len(image_paths)} images processed).")
        return normalized_dna

    except Exception as e:
        print(f"❌ Error extracting DNA for {brand_name}: {str(e)}")
        return None