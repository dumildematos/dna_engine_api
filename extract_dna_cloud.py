import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import fashion_clip.fashion_clip as fc_module
import torch
from tqdm import tqdm
from app.config.imagekit_config import imagekit

# --- 2026 COMPATIBILITY PATCH (Must match brain.py) ---
def patched_load_model(self, name, auth_token=None):
    from transformers import CLIPModel, CLIPProcessor
    actual_repo = "patrickjohncyh/fashion-clip"
    print(f"--- Patching: Loading weights from {actual_repo} ---")
    # Force return_dict=True to ensure we get a consistent object format
    model = CLIPModel.from_pretrained(actual_repo, token=auth_token, return_dict=True)
    preprocess = CLIPProcessor.from_pretrained(actual_repo, token=auth_token)
    return model, preprocess, None

def patched_encode_images(self, images, batch_size=16):
    """
    Manually extracts embeddings from modern Transformers output containers.
    """
    image_embeddings = []
    # Set model to eval mode for consistency
    self.model.eval()
    
    for i in tqdm(range(0, len(images), batch_size), desc="Encoding DNA"):
        batch_images = images[i : i + batch_size]
        # Preprocess and move to correct device (CPU or CUDA)
        inputs = self.preprocess(images=batch_images, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            
            # --- CRITICAL FIX START ---
            # 1. Try 'image_embeds' (standard for CLIPModel)
            if hasattr(outputs, "image_embeds"):
                features = outputs.image_embeds
            # 2. Try 'pooler_output' (standard for many vision models)
            elif hasattr(outputs, "pooler_output"):
                features = outputs.pooler_output
            # 3. Fallback to indexing if it's a tuple-like object
            elif isinstance(outputs, (list, tuple)):
                features = outputs[0]
            else:
                features = outputs
            # --- CRITICAL FIX END ---
                
            image_embeddings.extend(features.detach().cpu().numpy())
            
    return np.array(image_embeddings)
# Apply patch to the module before class instantiation
fc_module.FashionCLIP._load_model = patched_load_model
fc_module.FashionCLIP.encode_images = patched_encode_images
from fashion_clip.fashion_clip import FashionCLIP
# -----------------------------------------------------

def get_remote_image_urls(folder_path):
    """Lists images and verifies folder content using v4 SDK."""
    print(f"🔍 Checking ImageKit folder: {folder_path}...")
    
    try:
        # Correct v4 list method: imagekit.list_assets or imagekit.files.list
        response = imagekit.assets.list(
            path=folder_path,
            file_type="image")
    except Exception as e:
        print(f"⚠️ SDK Method issue: {e}. Trying fallback...")
        response = imagekit.assets.list(
            path=folder_path,
            file_type="image")

    if not response:
        print(f"❌ ERROR: No images found in '{folder_path}'.")
        return []

    print(f"✅ Found {len(response)} images in '{folder_path}'.")
    # Transform to 512x512 square for optimal Fashion-CLIP processing
    return [f"{f.url}?tr=w-512,h-512,cm-pad_resize" for f in response]

def extract_brand_dna_cloud(brand_name):
    print(f"🧬 Processing Brand: {brand_name.upper()}")
    
    # 1. Get URLs from ImageKit
    folder_path = f"/{brand_name}/"
    urls = get_remote_image_urls(folder_path)
    
    if not urls:
        return

    # 2. Download images into memory
    pil_images = []
    for url in urls:
        try:
            res = requests.get(url, timeout=10)
            img = Image.open(BytesIO(res.content)).convert("RGB")
            pil_images.append(img)
        except Exception as e:
            print(f"   ⚠️ Failed to load {url}: {e}")

    # 3. Calculate DNA with patched Fashion-CLIP
    # We pass 'fashion-clip' but our patch redirects it correctly
    fclip = FashionCLIP("fashion-clip")
    embeddings = fclip.encode_images(pil_images, batch_size=16)
    
    brand_dna = np.mean(embeddings, axis=0)
    brand_dna /= np.linalg.norm(brand_dna)

    # 4. Save the DNA locally
    output_path = f"data/centroids/{brand_name}.npy"
    os.makedirs("data/centroids", exist_ok=True)
    np.save(output_path, brand_dna)
    print(f"✅ DNA saved to {output_path}\n")

if __name__ == "__main__":
    # Ensure your ImageKit folders match these exactly (lowercase)
    brands = ["dior", "prada"]


    
    for b in brands:
        extract_brand_dna_cloud(b)