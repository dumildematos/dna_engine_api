import os
import json
import numpy as np
import requests
import torch
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

# Import the shared ImageKit config
from config.imagekit_config import imagekit
import fashion_clip.fashion_clip as fc_module

load_dotenv()

# --- 2026 COMPATIBILITY PATCH ---
def patched_load_model(self, name, auth_token=None):
    from transformers import CLIPModel, CLIPProcessor
    actual_repo = "patrickjohncyh/fashion-clip"
    model = CLIPModel.from_pretrained(actual_repo, token=auth_token, return_dict=True)
    preprocess = CLIPProcessor.from_pretrained(actual_repo, token=auth_token)
    return model, preprocess, None

def patched_encode_images(self, images, batch_size=16):
    image_embeddings = []
    self.model.eval()
    for i in tqdm(range(0, len(images), batch_size), desc="Encoding DNA"):
        batch_images = images[i : i + batch_size]
        inputs = self.preprocess(images=batch_images, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            features = getattr(outputs, "image_embeds", getattr(outputs, "pooler_output", outputs))
            if not isinstance(features, torch.Tensor): features = features[0]
            image_embeddings.extend(features.detach().cpu().numpy())
    return np.array(image_embeddings)

# --- 2026 SUPREME COMPATIBILITY PATCH: TEXT ENCODING ---
def patched_encode_text(self, text, batch_size=16):
    """Fixes AttributeError by unwrapping BaseModelOutputWithPooling for sliders."""
    text_embeddings = []
    self.model.eval()
    
    if isinstance(text, str):
        text = [text]
    
    for i in range(0, len(text), batch_size):
        batch_text = text[i : i + batch_size]
        inputs = self.preprocess(text=batch_text, return_tensors="pt", padding=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            
            # UNWRAP LOGIC: Extract the actual tensor from the Output object
            if hasattr(outputs, "text_embeds"):
                features = outputs.text_embeds
            elif hasattr(outputs, "pooler_output"):
                features = outputs.pooler_output
            elif isinstance(outputs, torch.Tensor):
                features = outputs
            else:
                features = outputs[0]

            # Normalize and move to CPU
            norm = features.norm(p=2, dim=-1, keepdim=True)
            features = features / (norm + 1e-8)
            text_embeddings.extend(features.detach().cpu().numpy())
            
    return np.array(text_embeddings)

# Apply patches to the FashionCLIP class
fc_module.FashionCLIP.encode_text = patched_encode_text
fc_module.FashionCLIP._load_model = patched_load_model
fc_module.FashionCLIP.encode_images = patched_encode_images

from fashion_clip.fashion_clip import FashionCLIP

# --- ARCHIVIST VOCABULARY ---
FASHION_TRAITS = [
    "quilted texture", "minimalist lines", "avant-garde silhouette",
    "industrial nylon", "architectural tailoring", "organic curves",
    "monogram print", "tweed fabric", "distressed leather",
    "structured shoulders", "flowing silk", "utility hardware"
]

def get_remote_image_urls(folder_path):
    print(f"🔍 Checking ImageKit folder: {folder_path}...")
    try:
        # v4 SDK uses .assets.list
        response = imagekit.assets.list(path=folder_path, file_type="image")
        if not response: return []
        print(f"✅ Found {len(response)} images.")
        return [f"{f.url}?tr=w-512,h-512,cm-pad_resize" for f in response]
    except Exception as e:
        print(f"❌ SDK Error: {e}")
        return []

def extract_brand_dna_cloud(brand_name):
    print(f"🧬 ARCHIVING BRAND: {brand_name.upper()}")
    
    urls = get_remote_image_urls(f"/{brand_name}/")
    if not urls: return

    pil_images = []
    for url in urls:
        try:
            res = requests.get(url, timeout=10)
            pil_images.append(Image.open(BytesIO(res.content)).convert("RGB"))
        except: continue

    # 1. Initialize FashionCLIP
    fclip = FashionCLIP("fashion-clip")
    
    # 2. Extract Individual Image Embeddings
    # Shape: (Number of Images, 512)
    embeddings = fclip.encode_images(pil_images, batch_size=16)
    
    # Calculate the Centroid (The average DNA for breeding)
    brand_dna = np.mean(embeddings, axis=0)
    brand_dna /= np.linalg.norm(brand_dna)

    # 3. Extract Heritage Traits (The "AI Archivist" Logic)
    # Shape: (Number of Traits, 512)
    text_embeddings = fclip.encode_text(FASHION_TRAITS, batch_size=32)

    # --- CRITICAL ALIGNMENT FIX ---
    # We want to know how much each image (N, 512) matches each trait (M, 512)
    # Resulting similarity shape: (N images, M traits)
    similarity = np.matmul(embeddings, text_embeddings.T)

    # Average the similarities across all images to find brand-wide strengths
    avg_scores = similarity.mean(axis=0)
    
    # Rank and pick the top 3
    top_indices = avg_scores.argsort()[-3:][::-1]
    dominant_traits = [FASHION_TRAITS[i] for i in top_indices]

    # 4. Save DNA (.npy)
    os.makedirs("data/centroids", exist_ok=True)
    np.save(f"data/centroids/{brand_name}.npy", brand_dna)

    # 5. Save Heritage Metadata (.json)
    heritage_data = {
        "brand": brand_name,
        "traits": dominant_traits,
        "image_count": len(pil_images),
        "archived_at": "2026-02-05"
    }
    with open(f"data/centroids/{brand_name}.json", "w") as f:
        json.dump(heritage_data, f, indent=4)

    print(f"✅ DNA Saved. Traits identified: {', '.join(dominant_traits)}\n")

if __name__ == "__main__":
    brands = ["valentino","rick_owens","dior","off_white","hermes","gucci","chanel","balenciaga","versace","prada"]
    for b in brands:
        extract_brand_dna_cloud(b)