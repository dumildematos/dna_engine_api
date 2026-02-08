import os
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import transformers

# --- THE DEFINITIVE MONKEY PATCH ---
# 1. Fix the input (use_auth_token error)
original_from_pretrained = transformers.modeling_utils.PreTrainedModel.from_pretrained

@classmethod
def patched_from_pretrained(cls, *args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    return original_from_pretrained.__func__(cls, *args, **kwargs)

transformers.modeling_utils.PreTrainedModel.from_pretrained = patched_from_pretrained

# 2. Fix the output (BaseModelOutputWithPooling error)
# This forces the CLIP model to return the raw tensor during 'get_image_features'
from transformers.models.clip.modeling_clip import CLIPModel
original_get_image_features = CLIPModel.get_image_features

def patched_get_image_features(self, *args, **kwargs):
    output = original_get_image_features(self, *args, **kwargs)
    # If it's a complex object, extract the pooler_output (the actual DNA vector)
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    return output

CLIPModel.get_image_features = patched_get_image_features
# --- END PATCH ---

from fashion_clip.fashion_clip import FashionCLIP
from config.imagekit_config import imagekit

def generate_trend_dna_from_cloud(folder_path="trends/street_2026"):
    """
    Fetches raw trend images from ImageKit and encodes them into a DNA vector.
    """
    print(f"📡 SYNCING WITH IMAGEKIT: {folder_path}")
    
    # 1. List files in the specific ImageKit folder
    list_files = imagekit.assets.list(path=folder_path, file_type="image")

    if not len(list_files):
        print("⚠️ No images found in ImageKit folder.")
        return None

    # 2. Download and Convert to PIL
    pil_images = []
    for file in list_files:
        response = requests.get(file.url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        pil_images.append(img)
    
    # 3. Process with FashionCLIP
    fclip = FashionCLIP("fashion-clip")
    embeddings = fclip.encode_images(pil_images, batch_size=16)

    # 4. Calculate & Save Centroid
    trend_dna = np.mean(embeddings, axis=0)
    trend_dna /= np.linalg.norm(trend_dna)

    # Save locally for the API to serve
    trend_name = folder_path.split("/")[-1]
    save_path = f"data/centroids/trends/{trend_name}.npy"

    # --- ADD THIS LINE ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # ---------------------

    np.save(save_path, trend_dna)
    
    print(f"✅ Cloud Trend DNA Synchronized: {trend_name}.npy")
    return trend_dna

if __name__ == "__main__":
    generate_trend_dna_from_cloud("trends/street_2026")