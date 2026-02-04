import os
from dotenv import load_dotenv
load_dotenv()
import torch
import numpy as np
from PIL import Image
import io
import fashion_clip.fashion_clip as fc_module
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

# Global model settings
MODEL_ID = "runwayml/stable-diffusion-v1-5" 
pipe = None

# --- 2026 SUPREME COMPATIBILITY PATCH ---
def patched_load_model(self, name, auth_token=None):
    from transformers import CLIPModel, CLIPProcessor
    actual_repo = "patrickjohncyh/fashion-clip"
    print(f"--- Brain: Fetching weights from {actual_repo} ---")
    # Force return_dict=True for modern transformers
    model = CLIPModel.from_pretrained(actual_repo, token=auth_token, return_dict=True)
    preprocess = CLIPProcessor.from_pretrained(actual_repo, token=auth_token)
    return model, preprocess, None

def patched_encode_images(self, images, batch_size=16):
    """Handles the modern 'BaseModelOutputWithPooling' object error."""
    image_embeddings = []
    self.model.eval()
    for i in tqdm(range(0, len(images), batch_size), desc="Brain Encoding"):
        batch_images = images[i : i + batch_size]
        inputs = self.preprocess(images=batch_images, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            # FIX: Extract raw tensor from the container object
            if not isinstance(outputs, torch.Tensor):
                features = getattr(outputs, "image_embeds", getattr(outputs, "pooler_output", outputs[0]))
            else:
                features = outputs
            image_embeddings.extend(features.detach().cpu().numpy())
    return np.array(image_embeddings)

# Apply patches before initializing FashionCLIP
fc_module.FashionCLIP._load_model = patched_load_model
fc_module.FashionCLIP.encode_images = patched_encode_images

from fashion_clip.fashion_clip import FashionCLIP
# --------------------------------

class StyleBrain:
    def __init__(self):
        # FIX: Corrected the CUDA check
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fclip = None

    def load_models(self):
        print(f"--- Loading Brain to {self.device} ---")
        self.fclip = FashionCLIP('fashion-clip')
        self.load_generator()
        print("--- Brain is Online ---")

    def breed_dna(self, dna_a: np.ndarray, dna_b: np.ndarray, ratio: float):
        ratio = max(0.0, min(1.0, ratio))
        blended_vector = (1 - ratio) * dna_a + ratio * dna_b
        norm = np.linalg.norm(blended_vector)
        return blended_vector / norm if norm > 0 else blended_vector

    def load_generator(self):
        global pipe
        if pipe is None:
            print(f"--- Loading Stable Diffusion Engine to {self.device}... ---")
            # FIX: Use float16 for CUDA, float32 for CPU to stop warnings
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID, 
                torch_dtype=dtype,
                # Disable safety checker on CPU for faster loads if needed
                safety_checker=None if self.device == "cpu" else "default"
            ).to(self.device)
            print("--- Engine Ready! ---")
        return pipe

    def generate_design(self, bred_dna: np.ndarray):
        generator = self.load_generator()
        
        # Use DNA sum as a reproducible seed
        seed = int(np.abs(bred_dna.sum()) * 1000000) % 2**32
        latents_generator = torch.Generator(device=self.device).manual_seed(seed)
        
        prompt = "high fashion runway look, avant-garde garment, detailed texture, professional photography, masterpiece"
        
        # In 2026, we inject the DNA "vibe" via the seed
        image = generator(
            prompt, 
            num_inference_steps=20 if self.device == "cpu" else 25, 
            guidance_scale=7.5,
            generator=latents_generator
        ).images[0]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr

# Singleton instance
brain = StyleBrain()