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

# --- New Imports for Visualization ---
import matplotlib
matplotlib.use('Agg')  # Required for headless servers/FastAPI
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# -------------------------------------

# Global model settings
MODEL_ID = "runwayml/stable-diffusion-v1-5" 
pipe = None

# --- 2026 SUPREME COMPATIBILITY PATCH ---
# --- 2026 SUPREME COMPATIBILITY PATCHES ---

def patched_load_model(self, name, auth_token=None):
    from transformers import CLIPModel, CLIPProcessor
    import torch
    
    # The fashion-specific weights
    actual_repo = "patrickjohncyh/fashion-clip"
    # The standard config files needed to satisfy Python 3.13 / Transformers 4.40+
    processor_repo = "openai/clip-vit-base-patch32"
    
    print(f"--- Brain: Fetching weights from {actual_repo} ---")
    
    # Load Model from Fashion-CLIP
    model = CLIPModel.from_pretrained(actual_repo, token=auth_token, return_dict=True)
    
    # Load Processor from OpenAI (This fixes the OSError / missing .json)
    preprocess = CLIPProcessor.from_pretrained(processor_repo)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return model, preprocess, "stable_2026_hash"

def patched_encode_images(self, images, batch_size=16):
    """Encodes images and ensures vectors are normalized for DNA breeding."""
    image_embeddings = []
    self.model.eval()
    
    for i in tqdm(range(0, len(images), batch_size), desc="Brain Encoding"):
        batch_images = images[i : i + batch_size]
        inputs = self.preprocess(images=batch_images, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            
            # Handle the BaseModelOutput object vs raw Tensor
            if not isinstance(outputs, torch.Tensor):
                features = getattr(outputs, "image_embeds", getattr(outputs, "pooler_output", outputs[0]))
            else:
                features = outputs
            
            # CRITICAL: Normalize vectors so cosine similarity and breeding math work
            norm = features.norm(p=2, dim=-1, keepdim=True)
            features = features / (norm + 1e-8)
            
            image_embeddings.extend(features.detach().cpu().numpy())
            
    return np.array(image_embeddings)

# Apply the overrides to the library
fc_module.FashionCLIP._load_model = patched_load_model
fc_module.FashionCLIP.encode_images = patched_encode_images
from fashion_clip.fashion_clip import FashionCLIP

# --------------------------------

class StyleBrain:
    def __init__(self):
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

    def generate_breeding_map(self, brand_a: str, brand_b: str, ratio: float, filename: str):
        centroid_dir = "data/centroids"
        dna_vectors = []
        brand_names = []

        if not os.path.exists(centroid_dir):
            return io.BytesIO()

        for file in os.listdir(centroid_dir):
            if file.endswith(".npy") and not file.startswith("trend"):
                dna_vectors.append(np.load(os.path.join(centroid_dir, file)))
                brand_names.append(file.replace(".npy", ""))

        data = np.array(dna_vectors)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(data)
        brand_map = dict(zip(brand_names, coords))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(coords[:, 0], coords[:, 1], c='lightgrey', alpha=0.5, label="Other Brands")

        if brand_a in brand_map and brand_b in brand_map:
            start, end = brand_map[brand_a], brand_map[brand_b]
            blend_coord = (1 - ratio) * start + ratio * end
            ax.plot([start[0], end[0]], [start[1], end[1]], 'r--', alpha=0.6, label="Breeding Path")
            ax.scatter(blend_coord[0], blend_coord[1], color='gold', s=200, marker='*', edgecolors='k', label="New DNA")
            ax.annotate(brand_a.upper(), (start[0], start[1]), xytext=(5,5), textcoords='offset points', fontsize=8)
            ax.annotate(brand_b.upper(), (end[0], end[1]), xytext=(5,5), textcoords='offset points', fontsize=8)

        ax.set_title(f"Style Space: {brand_a} x {brand_b}")
        ax.legend()
        
        output_dir = "data/history"
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight')

        map_buf = io.BytesIO()
        fig.savefig(map_buf, format='png')
        map_buf.seek(0)
        plt.close(fig) 
        return map_buf

    def load_generator(self):
        global pipe
        if pipe is None:
            print(f"--- Loading Stable Diffusion Engine ---")
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID, 
                torch_dtype=dtype,
                safety_checker=None
            ).to(self.device)
            print("--- Engine Ready! ---")
        return pipe

    def generate_design(self, bred_dna: np.ndarray, custom_prompt: str = "", layout_type: str = "hero"):
        """
        Generates a design based on DNA. 
        layout_type: 'hero' for the main garment, 'flatlay' for accessory presentation.
        """
        generator = self.load_generator()
        
        # 1. Deterministic Seed from DNA
        seed = int(np.abs(bred_dna.sum()) * 1000000) % 2**32
        latents_generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 2. Dynamic Layout Styles
        # 'flatlay' mimics your inspiration image: organized clothes, accessories, beige background.
        if layout_type == "flatlay":
            layout_style = (
                "organized flat lay photography, minimalist knolling composition, "
                "garment with matching watch and belt, accessories on beige background, "
                "soft studio lighting, top-down view"
            )
        else:
            layout_style = (
                "high fashion masterpiece, avant-garde garment silhouette, "
                "8k resolution, cinematic lighting, sharp focus, professional photography"
            )

        # 3. Prompt Construction
        # Ensure heritage traits (custom_prompt) are at the very front for maximum weight
        final_prompt = f"{custom_prompt}, {layout_style}" if custom_prompt else layout_style
        
        print(f"🎨 Lab Generating [{layout_type}]: {final_prompt}")

        # 4. Image Generation
        image = generator(
            final_prompt, 
            num_inference_steps=25, 
            guidance_scale=8.5, 
            generator=latents_generator
        ).images[0]

        # 5. Export to Buffer
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr
    
    def verify_design(self, generated_image, target_dna: np.ndarray):
        gen_embedding = self.fclip.encode_images([generated_image], batch_size=1)[0]
        gen_norm = gen_embedding / np.linalg.norm(gen_embedding)
        target_norm = target_dna / np.linalg.norm(target_dna)
        return float(np.dot(gen_norm, target_norm))

    def apply_style_slider(self, dna_vector: np.ndarray, slider_name: str, intensity: float):
        aesthetic_library = {
            "minimalism": ("clean minimalist simple luxury silhouette", "ornate maximalist complex busy details"),
            "vintage_futurism": ("futuristic sci-fi cyber techwear", "vintage retro historical heritage look")
        }

        if slider_name not in aesthetic_library:
            return dna_vector

        pos_text, neg_text = aesthetic_library[slider_name]
        
        with torch.no_grad():
            # FIX: Added batch_size=1 to avoid TypeError
            pos_v = self.fclip.encode_text([pos_text], batch_size=1)[0]
            neg_v = self.fclip.encode_text([neg_text], batch_size=1)[0]
        
        direction = pos_v - neg_v
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        shifted_dna = dna_vector + (direction * intensity)
        return shifted_dna / np.linalg.norm(shifted_dna)

brain = StyleBrain()