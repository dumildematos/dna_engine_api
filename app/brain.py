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
def patched_load_model(self, name, auth_token=None):
    from transformers import CLIPModel, CLIPProcessor
    actual_repo = "patrickjohncyh/fashion-clip"
    print(f"--- Brain: Fetching weights from {actual_repo} ---")
    model = CLIPModel.from_pretrained(actual_repo, token=auth_token, return_dict=True)
    preprocess = CLIPProcessor.from_pretrained(actual_repo, token=auth_token)
    return model, preprocess, None

def patched_encode_images(self, images, batch_size=16):
    image_embeddings = []
    self.model.eval()
    for i in tqdm(range(0, len(images), batch_size), desc="Brain Encoding"):
        batch_images = images[i : i + batch_size]
        inputs = self.preprocess(images=batch_images, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            if not isinstance(outputs, torch.Tensor):
                features = getattr(outputs, "image_embeds", getattr(outputs, "pooler_output", outputs[0]))
            else:
                features = outputs
            image_embeddings.extend(features.detach().cpu().numpy())
    return np.array(image_embeddings)

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
        """Generates a 2D PCA map and saves the breeding path."""
        centroid_dir = "data/centroids"
        dna_vectors = []
        brand_names = []

        # 1. Load all available centroids
        for file in os.listdir(centroid_dir):
            if file.endswith(".npy"):
                dna_vectors.append(np.load(os.path.join(centroid_dir, file)))
                brand_names.append(file.replace(".npy", ""))

        data = np.array(dna_vectors)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(data)
        brand_map = dict(zip(brand_names, coords))

        # 2. Create Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(coords[:, 0], coords[:, 1], c='lightgrey', alpha=0.5, label="Other Brands")

        # 3. Calculate and Plot the Path
        if brand_a in brand_map and brand_b in brand_map:
            start, end = brand_map[brand_a], brand_map[brand_b]
            blend_coord = (1 - ratio) * start + ratio * end
            
            ax.plot([start[0], end[0]], [start[1], end[1]], 'r--', alpha=0.6, label="Breeding Path")
            ax.scatter(blend_coord[0], blend_coord[1], color='gold', s=200, marker='*', edgecolors='k', label="New DNA")
            
            # Label the parents
            ax.annotate(brand_a.upper(), (start[0], start[1]), xytext=(5,5), textcoords='offset points', fontsize=8)
            ax.annotate(brand_b.upper(), (end[0], end[1]), xytext=(5,5), textcoords='offset points', fontsize=8)

        ax.set_title(f"Style Space: {brand_a} x {brand_b}")
        ax.legend()
        
        # 4. Save to Disk
        output_dir = "data/history"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        fig.savefig(file_path, bbox_inches='tight')

        # 5. Return as Buffer for API
        map_buf = io.BytesIO()
        fig.savefig(map_buf, format='png')
        map_buf.seek(0)
        plt.close(fig) # Memory cleanup
        return map_buf

    def load_generator(self):
        global pipe
        if pipe is None:
            print(f"--- Loading Stable Diffusion Engine ---")
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID, 
                torch_dtype=dtype,
                safety_checker=None if self.device == "cpu" else "default"
            ).to(self.device)
            print("--- Engine Ready! ---")
        return pipe

    def generate_design(self, bred_dna: np.ndarray, custom_prompt: str = ""):
        generator = self.load_generator()
        
        # 1. Deterministic Seed from DNA
        # This ensures the 'DNA' actually dictates the core structure
        seed = int(np.abs(bred_dna.sum()) * 1000000) % 2**32
        latents_generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 2. Archivist Prompt Integration
        # We combine your high-quality base prompt with the specific heritage traits
        base_style = "high fashion runway look, avant-garde garment, detailed texture, professional photography, masterpiece"
        
        if custom_prompt:
            # Heritage traits take priority at the front of the prompt
            final_prompt = f"{custom_prompt}, {base_style}"
        else:
            final_prompt = base_style
        
        print(f"🎨 Lab Generating: {final_prompt}")

        # 3. Image Generation
        image = generator(
            final_prompt, 
            num_inference_steps=20 if self.device == "cpu" else 25, 
            guidance_scale=8.0, # Slightly higher to respect the traits more
            generator=latents_generator
        ).images[0]

        # 4. Memory-Buffer Export
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr
    
    def verify_design(self, generated_image, target_dna: np.ndarray):
        """
        Calculates the 'Authenticity Score' of a generated design.
        """
        # 1. Convert image to embedding
        # We use the same CLIP model to see what the AI 'sees' in the result
        gen_embedding = self.fclip.encode_images([generated_image], batch_size=1)[0]
        
        # 2. Calculate Cosine Similarity
        # Normalize vectors
        gen_norm = gen_embedding / np.linalg.norm(gen_embedding)
        target_norm = target_dna / np.linalg.norm(target_dna)
        
        # Similarity score (0.0 to 1.0)
        similarity = np.dot(gen_norm, target_norm)
        
        # 3. Interpret the score
        # > 0.85: Extremely Authentic
        # > 0.70: Strong Hybrid
        # < 0.50: Style Hallucination (Discard)
        return float(similarity)

    def archive_brand_traits(self, images: list):
        """
        Interrogates a batch of brand images to find dominant visual traits.
        """
        # Define the 'Fashion Vocabulary' the archivist looks for
        traits_vocabulary = [
            "quilted texture", "minimalist lines", "avant-garde silhouette",
            "industrial nylon", "architectural tailoring", "organic curves",
            "monogram print", "tweed fabric", "distressed leather"
        ]
        
        # Use Fashion-CLIP to rank these traits against the images
        # This returns the probability of each text trait matching the images
        text_embeddings = self.fclip.encode_text(traits_vocabulary)
        image_embeddings = self.fclip.encode_images(images)
        
        # Calculate similarity
        similarity = image_embeddings @ text_embeddings.T
        avg_scores = similarity.mean(axis=0)
        
        # Get the top 3 traits
        top_indices = avg_scores.argsort()[-3:][::-1]
        return [traits_vocabulary[i] for i in top_indices]
    
    def apply_style_slider(self, dna_vector: np.ndarray, slider_name: str, intensity: float):
        """
        Pushes the DNA vector along a specific aesthetic axis.
        intensity: -1.0 (Opposite) to 1.0 (Full Style)
        """
        # Define our library of 'Basis Vectors'
        aesthetic_library = {
            "minimalism": ("clean minimalist simple", "ornate maximalist complex"),
            "utility": ("techwear utility tactical hardware", "formal elegant soft"),
            "vintage": ("retro vintage 1950s heritage", "modern futuristic sci-fi")
        }

        if slider_name not in aesthetic_library:
            return dna_vector

        pos_text, neg_text = aesthetic_library[slider_name]
        
        # Encode the two poles
        with torch.no_grad():
            pos_v = self.fclip.encode_text([pos_text])[0]
            neg_v = self.fclip.encode_text([neg_text])[0]
        
        # Calculate the Direction Vector
        direction = pos_v - neg_v
        direction = direction / np.linalg.norm(direction) # Normalize
        
        # Apply the shift to the DNA
        # new_dna = original + (direction * slider_value)
        shifted_dna = dna_vector + (direction * intensity)
        
        return shifted_dna / np.linalg.norm(shifted_dna) # Re-normalize

# Singleton instance
brain = StyleBrain()