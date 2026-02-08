import base64 as b64_module
import os
from PIL import Image
from typing import Dict, List
from dotenv import load_dotenv
load_dotenv()
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import numpy as np
import os
from .brain import brain
from config.imagekit_config import imagekit
import json

router = APIRouter()
CENTROID_DIR = "data/centroids"
TRENDS_DIR = "data/centroids/trends"
class BreedRequest(BaseModel):
    brand_a: str = Field(..., example="dior_vintage")
    brand_b: str = Field(..., example="mcqueen_gothic")
    # Enforce that the slider must be between 0 and 1
    mix_ratio: float = Field(..., ge=0.0, le=1.0, description="The blend ratio between Brand A and B"),
    minimalism: float
    futurism: float
    trend_id: str = None  # Optional
    trend_weight: float = 0.0

@router.post("/explore/breed")
async def breed_styles(request: BreedRequest):
    # 1. Setup Paths
    path_a = f"data/centroids/{request.brand_a}.npy"
    path_b = f"data/centroids/{request.brand_b}.npy"
    meta_a = f"data/centroids/{request.brand_a}.json"
    meta_b = f"data/centroids/{request.brand_b}.json"

    if not os.path.exists(path_a) or not os.path.exists(path_b):
        raise HTTPException(status_code=404, detail="Brand DNA files missing.")

    # 2. Load DNA & Heritage Traits
    dna_a, dna_b = np.load(path_a), np.load(path_b)
    
    traits = []
    for meta_path in [meta_a, meta_b]:
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                data = json.load(f)
                traits.extend(data.get("traits", []))
    
    # Clean and limit traits for the prompt
    heritage_prompt = ", ".join(list(set(traits))[:4]) 
    
    # 3. CORE LOGIC: Breed & Inject Trend
    new_dna = brain.breed_dna(dna_a, dna_b, request.mix_ratio)

    if hasattr(request, 'trend_id') and request.trend_id:
        trend_path = f"data/centroids/trends/{request.trend_id}.npy"
        if os.path.exists(trend_path):
            trend_dna = np.load(trend_path)
            # Mix trend and re-normalize
            new_dna = (new_dna * (1 - request.trend_weight)) + (trend_dna * request.trend_weight)
            new_dna /= (np.linalg.norm(new_dna) + 1e-8)

    # 4. Apply Aesthetic Sliders
    if request.minimalism != 0:
        new_dna = brain.apply_style_slider(new_dna, "minimalism", request.minimalism)
    
    if request.futurism != 0:
        new_dna = brain.apply_style_slider(new_dna, "vintage_futurism", request.futurism)

    # 5. Generate Images using the updated brain method
    # Main Hero Image
    image_buffer = brain.generate_design(new_dna, custom_prompt=heritage_prompt, layout_type="hero")

    # Style Flat-Lay (The Inspiration Presentation)
    style_buffer = brain.generate_design(new_dna, custom_prompt=heritage_prompt, layout_type="flatlay")

    # 6. Optional: Generate Lineage Map if requested
    # We'll save it with a unique name based on the brands
    map_filename = f"map_{request.brand_a}_{request.brand_b}.png"
    map_buffer = brain.generate_breeding_map(request.brand_a, request.brand_b, request.mix_ratio, map_filename)

    # 7. QC Loop & Encoding
    image_buffer.seek(0)
    generated_img_pil = Image.open(image_buffer).convert("RGB")
    confidence_score = brain.verify_design(generated_img_pil, new_dna)

    # Helper for encoding
    def to_b64(buf):
        buf.seek(0)
        return f"data:image/png;base64,{b64_module.b64encode(buf.read()).decode('utf-8')}"

    return {
        "brand_a": request.brand_a,
        "brand_b": request.brand_b,
        "mix_ratio": request.mix_ratio,
        "trend_applied": request.trend_id if request.trend_id else "None",
        "confidence_score": round(float(confidence_score), 3),
        "detected_heritage": heritage_prompt,
        "image_base64": to_b64(image_buffer),
        "style_base64": to_b64(style_buffer),
        "lineage_map": to_b64(map_buffer) if map_buffer else None,
        "metadata": {
            "parent_a": request.brand_a,
            "parent_b": request.brand_b,
            "ratio": request.mix_ratio
        }
    }

@router.get("/explore/brands", response_model=List[Dict])
async def get_archived_brands():
    """Returns a list of all brands with their detected Heritage Traits and DNA status."""
    archived_brands = []
    
    # Get all potential brand names from your directory
    # (Assuming you have a master list or just scanning the data folder)
    if not os.path.exists(CENTROID_DIR):
        return []

    # Get unique brand names by checking for .json files
    found_files = [f for f in os.listdir(CENTROID_DIR) if f.endswith(".json")]
    
    for file_name in found_files:
        brand_id = file_name.replace(".json", "")
        json_path = os.path.join(CENTROID_DIR, file_name)
        npy_path = os.path.join(CENTROID_DIR, f"{brand_id}.npy")
        
        try:
            # 1. Load the Heritage JSON
            with open(json_path, "r") as f:
                heritage = json.load(f)
            
            # 2. Check if the high-res .npy vector exists
            has_dna = os.path.exists(npy_path)
            
            # 3. Append to the response list
            archived_brands.append({
                "brand": heritage.get("brand", brand_id),
                "traits": heritage.get("traits", []),
                "image_count": heritage.get("image_count", 0),
                "dna_status": "Encoded" if has_dna else "Pending",
                "archived_at": heritage.get("archived_at", "Unknown")
            })
        except Exception as e:
            print(f"⚠️ Error reading data for {brand_id}: {e}")
            continue

    return archived_brands

@router.get("/explore/brands/{brand_name}/dna")
async def get_brand_dna(brand_name: str):
    """
    Returns the raw 512-dimension DNA vector for a specific brand.
    """
    # 1. Construct the file path
    file_path = os.path.join("data", "centroids", f"{brand_name}.npy")
    
    # 2. Check if the brand exists
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Brand DNA for '{brand_name}' not found."
        )
    
    try:
        # 3. Load the NumPy array
        dna_vector = np.load(file_path)
        
        # 4. Convert NumPy array to a standard Python list for JSON response
        return {
            "brand": brand_name,
            "dimensions": len(dna_vector),
            "dna": dna_vector.tolist()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading DNA file: {str(e)}"
        )
    
@router.get("/explore/breed/lineage")
async def get_lineage(brand_a: str, brand_b: str):
    """
    Returns the source metadata for both parents 
    so the frontend can show a comparison.
    """
    def load_meta(name):
        path = f"data/centroids/{name}.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {"error": "Metadata not found"}

    return {
        "parent_a": load_meta(brand_a),
        "parent_b": load_meta(brand_b)
    }

@router.get("/explore/brands/{brand_name}/images")
async def get_brand_images(brand_name: str):
    # The same 'imagekit' instance is used here
    list_files = imagekit.assets.list(path=f"/{brand_name}/", file_type="image")
    return {"images": [f.url for f in list_files]}

@router.post("/explore/breed_styles_with_map")
async def breed_styles(request: BreedRequest):
    # 1. Load DNA and Breed
    path_a = f"data/centroids/{request.brand_a}.npy"
    path_b = f"data/centroids/{request.brand_b}.npy"

    if not os.path.exists(path_a) or not os.path.exists(path_b):
        raise HTTPException(status_code=404, detail="DNA files missing.")

    dna_a, dna_b = np.load(path_a), np.load(path_b)
    new_dna = brain.breed_dna(dna_a, dna_b, request.mix_ratio)

    # 2. Generate the Fashion Design
    design_buffer = brain.generate_design(new_dna)
    
    # 3. Generate the Lineage Map (using the function we built)
    map_filename = f"map_{request.brand_a}_{request.brand_b}_{request.mix_ratio}.png"
    # Assuming 'save_breeding_map' returns a BytesIO object for memory efficiency
    map_buffer = brain.generate_breeding_map(
        request.brand_a, 
        request.brand_b, 
        request.mix_ratio,
        filename=map_filename
    )

    return {
        "metadata": {
            "parent_a": request.brand_a,
            "parent_b": request.brand_b,
            "ratio": request.mix_ratio
        },
        "design_image": f"data:image/png;base64,{to_b64(design_buffer)}",
        "lineage_map": f"data:image/png;base64,{to_b64(map_buffer)}"
    }

@router.get("/explore/galaxy")
async def get_galaxy_map():
    """Returns coordinates for the visual brand landscape."""
    path = "data/galaxy_map.json"
    if not os.path.exists(path):
        # Trigger generation if file is missing
        return {"error": "Galaxy not yet mapped. Run the generator script."}
    
    with open(path, "r") as f:
        return json.load(f)

@router.get("/explore/trends")
async def get_active_trends():
    """
    Scans the trends directory and returns a list of available 
    2026 Street Style DNA profiles.
    """
    try:
        if not os.path.exists(TRENDS_DIR):
            # If the folder doesn't exist yet, return an empty list or create it
            os.makedirs(TRENDS_DIR, exist_ok=True)
            return []

        # List all .npy files and strip the extension for the UI
        trends = [
            f.replace(".npy", "") 
            for f in os.listdir(TRENDS_DIR) 
            if f.endswith(".npy")
        ]
        
        # Sort them so 'cyber_nomad_2026' appears consistently
        return sorted(trends)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index trends: {str(e)}")

# 4. Encode to Base64 (to send in one JSON package)
def to_b64(buffer):
    return b64_module.b64encode(buffer.getvalue()).decode('utf-8')
