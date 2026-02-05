import base64 as b64_module
import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import numpy as np
import os
from .brain import brain
from config.imagekit_config import imagekit

router = APIRouter()

class BreedRequest(BaseModel):
    brand_a: str = Field(..., example="dior_vintage")
    brand_b: str = Field(..., example="mcqueen_gothic")
    # Enforce that the slider must be between 0 and 1
    mix_ratio: float = Field(..., ge=0.0, le=1.0, description="The blend ratio between Brand A and B")

@router.post("/explore/breed")
async def breed_styles(request: BreedRequest):
    # Standard logic follows...
    path_a = f"data/centroids/{request.brand_a}.npy"
    path_b = f"data/centroids/{request.brand_b}.npy"

    if not os.path.exists(path_a) or not os.path.exists(path_b):
        raise HTTPException(status_code=404, detail="DNA files missing.")

    dna_a = np.load(path_a)
    dna_b = np.load(path_b)
    
    new_dna = brain.breed_dna(dna_a, dna_b, request.mix_ratio)
    # image_bytes = brain.generate_design(new_dna)
    image_buffer = brain.generate_design(new_dna)

    return StreamingResponse(image_buffer, media_type="image/png")

@router.get("/explore/brands")
async def get_all_brands():
    """
    Scans the centroids directory and returns a list 
    of all available brand DNA profiles.
    """
    # Define the path to your DNA files
    centroid_dir = os.path.join("data", "centroids")
    
    # 1. Check if the directory exists to avoid errors
    if not os.path.exists(centroid_dir):
        return {"brands": [], "message": "No DNA files found. Run the generator script first."}

    # 2. Get all .npy files and strip the extension
    # We use a list comprehension for efficiency
    brands = [
        f.replace(".npy", "") 
        for f in os.listdir(centroid_dir) 
        if f.endswith(".npy")
    ]
    
    # 3. Sort them alphabetically for a better UI experience
    brands.sort()
    
    return {
        "count": len(brands),
        "brands": brands
    }

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

    # 4. Encode to Base64 (to send in one JSON package)
    def to_b64(buffer):
        return b64_module.b64encode(buffer.getvalue()).decode('utf-8')

    return {
        "metadata": {
            "parent_a": request.brand_a,
            "parent_b": request.brand_b,
            "ratio": request.mix_ratio
        },
        "design_image": f"data:image/png;base64,{to_b64(design_buffer)}",
        "lineage_map": f"data:image/png;base64,{to_b64(map_buffer)}"
    }