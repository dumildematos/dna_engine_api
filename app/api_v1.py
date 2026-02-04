import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import numpy as np
import os
from .brain import brain

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