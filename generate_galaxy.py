import os
import numpy as np
import json
from sklearn.decomposition import PCA
import pandas as pd

CENTROID_DIR = "data/centroids"

def generate_fashion_galaxy():
    dna_vectors = []
    brand_names = []
    metadata = []

    # 1. Load all archived DNA
    files = [f for f in os.listdir(CENTROID_DIR) if f.endswith(".npy")]
    
    for f in files:
        brand = f.replace(".npy", "")
        vector = np.load(os.path.join(CENTROID_DIR, f))
        
        # Load traits for the tooltip
        traits = []
        json_path = os.path.join(CENTROID_DIR, f"{brand}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as j:
                traits = json.load(j).get("traits", [])

        dna_vectors.append(vector)
        brand_names.append(brand)
        metadata.append(", ".join(traits))

    # 2. PCA Projection (512D -> 2D)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(dna_vectors)

    # 3. Create a clean Galaxy Map
    galaxy_df = pd.DataFrame({
        "brand": brand_names,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "traits": metadata
    })

    galaxy_df.to_json("data/galaxy_map.json", orient="records")
    print(f"✨ Galaxy Map generated with {len(brand_names)} brands.")

if __name__ == "__main__":
    generate_fashion_galaxy()