import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_with_path(brand_a_name, brand_b_name, ratio=0.5):
    centroid_dir = "data/centroids"
    dna_vectors = []
    brand_names = []

    # 1. Load data
    for file in os.listdir(centroid_dir):
        if file.endswith(".npy"):
            dna_vectors.append(np.load(os.path.join(centroid_dir, file)))
            brand_names.append(file.replace(".npy", ""))

    data = np.array(dna_vectors)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(data)
    brand_map = dict(zip(brand_names, coords))

    if brand_a_name not in brand_map or brand_b_name not in brand_map:
        print(f"❌ Error: One or both brands not found in {centroid_dir}")
        return

    # 2. Setup Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=100, c='lightgrey', alpha=0.5)

    # 3. Calculate Breeding Point
    start = brand_map[brand_a_name]
    end = brand_map[brand_b_name]
    # Mathematically find the point on the path
    blend_coord = (1 - ratio) * start + ratio * end

    # 4. Draw Path and Highlight
    plt.plot([start[0], end[0]], [start[1], end[1]], 'r--', alpha=0.8, label="Breeding Path")
    plt.scatter(blend_coord[0], blend_coord[1], color='gold', s=300, marker='*', edgecolors='black', label=f"New Design ({ratio*100}%)")
    
    # Label Parents
    for brand in [brand_a_name, brand_b_name]:
        pos = brand_map[brand]
        plt.annotate(brand.upper(), (pos[0], pos[1]), xytext=(5,5), textcoords='offset points', fontweight='bold', color='red')

    plt.title(f"Breeding Map: {brand_a_name} x {brand_b_name}", fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def save_breeding_map(brand_a_name, brand_b_name, ratio, output_filename):
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

    # 2. Create the plot silently (no popup)
    plt.ioff() 
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c='lightgrey', alpha=0.5)

    # 3. Draw the specific breeding event
    if brand_a_name in brand_map and brand_b_name in brand_map:
        start, end = brand_map[brand_a_name], brand_map[brand_b_name]
        blend_coord = (1 - ratio) * start + ratio * end
        
        ax.plot([start[0], end[0]], [start[1], end[1]], 'r--', alpha=0.6)
        ax.scatter(blend_coord[0], blend_coord[1], color='gold', s=200, marker='*', edgecolors='k')
        ax.annotate("NEW LOOK", (blend_coord[0], blend_coord[1]), xytext=(5,5), textcoords='offset points', color='darkgoldenrod', fontweight='bold')

    ax.set_title(f"DNA Origin: {brand_a_name} x {brand_b_name} ({int(ratio*100)}%)")
    ax.axis('off') # Keep it clean for the UI
    
    # 4. Save to a 'renders' or 'history' folder
    os.makedirs("data/history", exist_ok=True)
    fig.savefig(f"data/history/{output_filename}.png", bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    # Change these names to brands currently in your data/centroids folder
    visualize_with_path("prada", "dior", ratio=0.5)