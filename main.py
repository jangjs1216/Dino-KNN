import torch
import numpy as np
from src.model import DinoFeatureExtractor
from src.data import DefectDataset
from src.clustering import PatchClusterer
from src.utils import visualize_clusters
from tqdm import tqdm

def main():
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino = DinoFeatureExtractor(device=device)
    dataset = DefectDataset(use_dummy=True) # Change to False/Provide path for real users
    
    all_features = []
    patch_locations = [] # (img_idx, row, col)
    images_pil = []
    
    # 2. Extract
    print("Extracting features...")
    # Target size 640 -> 40x40 patches (Patch size 16)
    TARGET_SIZE = 640 
    PATCH_GRID = 40 
    
    for i in tqdm(range(len(dataset))):
        img, name = dataset[i]
        
        # We need to keep the resized version for accurate visualization crop 
        # OR just use the original providing it's close enough.
        # Let's resize the PIL image to match what the model sees for consistancy.
        img_resized = img.resize((TARGET_SIZE, TARGET_SIZE))
        images_pil.append(img_resized)
        
        # Preprocess & Extract
        tensor = dino.preprocess(img, target_size=TARGET_SIZE)
        features = dino.extract_features(tensor) # 1 x N x 384
        
        # features is 1 x 1600 x 384
        feat_np = features.cpu().numpy().reshape(-1, 384)
        all_features.append(feat_np)
        
        # Record locations
        # DINOv2 usually outputs row-major
        # grid 46x46
        for r in range(PATCH_GRID):
            for c in range(PATCH_GRID):
                patch_locations.append((i, r, c))
                
    # 3. Clustering
    # Stack all: (20 images * 2116 patches) ~ 42000 patches
    all_features_cat = np.concatenate(all_features, axis=0)
    
    print(f"Total features: {all_features_cat.shape}")
    
    clusterer = PatchClusterer(n_clusters=10, n_components=32)
    labels, _ = clusterer.fit_predict(all_features_cat)
    
    # 4. Viz
    print("Visualizing...")
    visualize_clusters(clusterer, images_pil, labels, patch_locations, k_to_show=10, output_path="clusters.png")
    
    print("Done! Check clusters.png to see what each cluster represents.")
    print("You can now assign meaning to clusters (e.g. Cluster 3 = Scratch).")

if __name__ == "__main__":
    main()
