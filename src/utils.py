import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize_clusters(cluster_obj, images, labels, patch_locations, k_to_show=5, patch_size=16, output_path="cluster_viz.png"):
    """
    images: List of PIL Images (original) or Paths
    labels: Full label array for all patches
    patch_locations: List of (image_index, row, col) for each patch in labels
    k_to_show: How many random samples to show per cluster
    """
    n_clusters = cluster_obj.n_clusters
    
    # Setup grid: Rows = Clusters, Cols = Samples
    fig, axes = plt.subplots(n_clusters, k_to_show + 1, figsize=(2 * (k_to_show+1), 2 * n_clusters))
    
    # Organize indices by cluster
    cluster_indices = {i: [] for i in range(n_clusters)}
    for idx, lbl in enumerate(labels):
        cluster_indices[lbl].append(idx)
        
    for c_id in range(n_clusters):
        # First column: Cluster ID text
        ax_text = axes[c_id, 0]
        ax_text.text(0.5, 0.5, f"Cluster {c_id}\nCount: {len(cluster_indices[c_id])}", 
                     ha='center', va='center', fontsize=12)
        ax_text.axis('off')
        
        # Sample patches
        indices = cluster_indices[c_id]
        if not indices:
            continue
            
        sampled_indices = np.random.choice(indices, min(len(indices), k_to_show), replace=False)
        
        for i, patch_global_idx in enumerate(sampled_indices):
            img_idx, row, col = patch_locations[patch_global_idx]
            original_img = images[img_idx]
            
            # Crop logic
            # Note: We resized in model to 644, but here we might be using original images.
            # For simplicity in this demo, assuming original mapping roughly holds or efficient crop.
            # To be precise, we should replicate the resize logic or map coord back.
            # Here we assume the input 'images' are the PIL images passed to model.
            
            # If original was 640x640 and we resized to 644x644, the patch 14x14 corresponds to check 
            # (row*14, col*14).
            
            # We will grab from the tensor-ready resized version if possible, but we passed PIL images?
            # Let's just crop from the PIL image assuming it was resized to 644 before crop, 
            # OR map coordinates back to 640.
            # 644 -> 640 ratio is close to 1. 0.99.
            # Let's just crop (col*14, row*14) to (col*14+14, row*14+14)
            # and ignore the slight boundary risk for visualization.
            
            left = col * patch_size
            top = row * patch_size
            # If the image is smaller (640), this might go out of bounds for the last patch.
            # We'll safeguard.
            
            # Ideally we should verify the image size.
            w, h = original_img.size
            
            crop = original_img.crop((left, top, min(left+patch_size, w), min(top+patch_size, h)))
            
            ax = axes[c_id, i+1]
            ax.imshow(crop)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
