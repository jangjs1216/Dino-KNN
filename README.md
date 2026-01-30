# Dino-KNN Defect Detection

This project works on unsupervised auto-labeling of defects on 640x640 images using Dino-v3 features and K-Means clustering.

## Setup

1. Install dependencies:
   ```bash
   pip install torch torchvision scikit-learn matplotlib tqdm pillow
   ```
2. Ensure the model weights `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` are in the root directory.

## Usage

1. Run the main script:
   ```bash
   python main.py
   ```
   By default, it uses dummy data. To use real data, modify `dataset = DefectDataset(use_dummy=False, image_dir="path/to/images")` in `main.py`.

## Structure

- `src/model.py`: DinoV3 wrapper.
- `src/dino_arch.py`: DinoV3 architecture definition.
- `src/clustering.py`: PCA and K-Means logic.
- `src/data.py`: Data loading and dummy data generation.
- `src/utils.py`: Visualization tools.
