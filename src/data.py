import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import random
import glob

class DefectDataset(Dataset):
    def __init__(self, use_dummy=False, image_dir=None):
        self.use_dummy = use_dummy
        self.image_paths = []
        if not use_dummy and image_dir:
            self.image_paths = glob.glob(os.path.join(image_dir, "*.png")) + \
                               glob.glob(os.path.join(image_dir, "*.jpg"))
        
        # If no images found or use_dummy is True, we simulate data
        if not self.image_paths:
            self.use_dummy = True
            print("No images found or dummy mode selected. Generating dummy data on fly.")

    def __len__(self):
        return 20 if self.use_dummy else len(self.image_paths)

    def __getitem__(self, idx):
        if self.use_dummy:
            return self._generate_dummy_image(idx)
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return image, os.path.basename(img_path)

    def _generate_dummy_image(self, idx):
        # 640x640 gray background
        img = Image.new('RGB', (640, 640), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        
        # Add some random "texture" (noise)
        # Using simple shapes to simulate defects
        
        # Type 1: Scratches (Lines) - localized
        if idx % 3 == 0:
            for _ in range(5):
                x1 = random.randint(100, 500)
                y1 = random.randint(100, 500)
                draw.line((x1, y1, x1+random.randint(-20, 20), y1+random.randint(20, 50)), fill=(50, 50, 50), width=2)
                
        # Type 2: Dents (Circles) - localized
        elif idx % 3 == 1:
            for _ in range(3):
                x1 = random.randint(100, 500)
                y1 = random.randint(100, 500)
                r = random.randint(5, 15)
                draw.ellipse((x1-r, y1-r, x1+r, y1+r), fill=(150, 150, 150), outline=(100, 100, 100))

        # Type 3: Clean (mostly)
        else:
            pass

        return img, f"dummy_{idx}.png"
