import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from src.dino_arch import DinoV3

class DinoFeatureExtractor(nn.Module):
    def __init__(self, model_path="dinov3_vits16_pretrain_lvd1689m-08c60483.pth", device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        print(f"Loading local DinoV3 model from {model_path} on {device}...")
        
        # Initialize model architecture based on the file content seen in dino.py
        self.model = DinoV3(embed_dim=384, depth=12, num_heads=6, patch_size=16)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"WARNING: Model file {model_path} not found! Using random weights.")
            
        self.model.to(self.device)
        self.model.eval()
        self.patch_size = 16
        
        # Standard ImageNet normalization
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def preprocess(self, image: Image.Image, target_size=640):
        """
        Resize image to be a multiple of patch_size (16).
        640 is exactly 40 * 16.
        """
        # Resize to target_size directly
        return self.normalize(image.resize((target_size, target_size))).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract_features(self, images):
        """
        input: BxCxHxW tensor
        output: BxNxD tensor (N = patches, D = embedding dim)
        """
        # Model returns [B, N_tokens, Dim].
        # In DinoV3 code: tokens = cls(1) + storage(4) + patches(N)
        output = self.model(images)
        
        # Remove the first 5 special tokens to get only patch tokens
        patch_features = output[:, 5:, :] 
        return patch_features

    @property
    def embed_dim(self):
        return 384
