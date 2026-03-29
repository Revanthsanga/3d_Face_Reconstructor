# scripts/dataset.py
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    """
    Dataset for 3D face reconstruction.
    Returns: image, coefficients, UV triangles (tri_uvs)
    """
    def __init__(self, img_dir, coeff_dir, tri_uvs_path="data/tri_uvs.json", transform=None):
        self.img_dir = Path(img_dir)
        self.coeff_dir = Path(coeff_dir)
        self.transform = transform

        # ✅ Collect all PNG and JPG images
        self.img_files = sorted(list(self.img_dir.glob("*.png")) + list(self.img_dir.glob("*.jpg")))
        self.coeff_files = sorted(list(self.coeff_dir.glob("*.npy")))

        if len(self.img_files) == 0:
            raise RuntimeError(f"❌ No image files found in {self.img_dir}")
        if len(self.coeff_files) == 0:
            raise RuntimeError(f"❌ No coefficient files found in {self.coeff_dir}")

        if len(self.img_files) != len(self.coeff_files):
            print(f"⚠️ Warning: {len(self.img_files)} images vs {len(self.coeff_files)} coeffs")

         # ✅ Load tri_uvs safely (CPU) - FIXED for performance
        self.tri_uvs = None
        if tri_uvs_path:
            with open(tri_uvs_path, "r") as f:
                data = json.load(f)
            if "tri_uvs" in data:
                try:
                    # ✅ FIX: Convert entire list to numpy array at once for speed
                    tri_array = np.array(data["tri_uvs"], dtype=np.float32)
                    # Check if shape is correct (N, 3, 2)
                    if tri_array.ndim == 3 and tri_array.shape[1:] == (3, 2):
                        self.tri_uvs = torch.from_numpy(tri_array)
                        print(f"[Info] Loaded {len(self.tri_uvs)} triangles from {tri_uvs_path}")
                    else:
                        print(f"[Warning] Invalid tri_uvs shape {tri_array.shape}, expected (N, 3, 2)")
                        self.tri_uvs = torch.zeros((0, 3, 2), dtype=torch.float32)
                except Exception as e:
                    print(f"[Warning] Failed to load tri_uvs: {e}. Using empty tensor.")
                    self.tri_uvs = torch.zeros((0, 3, 2), dtype=torch.float32)
            else:
                print("[Warning] 'tri_uvs' key not found in JSON. Using empty tensor.")
                self.tri_uvs = torch.zeros((0, 3, 2), dtype=torch.float32)
        else:
            self.tri_uvs = torch.zeros((0, 3, 2), dtype=torch.float32)

    def __len__(self):
        # Safe bound: min of images and coeffs
        return min(len(self.img_files), len(self.coeff_files))

    
    def __getitem__(self, idx):
    # ✅ Load image
      img_path = self.img_files[idx]
      img = Image.open(img_path).convert("RGB")
      
      if self.transform:
          img = self.transform(img)
      else:
          img = np.array(img).transpose(2, 0, 1) / 255.0
          img = torch.tensor(img, dtype=torch.float32)
  
      # ✅ Load coefficients
      coeff_path = self.coeff_files[idx]
      coeffs = torch.tensor(np.load(coeff_path), dtype=torch.float32)
      
      # ✅ NORMALIZE COEFFICIENTS
      coeffs = coeffs / 1000000.0
      
      # ✅ CLIP EXTREME OUTLIERS (silently - no printing)
      coeffs = torch.clamp(coeffs, min=-5.0, max=5.0)
      
      # ✅ COMPLETELY REMOVE ALL PRINTING FROM DATASET
      # No sample printing, no warnings
      
      # ✅ Load UVs
      tri_uvs = self.tri_uvs.clone() if self.tri_uvs is not None else torch.zeros((0,3,2), dtype=torch.float32)
      
      return img, coeffs, tri_uvs