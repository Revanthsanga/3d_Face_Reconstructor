"""
scripts/extract_coeffs_from_mat.py
-----------------------------------
Converts 3DMM parameter .mat files (from 300W-LP or AFLW2000-3D)
into numpy .npy coefficient vectors for supervised CNN training.

Each .mat file contains:
  - Shape_Para: (199, 1)
  - Exp_Para:   (29, 1)
Output shape: (228,)
"""

import scipy.io as sio
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path

def extract_coeffs(mat_dir="data/raw/300W_LP", out_dir="data/processed/coeffs"):
    mat_dir = Path(mat_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mat_files = list(mat_dir.rglob("*.mat"))
    if not mat_files:
        print(f"❌ No .mat files found in {mat_dir}. Check your dataset path.")
        return

    print(f"🔍 Found {len(mat_files)} .mat files under {mat_dir}")
    print("⏳ Extracting Shape_Para and Exp_Para ...")

    for mat_path in tqdm(mat_files, desc="Processing"):
        try:
            mat = sio.loadmat(mat_path)
            shape = mat.get("Shape_Para")
            expr = mat.get("Exp_Para")

            if shape is None or expr is None:
                print(f"⚠️ Skipping {mat_path.name}: missing Shape_Para or Exp_Para")
                continue

            shape = shape.flatten()
            expr = expr.flatten()

            if shape.shape[0] != 199 or expr.shape[0] != 29:
                print(f"⚠️ Unexpected shape in {mat_path.name}: {shape.shape}, {expr.shape}")
                continue

            coeffs = np.concatenate([shape, expr])
            out_path = out_dir / (mat_path.stem + ".npy")
            np.save(out_path, coeffs)

        except Exception as e:
            print(f"❌ Error reading {mat_path.name}: {e}")
            continue

    print(f"✅ Extraction complete. Saved .npy files to: {out_dir}")

if __name__ == "__main__":
    # default paths
    extract_coeffs(
        mat_dir="data/raw/300W_LP",         # or "data/raw/AFLW2000-3D"
        out_dir="data/processed/coeffs"
    )
