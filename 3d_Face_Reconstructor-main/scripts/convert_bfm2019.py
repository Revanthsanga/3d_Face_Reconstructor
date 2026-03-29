import h5py
import numpy as np
from pathlib import Path

# Path to your downloaded BFM 2019 .h5 file
BFM_H5_PATH =  "data/raw/bfm2019/model2019_bfm.h5"

# Output directory
OUT_DIR = Path("data/3dmm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def convert_bfm2019():
    print(f"🔍 Loading {BFM_H5_PATH} ...")
    with h5py.File(BFM_H5_PATH, "r") as f:
        # Mean shape (vectorized)
        S_mean = np.array(f["shape/model/mean"][:])
        np.save(OUT_DIR / "S_mean.npy", S_mean)

        # Identity basis
        B_id = np.array(f["shape/model/pcaBasis"][:])
        np.save(OUT_DIR / "B_id.npy", B_id)

        # Expression basis
        B_exp = np.array(f["expression/model/pcaBasis"][:])
        np.save(OUT_DIR / "B_exp.npy", B_exp)

        # Triangular faces
        faces = np.array(f["shape/representer/cells"][:]).T
        np.save(OUT_DIR / "faces.npy", faces)

        # UV coordinates (needed for texture mapping)
        if "tex/representer/points" in f:
            uv_coords = np.array(f["tex/representer/points"][:])
            np.save(OUT_DIR / "uv_coords.npy", uv_coords)
            print("✅ Saved UV coordinates (uv_coords.npy)")
        else:
            print("⚠️ UV coordinates not found in file, texture mapping will not work.")

    print("✅ Conversion complete. Files saved in data/3dmm/")

if __name__ == "__main__":
    convert_bfm2019()
