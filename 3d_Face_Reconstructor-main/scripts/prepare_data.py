import cv2, os
from pathlib import Path
from retinaface import RetinaFace
from tqdm import tqdm

INPUT_DIR = "data/raw/300W_LP"
OUTPUT_DIR = "data/processed/images"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def preprocess_image(img_path, out_path, size=224):
    img = cv2.imread(img_path)
    if img is None:
        return False

    faces = RetinaFace.detect_faces(img)
    if isinstance(faces, dict):
        for _, face in faces.items():
            x1, y1, x2, y2 = face["facial_area"]
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (size, size))
            cv2.imwrite(out_path, crop)
            return True
    return False

def main():
    all_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(INPUT_DIR)
        for f in files if f.lower().endswith((".jpg", ".png"))
    ]

    print(f"Found {len(all_files)} images to preprocess.")

    for f in tqdm(all_files, desc="Preprocessing images"):
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(f))
        if os.path.exists(out_path):
            continue
        ok = preprocess_image(f, out_path)
        if not ok:
            print(f"❌ No face detected in {f}")

if __name__ == "__main__":
    main()
