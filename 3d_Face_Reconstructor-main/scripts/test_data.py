# Create a test script: check_data.py
import numpy as np
from pathlib import Path

coeff_dir = Path("data/processed/coeffs")
coeff_files = list(coeff_dir.glob("*.npy"))

print(f"Total coefficient files: {len(coeff_files)}")

# Check for duplicate coefficients
unique_coeffs = {}
duplicates = []

for i, coeff_file in enumerate(coeff_files[:100]):  # Check first 100
    coeffs = np.load(coeff_file)
    coeff_hash = str(coeffs[:10])  # First 10 values as fingerprint
    
    if coeff_hash in unique_coeffs:
        duplicates.append((coeff_file, unique_coeffs[coeff_hash]))
    else:
        unique_coeffs[coeff_hash] = coeff_file
    
    if i % 20 == 0:
        print(f"Checked {i} files...")

print(f"Found {len(duplicates)} potential duplicates")
for dup in duplicates[:5]:  # Show first 5 duplicates
    print(f"Duplicate: {dup[0].name} and {dup[1].name}")