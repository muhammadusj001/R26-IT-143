"""
Dataset Combination Script
SLIIT Final Year Project - Component 1
Crowd-Aware Maintenance Scheduling Module
Author: Mohamed Saajith
"""

import os
import shutil
from pathlib import Path

# ================================================
# PATHS CONFIGURATION
# ================================================
# Base directory
BASE_DIR = Path(r"C:\Users\muham\Desktop\Final_Year_Project\component1_crowd_maintenance")

# Your 3 dataset folders
DATASETS = [
    BASE_DIR / "datasets" / "People Swimming.v1-roboflow-instant-1--eval-.yolov8",
    BASE_DIR / "datasets" / "swimmer detector2.v6i.yolov8",
    BASE_DIR / "datasets" / "swimmer- swimmer.v1i.yolov8",
]

# Output folder
OUTPUT = BASE_DIR / "datasets" / "combined_dataset"

# ================================================
# CREATE FOLDER STRUCTURE
# ================================================
print("\n" + "="*50)
print("STEP 1: Creating folder structure...")
print("="*50)

for split in ['train', 'valid', 'test']:
    (OUTPUT / split / 'images').mkdir(parents=True, exist_ok=True)
    (OUTPUT / split / 'labels').mkdir(parents=True, exist_ok=True)
    print(f"  ✅ Created: {split}/images and {split}/labels")

# ================================================
# COPY FILES
# ================================================
print("\n" + "="*50)
print("STEP 2: Copying images and labels...")
print("="*50)

totals = {'train': 0, 'valid': 0, 'test': 0}

for i, dataset in enumerate(DATASETS, 1):
    if not dataset.exists():
        print(f"\n  ⚠️  Dataset {i} NOT FOUND: {dataset.name}")
        print(f"      Check your folder name matches exactly!")
        continue

    print(f"\n  📦 Dataset {i}: {dataset.name}")

    for split in ['train', 'valid', 'test']:
        src_img = dataset / split / 'images'
        src_lbl = dataset / split / 'labels'
        dst_img = OUTPUT / split / 'images'
        dst_lbl = OUTPUT / split / 'labels'

        if not src_img.exists():
            print(f"    ⚠️  No {split} folder")
            continue

        count = 0
        for img in src_img.iterdir():
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Add prefix to avoid duplicate names
                new_name = f"d{i}_{img.name}"
                shutil.copy2(img, dst_img / new_name)

                # Copy label
                lbl = src_lbl / (img.stem + '.txt')
                if lbl.exists():
                    shutil.copy2(lbl, dst_lbl / f"d{i}_{lbl.name}")
                count += 1

        totals[split] += count
        print(f"    ✅ {split}: {count} images")

# ================================================
# CREATE data.yaml
# ================================================
print("\n" + "="*50)
print("STEP 3: Creating data.yaml...")
print("="*50)

yaml = f"""# Combined Swimmer Dataset
# SLIIT FYP - Component 1: Crowd-Aware Maintenance

path: {OUTPUT.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['swimmer']
"""

with open(OUTPUT / 'data.yaml', 'w') as f:
    f.write(yaml)

print("  ✅ data.yaml created!")

# ================================================
# FINAL SUMMARY
# ================================================
print("\n" + "="*50)
print("✅ COMBINATION COMPLETE!")
print("="*50)
print(f"  Training images:   {totals['train']:,}")
print(f"  Validation images: {totals['valid']:,}")
print(f"  Test images:       {totals['test']:,}")
print(f"  TOTAL:             {sum(totals.values()):,}")
print(f"\n  📂 Output: {OUTPUT}")
print("\n  ✅ Ready for label fixing!")