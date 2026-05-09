"""
Fix Label Classes Script
Makes all class labels = 0 (swimmer)
SLIIT FYP - Component 1
"""

from pathlib import Path

BASE_DIR = Path(r"C:\Users\muham\Desktop\Final_Year_Project\component1_crowd_maintenance")
COMBINED = BASE_DIR / "datasets" / "combined_dataset"

print("\n" + "="*50)
print("Fixing label files...")
print("="*50)

fixed = 0
skipped = 0

for split in ['train', 'valid', 'test']:
    labels_dir = COMBINED / split / 'labels'
    if not labels_dir.exists():
        continue

    for lbl_file in labels_dir.glob('*.txt'):
        lines = lbl_file.read_text().strip().splitlines()
        new_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                parts[0] = '0'  # Set class to 0 = swimmer
                new_lines.append(' '.join(parts))

        if new_lines:
            lbl_file.write_text('\n'.join(new_lines))
            fixed += 1
        else:
            skipped += 1

print(f"  ✅ Fixed:   {fixed} label files")
print(f"  ⚠️  Skipped: {skipped} empty files")
print("\n  ✅ All labels set to class 0 (swimmer)")
print("  ✅ Ready for training!")