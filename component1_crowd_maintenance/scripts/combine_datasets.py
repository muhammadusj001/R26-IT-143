"""
Combine multiple datasets into a single unified dataset.
"""

import os
import shutil
from pathlib import Path


def combine_datasets(dataset_paths, output_path):
    """
    Combine multiple datasets into a single output directory.
    
    Args:
        dataset_paths: List of paths to datasets to combine
        output_path: Path to output combined dataset
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    for i, dataset_path in enumerate(dataset_paths):
        if os.path.exists(dataset_path):
            print(f"Processing dataset {i+1}: {dataset_path}")
            # Copy dataset contents to output
            for item in os.listdir(dataset_path):
                src = os.path.join(dataset_path, item)
                dst = os.path.join(output_path, f"dataset_{i}_{item}")
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
    
    print(f"Combined dataset saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    datasets = ["./datasets/dataset1", "./datasets/dataset2"]
    combine_datasets(datasets, "./datasets/combined")
