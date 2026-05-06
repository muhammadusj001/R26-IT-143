# Component 1: Crowd Maintenance

## Overview
This component handles crowd detection and monitoring in the swimming pool area using computer vision and deep learning techniques.

## Features
- Real-time crowd detection
- Capacity management
- Occupancy tracking
- Crowd density analysis

## Directory Structure
- `datasets/` - Training and validation datasets
- `models/` - Pre-trained and fine-tuned models
- `scripts/` - Python scripts for training, evaluation, and prediction
- `results/` - Model outputs and analysis results

## Scripts
- `combine_datasets.py` - Combine multiple datasets
- `fix_labels.py` - Fix and validate label data
- `train.py` - Train the crowd detection model
- `evaluate.py` - Evaluate model performance
- `predict.py` - Run predictions on new data

## Usage
```bash
python scripts/train.py
python scripts/evaluate.py
python scripts/predict.py --input <path_to_image>
```

## Dependencies
See `requirements.txt` in the root directory
