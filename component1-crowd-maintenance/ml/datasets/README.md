# YOLO Dataset Structure

This directory is organized in the standard YOLO format for training, validation, and testing.

## Folder Layout

```text
datasets/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## What Goes Where

- `train/images`: Training images
- `train/labels`: YOLO label files for training images
- `valid/images`: Validation images
- `valid/labels`: YOLO label files for validation images
- `test/images`: Test images
- `test/labels`: YOLO label files for test images

## YOLO Label Format

Each image should have a matching `.txt` file with the same filename in the corresponding `labels` folder.

Each label line must follow this format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All bounding box values must be normalized between `0` and `1` relative to the image width and height.

### Example

```text
0 0.512 0.438 0.120 0.210
```

This means:
- Class `0`
- Bounding box center at `x=0.512`, `y=0.438`
- Box width `0.120`
- Box height `0.210`

## Notes

- Keep image and label filenames aligned.
- Use one label file per image.
- Empty images can have empty `.txt` files if needed.
- Update your YOLO `data.yaml` to point to these folders when training.
