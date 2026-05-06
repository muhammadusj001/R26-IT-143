"""
Run predictions using the trained crowd detection model.
"""

import torch
import cv2
import numpy as np
from pathlib import Path


def predict_image(model, image_path, device="cpu"):
    """
    Run prediction on a single image.
    
    Args:
        model: Trained model
        image_path: Path to input image
        device: Device to run prediction on
        
    Returns:
        Prediction result and confidence score
    """
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize to model input size
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Run prediction
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence


def predict_batch(model, image_dir, device="cpu"):
    """
    Run predictions on a batch of images.
    
    Args:
        model: Trained model
        image_dir: Directory containing images
        device: Device to run predictions on
        
    Returns:
        List of predictions with confidence scores
    """
    results = []
    image_path = Path(image_dir)
    
    for img_file in image_path.glob("*.jpg"):
        try:
            predicted_class, confidence = predict_image(model, str(img_file), device)
            results.append({
                'image': img_file.name,
                'class': predicted_class,
                'confidence': confidence
            })
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    return results


if __name__ == "__main__":
    print("Running predictions...")
    # Example prediction code would go here
