"""
Evaluate crowd detection model performance.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, test_loader, device="cpu"):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics


def print_metrics(metrics):
    """Print evaluation metrics."""
    print("=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    print("Evaluating model...")
    # Example evaluation code would go here
