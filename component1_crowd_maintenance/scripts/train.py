"""
Train crowd detection model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


class CrowdDetectionModel(nn.Module):
    """Simple crowd detection model."""
    
    def __init__(self, num_classes=2):
        super(CrowdDetectionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 112 * 112, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_model(train_loader, epochs=10, learning_rate=0.001):
    """
    Train the crowd detection model.
    
    Args:
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrowdDetectionModel().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


if __name__ == "__main__":
    print("Training crowd detection model...")
    # Example training code would go here
