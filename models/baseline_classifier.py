"""
baseline_classifier.py

1) Baseline classifier (x → y): 
A standard image classifier that directly maps images to labels.

- Backbone: convolutional neural network (ResNet-18)
- Single classification head
"""

#import torch.nn as nn


class BaselineClassifier(nn.Module):
    """(x -> y) mapping."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(512, 1) # The output of our backbone is 512-dimensional. Afterwads we place a single
                                    # classification head to predict the binary label (smiling)

    def forward(self, x):
        features = self.backbone(x).view(x.size(0), -1)
        return self.fc(features)
