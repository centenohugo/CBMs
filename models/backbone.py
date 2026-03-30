"""
backbone.py
Provides the shared pretrained ResNet18 feature extractor used by all models.

- Use ResNet-18, available through torchvision.models
- Initialize the model with ImageNet-pretrained weights
- Remove the final classification layer

"""
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights



class Backbone(nn.Module):
    """Pretrained ResNet18 backbone without the final fully connected layer."""
    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # We take all layers except the final fully connected layer
        # and wrap them in a Sequential so that they act as a single module.
        self.backbone = nn.Sequential(*list(model.children())[:-1])  # Output: (B, 512, 1, 1) being B the batch size

    def forward(self, x):
        return self.backbone(x)
