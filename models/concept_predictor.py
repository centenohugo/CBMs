"""
concept_predictor.py

2) Concept predictor (x → c)
A multi-label model that predicts the predefined concept vector from the input image.

- Shared backbone (ResNet-18)
- One binary prediction head per concept (or an equivalent multi-output head)
- Binary cross-entropy loss (use class weighting if appropriate)

"""
import torch.nn as nn


class ConceptPredictor(nn.Module):
    """x -> c mapping. Predicts 10 concepts."""

    def __init__(self, backbone, num_concepts=10):
        super().__init__()
        self.backbone = backbone
        self.fc_concepts = nn.Linear(512, num_concepts)

    def forward(self, x):
        features = self.backbone(x).view(x.size(0), -1)
        # Return logits for numerical stability in BCEWithLogitsLoss
        return self.fc_concepts(features)
