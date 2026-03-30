"""
concept_bottleneck_model.py

3) Concept Bottleneck Model (x → c → y)
A model in which predictions must pass explicitly through the concept representation.

- Concept predictor as defined above
- Label head operating only on predicted concepts (e.g. linear layer or small MLP)

Training strategy
Choose one of the following and state it clearly:

- Independent CBM: Train x → c, freeze it, then train c → y
- Joint CBM: Train the entire model end-to-end with a combined loss
"""
import torch
import torch.nn as nn


class ConceptBottleneckModel(nn.Module):
    """x -> c -> y mapping"""

    def __init__(self, concept_predictor, num_concepts=10):
        super().__init__()
        self.concept_predictor = concept_predictor
        self.c_to_y = nn.Linear(num_concepts, 1)

    def forward(self, x):
        c_logits = self.concept_predictor(x)
        # Use sigmoid to pass bounded "soft concepts" to the final layer
        c_soft = torch.sigmoid(c_logits)
        y_logits = self.c_to_y(c_soft)
        return c_logits, y_logits
