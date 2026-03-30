"""
hybrid_cbm.py

4) Hybrid CBM with side channel
Extend the CBM by adding a direct image-to-label pathway:

y = f(c) + s(x)

where:

f(c) is the concept-based classifier, and
s(x) is a side-channel head operating on backbone features.

The final prediction is obtained by summing the logits from both paths.

"""

import torch
import torch.nn as nn


class HybridCBM(nn.Module):
    """y = f(c) + s(x) mapping with configurable side-channel dropout."""

    def __init__(self, concept_predictor, num_concepts=10, dropout_p=0.0):
        super().__init__()
        self.concept_predictor = concept_predictor
        self.c_to_y = nn.Linear(num_concepts, 1)

        # Side channel s(x)
        self.side_dropout = nn.Dropout(p=dropout_p)
        self.side_channel = nn.Linear(512, 1)

    def forward(self, x):
        # 1. Concept path
        features = self.concept_predictor.backbone(x).view(x.size(0), -1)
        c_logits = self.concept_predictor.fc_concepts(features)
        c_soft = torch.sigmoid(c_logits)
        f_c = self.c_to_y(c_soft)

        # 2. Side channel path (operating on backbone features)
        features_dropped = self.side_dropout(features)
        s_x = self.side_channel(features_dropped)

        # 3. Combine
        y_logits = f_c + s_x
        return c_logits, y_logits
