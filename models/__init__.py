"""
models package
Re-exports every model component so existing code can still do:
    from models import get_backbone, BaselineClassifier, ...
"""
from .backbone import Backbone
from .baseline_classifier import BaselineClassifier
from .concept_predictor import ConceptPredictor
from .concept_bottleneck_model import ConceptBottleneckModel
from .hybrid_cbm import HybridCBM

__all__ = [
    "Backbone",
    "BaselineClassifier",
    "ConceptPredictor",
    "ConceptBottleneckModel",
    "HybridCBM",
]
