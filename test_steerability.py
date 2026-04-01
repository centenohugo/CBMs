"""
test_steerability.py

Integration test for evaluate_steerability using the real model classes
from the models/ package. A lightweight DummyBackbone replaces the real
ResNet-18 so the test runs instantly on CPU without pretrained weights.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load the three model files directly via importlib so that models/__init__.py
# is never executed. That file eagerly imports Backbone → torchvision, which
# is not installed here and is not needed: DummyBackbone replaces it entirely.
import importlib.util, pathlib

def _load_module(name: str, rel: str):
    path = pathlib.Path(__file__).parent / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ConceptPredictor      = _load_module("concept_predictor",       "models/concept_predictor.py").ConceptPredictor
ConceptBottleneckModel = _load_module("concept_bottleneck_model", "models/concept_bottleneck_model.py").ConceptBottleneckModel
HybridCBM             = _load_module("hybrid_cbm",              "models/hybrid_cbm.py").HybridCBM
from steerablity import evaluate_steerability, print_steerability_ranking

# NOTE: BaselineClassifier is intentionally excluded.
# Its forward() returns a single logit (B, 1) — it has no concept bottleneck
# and no c_to_y attribute — so it is incompatible with evaluate_steerability
# by design. Steerability evaluation only makes sense for CBM-style models.


# ---------------------------------------------------------------------------
# Dummy backbone
# ---------------------------------------------------------------------------

class DummyBackbone(nn.Module):
    """
    Lightweight stand-in for the real ResNet-18 Backbone class.

    The real backbone outputs (B, 512, 1, 1) via ResNet-18's AdaptiveAvgPool.
    This dummy replicates that exact output shape by:
      1. AdaptiveAvgPool2d → collapses any spatial input to (B, 3, 1, 1)
      2. Conv2d(3, 512, kernel_size=1) → projects to (B, 512, 1, 1)

    Pooling first means the conv operates on a single spatial position,
    making this essentially free on CPU regardless of input resolution.
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Conv2d(3, 512, kernel_size=1)

    def forward(self, x):
        return self.proj(self.pool(x))   # (B, 512, 1, 1)


# ---------------------------------------------------------------------------
# Dummy DataLoader
# ---------------------------------------------------------------------------

def make_dummy_dataloader(num_samples: int = 64, batch_size: int = 16) -> DataLoader:
    """
    Returns a DataLoader that yields (x, c, y) batches with the shapes the
    real CelebA pipeline produces:
      x : (B, 3, 224, 224)  — normalised image tensor
      c : (B, 10)           — binary concept labels (float)
      y : (B, 1)            — binary target label  (float)
    """
    torch.manual_seed(42)
    x = torch.randn(num_samples, 3, 224, 224)
    c = torch.randint(0, 2, (num_samples, 10)).float()
    y = torch.randint(0, 2, (num_samples, 1)).float()
    return DataLoader(TensorDataset(x, c, y), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Model factories (using real classes from models/)
# ---------------------------------------------------------------------------

def build_cbm() -> ConceptBottleneckModel:
    """Standard CBM: x → c → y"""
    backbone = DummyBackbone()
    concept_predictor = ConceptPredictor(backbone, num_concepts=10)
    return ConceptBottleneckModel(concept_predictor, num_concepts=10)


def build_hybrid_cbm() -> HybridCBM:
    """Hybrid CBM: y = f(c) + s(x)"""
    backbone = DummyBackbone()
    concept_predictor = ConceptPredictor(backbone, num_concepts=10)
    return HybridCBM(concept_predictor, num_concepts=10, dropout_p=0.0)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(model_name: str, model: nn.Module, dataloader: DataLoader, device: torch.device):
    print(f"\n{'=' * 60}")
    print(f"Model : {model_name}")
    print(f"Has side_channel: {hasattr(model, 'side_channel')}")
    print(f"{'=' * 60}")
    model.to(device)
    ranking = evaluate_steerability(model, dataloader, device, num_concepts=10)
    print_steerability_ranking(ranking)


if __name__ == "__main__":
    device = torch.device("cpu")
    dataloader = make_dummy_dataloader(num_samples=64, batch_size=16)

    models_to_test = [
        ("ConceptBottleneckModel (standard CBM)", build_cbm()),
        ("HybridCBM (with side channel)",         build_hybrid_cbm()),
    ]

    for model_name, model in models_to_test:
        run_test(model_name, model, dataloader, device)
