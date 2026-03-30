"""
concept_predictor.py

2) Concept predictor (x → c)
A multi-label model that predicts the predefined concept vector from the input image.

- Shared backbone (ResNet-18)
- One binary prediction head per concept (or an equivalent multi-output head)
- Binary cross-entropy loss (use class weighting if appropriate)

"""
import torch
import torch.nn as nn
import torch.optim as optim


class ConceptPredictor(nn.Module):
    """x -> c mapping. Predicts 10 concepts."""

    def __init__(self, backbone, num_concepts=10, pos_weight=None):
        super().__init__()
        self.backbone = backbone
        # One output per concept
        self.fc_concepts = nn.Linear(512, num_concepts)
        # Optional: class weighting for imbalanced concepts. Considered in during training by passing the pos_weight argument to the
        # loss critetion (BCEWithLogitsLoss) as pos_weight argument. If None, no weighting is applied.
        self.pos_weight = pos_weight

    def forward(self, x):
        features = self.backbone(x).view(x.size(0), -1)
        # Return logits for numerical stability in BCEWithLogitsLoss
        return self.fc_concepts(features)


class ConceptPredictor_extended(ConceptPredictor):

    def __init__(self, backbone, num_concepts=10, epochs=100, lr=0.001, pos_weight=None):

        super().__init__(backbone, num_concepts, pos_weight)  # To initialize ConceptPredictor

        self.lr = lr  # Learning Rate

        self.optim = optim.Adam(self.parameters(), self.lr)

        self.epochs = epochs

        # Each of the 10 concepts is an independent binary classification problem
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)  # Use pos_weight for class weighting if provided

        # A list to store the loss evolution along training
        self.loss_during_training = []

        self.valid_loss_during_training = []

    def trainloop(self, trainloader, validloader):

        device = next(self.parameters()).device
        print(f"Starting training: {int(self.epochs)} epochs, lr={self.lr}, "
              f"{len(trainloader)} train batches, {len(validloader)} val batches, "
              f"device={device}\n", flush=True)

        # set model back to train mode
        self.train()

        # Optimization Loop
        for e in range(int(self.epochs)):

            print(f"\nEpoch {e+1}/{int(self.epochs)} — training...", flush=True)

            running_loss = 0.

            for batch_idx, (x, c, _) in enumerate(trainloader):

                x, c = x.to(device), c.to(device)

                self.optim.zero_grad()

                # out shape: (batch_size, num_concepts)
                out = self.forward(x)

                # c must be float for BCEWithLogitsLoss
                loss = self.criterion(out, c.float())

                running_loss += loss.item()

                loss.backward()

                self.optim.step()

                if (batch_idx + 1) % 100 == 0:
                    print(f"  Epoch {e+1}/{int(self.epochs)}  "
                          f"batch {batch_idx+1}/{len(trainloader)}  "
                          f"batch loss: {loss.item():.4f}", flush=True)

            self.loss_during_training.append(running_loss / len(trainloader))

            # Validation Loss

            with torch.no_grad():

                self.eval()

                running_loss = 0.

                for x, c, _ in validloader:

                    x, c = x.to(device), c.to(device)

                    out = self.forward(x)

                    loss = self.criterion(out, c.float())

                    running_loss += loss.item()

                self.valid_loss_during_training.append(running_loss / len(validloader))

            self.train()

            print("Epoch %d/%d. Training loss: %f, Validation loss: %f"
                  % (e+1, int(self.epochs),
                     self.loss_during_training[-1], self.valid_loss_during_training[-1]),
                  flush=True)

        print("\nTraining complete.")
        self.eval()
