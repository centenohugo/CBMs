"""
baseline_classifier.py

1) Baseline classifier (x → y):
A standard image classifier that directly maps images to labels.

- Backbone: convolutional neural network (ResNet-18)
- Single classification head
"""

import torch
import torch.nn as nn
import torch.optim as optim


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


class BaselineClassifier_extended(BaselineClassifier):

    def __init__(self, backbone, epochs=100, lr=0.001):

        super().__init__(backbone)  # To initialize BaselineClassifier

        self.lr = lr  # Learning Rate

        self.optim = optim.Adam(self.parameters(), self.lr)

        self.epochs = epochs

        # Binary Cross-Entropy Loss with Logits (combines a Sigmoid layer and the BCELoss in one single class)
        self.criterion = nn.BCEWithLogitsLoss()  # Binary classification (smiling/not smiling)

        # A list to store the loss evolution along training
        self.loss_during_training = []

        self.valid_loss_during_training = []

    def trainloop(self, trainloader, validloader):

        # Move parameters to external device, luckily GPU.
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

            # Concepts c are not used in this baseline model, so we ignore them during training and validation.
            for batch_idx, (x, _, y) in enumerate(trainloader):

                x, y = x.to(device), y.to(device)

                self.optim.zero_grad()

                out = self.forward(x)

                loss = self.criterion(out, y.unsqueeze(1).float())

                running_loss += loss.item()

                loss.backward()

                self.optim.step()

                if (batch_idx + 1) % 100 == 0:
                    print(f"  Epoch {e+1}/{int(self.epochs)}  "
                          f"batch {batch_idx+1}/{len(trainloader)}  "
                          f"batch loss: {loss.item():.4f}", flush=True)

            self.loss_during_training.append(running_loss / len(trainloader))

            # Validation Loss

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():

                # set model to evaluation mode
                self.eval()

                running_loss = 0.

                for x, _, y in validloader:

                    x, y = x.to(device), y.to(device)

                    out = self.forward(x)

                    loss = self.criterion(out, y.unsqueeze(1).float())

                    running_loss += loss.item()

                self.valid_loss_during_training.append(running_loss / len(validloader))

            # set model back to train mode
            self.train()

            print("Epoch %d/%d. Training loss: %f, Validation loss: %f"
                  % (e+1, int(self.epochs),
                     self.loss_during_training[-1], self.valid_loss_during_training[-1]))

        print("\nTraining complete.")
        self.eval()
