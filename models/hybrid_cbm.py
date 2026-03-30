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
import torch.optim as optim


class HybridCBM(nn.Module):
    """y = f(c) + s(x) mapping with configurable side-channel dropout."""

    def __init__(self, concept_predictor, backbone, num_concepts=10, dropout_p=0.0):
        super().__init__()
        self.concept_predictor = concept_predictor
        self.c_to_y = nn.Linear(num_concepts, 1)

        # Side channel s(x)
        self.side_dropout = nn.Dropout(p=dropout_p)
        self.backbone = backbone
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        # 1. f(c) --> concept-based classifier
        features = self.concept_predictor.backbone(x).view(x.size(0), -1)
        c_logits = self.concept_predictor.fc_concepts(features)
        c_soft = torch.sigmoid(c_logits)
        f_c = self.c_to_y(c_soft)

        # 2. s(x) --> side-channel head operating on backbone features.
        features = self.backbone(x).view(x.size(0), -1)
        features_dropped = self.side_dropout(features)
        s_x = self.fc(features_dropped)

        # 3. Combine (summing the logits from both paths)
        y_logits = f_c + s_x
        return c_logits, y_logits
    
    

class HybridCBM_extended(HybridCBM):

    def __init__(self, concept_predictor, backbone, epochs=100, lr=0.001, lambda_c=1.0):

        super().__init__(concept_predictor, backbone, num_concepts=10, dropout_p=0.0)  # To initialize BaselineClassifier

        self.lr = lr  # Learning Rate

        self.optim = optim.Adam(self.parameters(), self.lr)

        self.epochs = epochs
        
        self.lambda_c = lambda_c  # Weight for the concept bottleneck loss

        # Loss for the final prediction y
        self.criterion = nn.BCEWithLogitsLoss()

        # Loss for the intermediate concepts c
        self.concept_criterion = nn.BCEWithLogitsLoss()
        
        self.loss_during_training = []

        self.valid_loss_during_training = []

    def trainloop(self, trainloader, validloader):

        # Move parameters to GPU.
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

            # Iterate over batches
            for batch_idx, (x, c_true, y) in enumerate(trainloader):

                x, c_true, y = x.to(device), c_true.to(device), y.to(device)

                self.optim.zero_grad()

                c_logits, out = self.forward(x)

                # Final task loss (y)
                loss_y = self.criterion(out, y.unsqueeze(1).float())
                
                # Concept bottleneck loss (c)
                loss_c = self.concept_criterion(c_logits, c_true.float())

                loss = loss_y + (self.lambda_c * loss_c)

                running_loss += loss.item()

                loss.backward()

                self.optim.step()

                if (batch_idx + 1) % 100 == 0:
                    print(f"  Epoch {e+1}/{int(self.epochs)}  "
                            f"batch {batch_idx+1}/{len(trainloader)}  "
                            f"Total Loss: {loss.item():.4f} "
                            f"(Task: {loss_y.item():.4f}, Concept: {loss_c.item():.4f})", flush=True)

            self.loss_during_training.append(running_loss / len(trainloader))

            # Validation Loss

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():

                # set model to evaluation mode
                self.eval()

                running_loss = 0.

                for x, c_true, y in validloader:

                    x, c_true, y = x.to(device), c_true.to(device), y.to(device)

                    c_logits, out = self.forward(x)

                    loss_y = self.criterion(out, y.unsqueeze(1).float())
                    
                    loss_c = self.concept_criterion(c_logits, c_true.float())

                    loss = loss_y + (self.lambda_c * loss_c)

                    running_loss += loss.item()

                self.valid_loss_during_training.append(running_loss / len(validloader))

            # set model back to train mode
            self.train()

            print("Epoch %d/%d. Training loss: %f, Validation loss: %f"
                    % (e+1, int(self.epochs),
                    self.loss_during_training[-1], self.valid_loss_during_training[-1]))

        print("\nTraining complete.")
        self.eval()