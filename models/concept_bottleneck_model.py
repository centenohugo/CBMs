"""
Concept Bottleneck Model (CBM) Implementation
Project: Heavy Version (CelebA)

Architecture Choice: Joint CBM
We selected the Joint Concept Bottleneck Model (end-to-end training) rather than 
the Independent CBM (sequential training). 
Reasoning: 
1. The Joint CBM optimizes the shared backbone for both concept prediction and 
   the final target task simultaneously. 
2. This mitigates the "accuracy tax" (loss in predictive performance) typically 
   associated with strict bottlenecking, allowing the model to achieve target 
   accuracy (Smiling) on par with a standard black-box (x -> y) model.
3. It retains full interpretability because the final decision is strictly 
   computed from the 10 concept probabilities.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np

class ConceptBottleneckModel(nn.Module):
    """
    The PyTorch Neural Network architecture for the Concept Bottleneck Model.
    It takes an image (x), predicts 10 interpretable concepts (c), and uses 
    ONLY those concepts to predict the final target (y).
    """
    def __init__(self, num_concepts=10, num_classes=1):
        super(ConceptBottleneckModel, self).__init__()

        # 1. Shared Backbone: ResNet-18 (Pretrained on ImageNet)
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.backbone.fc.in_features

        # Remove the final classification layer so it just outputs feature embeddings (512 dims)
        self.backbone.fc = nn.Identity()

        # 2. Concept Predictor Head (x -> c)
        # Maps the backbone features to the 10 facial attribute concepts
        self.concept_head = nn.Linear(num_features, num_concepts)

        # 3. Target Label Head (c -> y)
        # Maps the 10 predicted concepts to the final target ('Smiling')
        self.label_head = nn.Linear(num_concepts, num_classes)

    def forward(self, x):
        # Pass image through the backbone
        features = self.backbone(x)

        # Get raw predictions for the concepts (used for calculating concept loss)
        concept_logits = self.concept_head(features)

        # Apply sigmoid to convert concept logits into probabilities (0 to 1)
        # This acts as the actual "bottleneck"
        concept_probs = torch.sigmoid(concept_logits)

        # Predict 'Smiling' using ONLY the 10 concept probabilities
        y_logits = self.label_head(concept_probs)

        return concept_logits, y_logits


class CBMTrainer:
    """
    A helper class to manage the training and evaluation loop of the CBM.
    Encapsulates the training logic, target evaluation (Accuracy, AUROC), 
    and concept evaluation (Accuracy, F1).
    """
    def __init__(self, model, device, trainloader, testloader):
        self.model = model
        self.device = device
        self.trainloader = trainloader
        self.testloader = testloader

    def train(self, epochs=5, lr=1e-3, pos_weight_tensor=None, lambda_c=5.0):
        """
        Executes the Joint Training loop.
        Computes the loss for both the concepts and the target, adding them together.
        """
        print(f"Starting Joint Training: {epochs} epochs...")
        
        # Use pos_weights for concepts if provided (helps with imbalanced CelebA classes)
        if pos_weight_tensor is not None:
            criterion_c = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(self.device))
        else:
            criterion_c = nn.BCEWithLogitsLoss()
            
        criterion_y = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for i, (images, true_concepts, true_labels) in enumerate(self.trainloader):
                images = images.to(self.device)
                true_concepts = true_concepts.to(self.device)
                true_labels = true_labels.to(self.device).unsqueeze(1)

                # Forward pass
                concept_logits, y_logits = self.model(images)

                # Calculate losses
                loss_c = criterion_c(concept_logits, true_concepts.float())
                loss_y = criterion_y(y_logits, true_labels.float())

                # Combine the losses (Joint Training)
                total_loss = loss_y + (lambda_c * loss_c)

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
                
                # Optional: Print progress every 500 batches
                if (i + 1) % 500 == 0:
                    print(f"  Epoch [{epoch+1}/{epochs}] Batch [{i+1}/{len(self.trainloader)}] Loss: {total_loss.item():.4f}")

            print(f"Epoch [{epoch+1}/{epochs}] Completed. Average Loss: {running_loss/len(self.trainloader):.4f}\n")

    def evaluate_target(self):
        """
        Evaluates the main classification task (Smiling).
        Calculates and prints Test Accuracy and Test AUROC.
        """
        print("Evaluating Target Task on the test set...")
        self.model.eval()

        correct_y = 0
        total_y = 0
        all_y_true = []
        all_y_probs = []

        with torch.no_grad():
            for images, _, true_labels in self.testloader:
                images = images.to(self.device)
                true_labels = true_labels.to(self.device).unsqueeze(1)

                _, y_logits = self.model(images)
                y_probs = torch.sigmoid(y_logits)
                y_preds = (y_probs > 0.5).float()

                correct_y += (y_preds == true_labels).sum().item()
                total_y += true_labels.size(0)

                all_y_true.extend(true_labels.cpu().numpy())
                all_y_probs.extend(y_probs.cpu().numpy())

        test_acc = correct_y / total_y
        test_auroc = roc_auc_score(all_y_true, all_y_probs)

        print(f"-----------------------------------\n")
        print(f"Target Task (Smiling) - Test Accuracy : {test_acc * 100:.2f}%")
        print(f"Target Task (Smiling) - Test AUROC    : {test_auroc:.4f}")
        print(f"-----------------------------------\n")

    def evaluate_concepts(self, concept_names):
        """
        Evaluates the model's ability to predict the 10 intermediate concepts.
        Calculates and prints individual and macro-average Accuracy and F1 Scores.
        """
        print("Evaluating Concept Predictions on the test set...")
        self.model.eval()

        all_c_true = []
        all_c_preds = []

        with torch.no_grad():
            for images, true_concepts, _ in self.testloader:
                images = images.to(self.device)
                
                concept_logits, _ = self.model(images)
                c_probs = torch.sigmoid(concept_logits)
                c_preds = (c_probs > 0.5).int()

                all_c_true.append(true_concepts.cpu().numpy())
                all_c_preds.append(c_preds.cpu().numpy())

        all_c_true = np.vstack(all_c_true)
        all_c_preds = np.vstack(all_c_preds)

        print("\n=== Concept Prediction Metrics ===")
        f1_scores = []
        accuracies = []

        for i, name in enumerate(concept_names):
            acc = accuracy_score(all_c_true[:, i], all_c_preds[:, i])
            f1 = f1_score(all_c_true[:, i], all_c_preds[:, i], zero_division=0)
            
            accuracies.append(acc)
            f1_scores.append(f1)
            print(f"{name:20s} -> Accuracy: {acc*100:.2f}% | F1 Score: {f1:.4f}")

        print("--------------------------------------------------")
        print(f"Mean Concept Accuracy (Macro): {np.mean(accuracies)*100:.2f}%")
        print(f"Mean Concept F1 Score (Macro): {np.mean(f1_scores):.4f}")
        print("--------------------------------------------------")
