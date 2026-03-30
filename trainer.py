"""
trainer.py
Training loops and evaluation functions for all architectures.
"""
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm

def evaluate_model(model, dataloader, device, is_cbm=False):
    model.eval()
    all_y_true, all_y_pred, all_y_probs = [], [], []
    all_c_true, all_c_pred = [], []
    
    with torch.no_grad():
        for x, c, y in dataloader:
            x, c, y = x.to(device), c.to(device), y.to(device)
            
            if is_cbm:
                c_logits, y_logits = model(x)
                c_probs = torch.sigmoid(c_logits)
                all_c_true.append(c.cpu())
                all_c_pred.append((c_probs > 0.5).float().cpu())
            else:
                y_logits = model(x)
                
            y_probs = torch.sigmoid(y_logits)
            all_y_probs.append(y_probs.cpu())
            all_y_pred.append((y_probs > 0.5).float().cpu())
            all_y_true.append(y.cpu())

    y_true = torch.cat(all_y_true).numpy()
    y_pred = torch.cat(all_y_pred).numpy()
    y_probs = torch.cat(all_y_probs).numpy()
    
    metrics = {
        'test_acc': accuracy_score(y_true, y_pred),
        'test_auroc': roc_auc_score(y_true, y_probs)
    }

    if is_cbm:
        c_true = torch.cat(all_c_true).numpy()
        c_pred = torch.cat(all_c_pred).numpy()
        metrics['concept_f1_macro'] = f1_score(c_true, c_pred, average='macro')
        metrics['concept_acc'] = accuracy_score(c_true.flatten(), c_pred.flatten())

    return metrics

def train_epoch(model, dataloader, optimizer, device, is_cbm=False, joint_loss_weight=1.0):
    model.train()
    criterion_bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for x, c, y in tqdm(dataloader, desc="Training"):
        x, c, y = x.to(device), c.to(device), y.unsqueeze(1).to(device)
        optimizer.zero_grad()

        if is_cbm:
            c_logits, y_logits = model(x)
            loss_c = criterion_bce(c_logits, c)
            loss_y = criterion_bce(y_logits, y)
            loss = loss_y + joint_loss_weight * loss_c
        else:
            y_logits = model(x)
            loss = criterion_bce(y_logits, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)