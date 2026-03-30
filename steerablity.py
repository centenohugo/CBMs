"""
steerability.py

Steerability and Concept Interventions
Evaluate how controllable each model is through concept-level interventions.

Intervention procedure
For a set of test images:
    1. Compute the predicted concept vector
    2. For each concept, flip its value (0 → 1 or 1 → 0), keeping all others fixed
    3. Recompute the label prediction
    4. Measure:
        - the average change in predicted label probability, and
        - the fraction of samples for which the predicted label changes

Rank concepts according to their average intervention effect.

"""
import torch
import numpy as np
from tqdm import tqdm

def evaluate_steerability(model, dataloader, device, num_concepts=10):
    """
    Evaluates steerability by flipping one concept at a time and 
    measuring the effect on the final target prediction.
    """
    model.eval()
    intervention_effects = {i: {'prob_change': [], 'flips': 0, 'total': 0} for i in range(num_concepts)}
    
    with torch.no_grad():
        for x, c, y in tqdm(dataloader, desc="Intervening"):
            x = x.to(device)
            
            # 1. Base prediction
            c_logits_base, y_logits_base = model(x)
            y_prob_base = torch.sigmoid(y_logits_base)
            y_pred_base = (y_prob_base > 0.5).float()
            
            c_soft_base = torch.sigmoid(c_logits_base)
            
            # 2. Intervene on each concept separately
            for i in range(num_concepts):
                c_intervened = c_soft_base.clone()
                
                # Flip the concept value: if >0.5 it becomes 0, else 1
                c_intervened[:, i] = 1.0 - torch.round(c_intervened[:, i])
                
                # Forward pass through c -> y (and side channel if Hybrid)
                if hasattr(model, 'side_channel'):
                    features = model.concept_predictor.backbone(x).view(x.size(0), -1)
                    # Dropout is off in eval mode, so side channel is deterministic
                    s_x = model.side_channel(features)
                    f_c = model.c_to_y(c_intervened)
                    y_logits_new = f_c + s_x
                else:
                    y_logits_new = model.c_to_y(c_intervened)
                    
                y_prob_new = torch.sigmoid(y_logits_new)
                y_pred_new = (y_prob_new > 0.5).float()
                
                # Record metrics
                prob_diff = torch.abs(y_prob_new - y_prob_base).cpu().numpy()
                flips = (y_pred_new != y_pred_base).cpu().numpy()
                
                intervention_effects[i]['prob_change'].extend(prob_diff.flatten())
                intervention_effects[i]['flips'] += flips.sum()
                intervention_effects[i]['total'] += len(flips)

    # Compile ranking
    ranking = []
    for i in range(num_concepts):
        avg_prob_change = np.mean(intervention_effects[i]['prob_change'])
        fraction_flipped = intervention_effects[i]['flips'] / intervention_effects[i]['total']
        ranking.append({
            'concept_idx': i,
            'avg_prob_change': avg_prob_change,
            'fraction_flipped': fraction_flipped
        })
        
    ranking = sorted(ranking, key=lambda x: x['avg_prob_change'], reverse=True)
    return ranking