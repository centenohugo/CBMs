"""
steerability.py

Steerability and Concept Interventions
Evaluate how controllable each model is through concept-level interventions.

Intervention procedure
For a set of test images:
    1. Compute the predicted concept vector
    2. For each concept, flip its value (0 -> 1 or 1 -> 0), keeping all others fixed
    3. Recompute the label prediction
    4. Measure:
        - the average change in predicted label probability, and
        - the fraction of samples for which the predicted label changes

Rank concepts according to their average intervention effect.
"""

import torch
import numpy as np
from tqdm import tqdm

# The 10 facial concepts required for the CelebA "Heavy Version"
CONCEPT_NAMES = [
    "Mouth_Slightly_Open", "High_Cheekbones", "Chubby", "Narrow_Eyes",
    "Bags_Under_Eyes", "Big_Lips", "Big_Nose", "Pointy_Nose",
    "Bushy_Eyebrows", "Arched_Eyebrows"
]

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
            
            # Binarize the base concepts to prevent soft-probability leakage during intervention
            c_binary_base = (c_soft_base > 0.5).float()
            
            # 2. Intervene on each concept separately
            for i in range(num_concepts):
                # Clone the hard binary concepts
                c_intervened = c_binary_base.clone()
                
                # Flip the concept value strictly: 0 -> 1 or 1 -> 0
                c_intervened[:, i] = 1.0 - c_intervened[:, i]
                
                # 3. Forward pass through c -> y (and side channel if Hybrid)
                if hasattr(model, 'side_channel'):
                    # Extract features using Hugo's backbone path
                    features = model.concept_predictor.backbone(x).view(x.size(0), -1)
                    
                    # Dropout is off in eval mode, so side channel is deterministic
                    s_x = model.side_channel(features)
                    f_c = model.c_to_y(c_intervened)
                    y_logits_new = f_c + s_x
                else:
                    y_logits_new = model.c_to_y(c_intervened)
                    
                y_prob_new = torch.sigmoid(y_logits_new)
                y_pred_new = (y_prob_new > 0.5).float()
                
                # 4. Record metrics
                prob_diff = torch.abs(y_prob_new - y_prob_base).cpu().numpy()
                flips = (y_pred_new != y_pred_base).cpu().numpy()
                
                intervention_effects[i]['prob_change'].extend(prob_diff.flatten())
                intervention_effects[i]['flips'] += flips.sum()
                intervention_effects[i]['total'] += len(flips)

    # Compile ranking with proper concept names
    ranking = []
    for i in range(num_concepts):
        avg_prob_change = np.mean(intervention_effects[i]['prob_change'])
        fraction_flipped = intervention_effects[i]['flips'] / intervention_effects[i]['total']
        
        # Guard clause in case fewer concepts are used
        concept_name = CONCEPT_NAMES[i] if i < len(CONCEPT_NAMES) else f"Concept_{i}"
        
        ranking.append({
            'concept_name': concept_name,
            'avg_prob_change': avg_prob_change,
            'fraction_flipped': fraction_flipped
        })
        
    # Sort by the highest average probability change
    ranking = sorted(ranking, key=lambda x: x['avg_prob_change'], reverse=True)
    return ranking


def print_steerability_ranking(ranking):
    """
    Formats and prints the intervention ranking for the final 2-3 page report.
    """
    print("--- STEERABILITY AND CONCEPT INTERVENTION RANKING ---")
    for rank, res in enumerate(ranking, 1):
        print(f"{rank}. {res['concept_name']}")
        print(f"   Avg. change in predicted label probability: {res['avg_prob_change']:.4f}")
        print(f"   Fraction of samples with changed label:     {res['fraction_flipped']:.2%}\n")