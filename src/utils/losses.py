import torch
import torch.nn as nn

class WeightedBCEWithLogits(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for multi-label classification.
    Designed to combat minority class data scarcity by penalizing false negatives
    on rare bioplastics via a customizable positive scaling vector.
    """
    def __init__(self, pos_weight=None):
        super(WeightedBCEWithLogits, self). __init__()
        self.pos_weight = pos_weight
        self.eps = 1e-12

    def forward(self, logits, targets):
        """
        Formulation:
        L = -1/N * sum(w_pos * y * log(p) + (1 - y) * log(1 - p))
        """
        # Note: Since the model output ends with Sigmoid, we expect probabilities.
        # However, for numerical stability, BCEWithLogits is usually preferred.
        # If the model ends in Sigmoid, we use the probability directly:
        probs = torch.clamp(logits, self.eps, 1.0 - self.eps)

        if self.pos_weight is not None:
            # Applying positive scaling vector
            loss = -(self.pos_weight * targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
        else:
            loss = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))

        return torch.mean(loss)

def get_loss_function(use_wbce=True, pos_weights=None):
    """
    Factory function to switch between classic BCE and WBCE.
    """
    if use_wbce and pos_weights is not None:
        return WeightedBCEWithLogits(pos_weight=pos_weights)
    else:
        return WeightedBCEWithLogits(pos_weight=None)
