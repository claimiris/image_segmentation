# Loss Functions for Medical Image Segmentation
<<<<<<< HEAD

=======
>>>>>>> e8769cf (Prepare local files for synchronization)
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dice Loss - measures overlap between predicted mask and ground truth
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        tgt   = targets.float().view(targets.size(0), -1)
        inter = (probs * tgt).sum(dim=1)
        union = probs.sum(dim=1) + tgt.sum(dim=1)
        return 1.0 - ((2*inter + self.smooth) / (union + self.smooth)).mean()

# Focal Loss - addresses class imbalance by focusing on hard examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt  = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - pt)**self.gamma * bce).mean()

# Hybrid Loss - combines dice and focal losses
class HybridLoss(nn.Module):
    """Dice + Focal hybrid loss."""
    def __init__(self, dice_w=0.5, focal_w=0.5):
        super().__init__()
        self.dice_loss  = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dw, self.fw = dice_w, focal_w

    def forward(self, logits, targets):
        dl = self.dice_loss(logits, targets)
        fl = self.focal_loss(logits, targets)
        return self.dw * dl + self.fw * fl, dl, fl

# Deep Supervision Loss - helps to improve gradient flow and training stability
class DeepSupervisionLoss(nn.Module):
    """Weighted loss over multi-output heads (for UNet++)."""
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or [0.1, 0.2, 0.3, 0.4]
        self.loss_fn = HybridLoss()

    def forward(self, outputs, targets):
        if not isinstance(outputs, (list, tuple)):
            loss, _, _ = self.loss_fn(outputs, targets)
            return loss

        total = 0.0
        for out, w in zip(outputs, self.weights):
            loss, _, _ = self.loss_fn(out, targets)
            total += w * loss
        return total
