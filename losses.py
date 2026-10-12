"""
losses.py - Physics-Informed Composite Loss Functions for Topology Optimization

Implements the multi-objective loss formulated in handover.md:
  L = w_bce * L_bce + w_dice * L_dice + alpha * L_vf + beta * L_bin

1. Binary Cross Entropy (L_bce): Pixel-wise ground-truth material matching.
2. Soft Dice Loss (L_dice): Structural connectivity & overlap penalizing topological disconnects.
3. Volume Fraction Penalty (L_vf): Penalizes deviation from the prescribed volume fraction Vf.
4. Binarization Regularizer (L_bin): Penalizes intermediate gray densities, driving crisp 0/1 members.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for continuous density structural overlap.
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred and target shapes: (B, 1, H, W)
        pred_flat = pred.contiguous().view(pred.size(0), -1)
        target_flat = target.contiguous().view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class TopoCompositeLoss(nn.Module):
    """
    Composite Physics-Guided Loss for U-Net Topology Optimization.
    
    Args:
        w_bce (float): Weight for binary cross entropy (default: 1.0)
        w_dice (float): Weight for structural Dice loss (default: 0.5)
        alpha_vf (float): Weight for volume fraction constraint (default: 10.0)
        beta_bin (float): Initial weight for binarization grayness penalty (default: 1e-3)
    """
    def __init__(self, w_bce=1.0, w_dice=0.5, alpha_vf=10.0, beta_bin=1e-3):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.alpha_vf = alpha_vf
        self.beta_bin = beta_bin
        
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target, inputs, epoch=1):
        """
        pred:   (B, 1, H, W) predicted density in [0, 1]
        target: (B, 1, H, W) ground-truth SIMP density in [0, 1]
        inputs: (B, 5, H, W) input tensor; Channel 4 contains target Vf
        epoch:  current training epoch (for scheduling binarization weight)
        """
        # 1. Pixel-level BCE loss
        loss_bce = self.bce(pred, target)
        
        # 2. Structural Dice loss
        loss_dice = self.dice(pred, target)
        
        # 3. Volume fraction constraint loss
        # Target Vf is encoded in Channel 4 of inputs (B, 1, H, W or B, 5, H, W)
        target_vf = inputs[:, 4:5, :, :].mean(dim=(-2, -1)) # shape: (B, 1)
        pred_vf = pred.mean(dim=(-2, -1))                   # shape: (B, 1)
        loss_vf = F.mse_loss(pred_vf, target_vf)
        
        # 4. Binarization penalty: 4 * rho * (1 - rho) is 0 at rho in {0, 1}, 1 at rho = 0.5
        # Soft ramp-up after epoch 5 to avoid restricting early feature discovery
        beta_eff = self.beta_bin * min(1.0, max(0.0, (epoch - 5) / 10.0))
        loss_bin = torch.mean(4.0 * pred * (1.0 - pred))
        
        total_loss = (
            self.w_bce * loss_bce +
            self.w_dice * loss_dice +
            self.alpha_vf * loss_vf +
            beta_eff * loss_bin
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'bce': loss_bce.item(),
            'dice': loss_dice.item(),
            'vf_mse': loss_vf.item(),
            'bin': loss_bin.item()
        }
        
        return total_loss, loss_dict


if __name__ == '__main__':
    loss_fn = TopoCompositeLoss()
    pred = torch.rand(4, 1, 60, 120)
    target = torch.randint(0, 2, (4, 1, 60, 120)).float()
    inputs = torch.zeros(4, 5, 60, 120)
    inputs[:, 4, :, :] = 0.4  # target volume fraction
    
    total, d = loss_fn(pred, target, inputs, epoch=1)
    print("Loss test succeeded!")
    for k, v in d.items():
        print(f"  {k}: {v:.6f}")
