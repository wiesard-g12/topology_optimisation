"""
train_unet.py - Production Training Pipeline for TopoUNet (Phase 2)
Trains the 4-level U-Net on SIMP-generated datasets using the physics-guided composite loss.

Key Features:
- Physics-guided multi-objective loss (BCE + Soft Dice + Volume Penalty + Binarization)
- Automatic train / validation split
- Physics-consistent spatial data augmentation
- Cosine Annealing learning rate schedule
- Best checkpoint saving based on validation Dice score / Total loss
- Automatic export of training loss curves and history logs
"""

import argparse
import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from unet_model import TopoUNet, count_parameters
from losses import TopoCompositeLoss
from dataset import get_dataloaders


def compute_metrics(preds, targets):
    """
    Computes structural topology metrics:
    - MSE (Mean Squared Error)
    - MAE (Mean Absolute Error)
    - Binary Accuracy (at threshold 0.5)
    - Binary Dice Score (at threshold 0.5)
    """
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    
    bin_preds = (preds >= 0.5).float()
    bin_targets = (targets >= 0.5).float()
    
    acc = (bin_preds == bin_targets).float().mean().item()
    
    intersection = (bin_preds * bin_targets).sum().item()
    total = bin_preds.sum().item() + bin_targets.sum().item()
    dice = (2.0 * intersection + 1e-6) / (total + 1e-6)
    
    return {'mse': mse, 'mae': mae, 'acc': acc, 'dice': dice}


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    running_vf = 0.0
    running_bin = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        preds = model(inputs)
        
        loss, loss_dict = criterion(preds, targets, inputs, epoch=epoch)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        running_loss += loss.item()
        running_bce += loss_dict['bce']
        running_dice += loss_dict['dice']
        running_vf += loss_dict['vf_mse']
        running_bin += loss_dict['bin']
        
    n_batches = len(loader)
    return {
        'total': running_loss / n_batches,
        'bce': running_bce / n_batches,
        'dice': running_dice / n_batches,
        'vf_mse': running_vf / n_batches,
        'bin': running_bin / n_batches
    }


def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    running_vf = 0.0
    running_bin = 0.0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            preds = model(inputs)
            loss, loss_dict = criterion(preds, targets, inputs, epoch=epoch)
            
            running_loss += loss.item()
            running_bce += loss_dict['bce']
            running_dice += loss_dict['dice']
            running_vf += loss_dict['vf_mse']
            running_bin += loss_dict['bin']
            
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            
    n_batches = len(loader)
    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(preds_tensor, targets_tensor)
    
    loss_summary = {
        'total': running_loss / n_batches,
        'bce': running_bce / n_batches,
        'dice': running_dice / n_batches,
        'vf_mse': running_vf / n_batches,
        'bin': running_bin / n_batches
    }
    return loss_summary, metrics


def plot_training_curves(history, save_path):
    epochs = [h['epoch'] for h in history]
    train_total = [h['train_loss']['total'] for h in history]
    val_total = [h['val_loss']['total'] if h['val_loss'] else None for h in history]
    val_dice = [h['val_metrics']['dice'] if h['val_metrics'] else None for h in history]
    val_mse = [h['val_metrics']['mse'] if h['val_metrics'] else None for h in history]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    
    # 1. Total Loss
    axes[0].plot(epochs, train_total, 'b-', label='Train Loss', lw=2)
    if val_total[0] is not None:
        axes[0].plot(epochs, val_total, 'r--', label='Val Loss', lw=2)
    axes[0].set_title('Composite Optimization Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()
    
    # 2. Validation Dice Score
    if val_dice[0] is not None:
        axes[1].plot(epochs, val_dice, 'g-', label='Val Dice', lw=2)
        axes[1].set_title('Validation Dice Similarity', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score (0 to 1)')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend()
        
    # 3. Validation MSE
    if val_mse[0] is not None:
        axes[2].plot(epochs, val_mse, 'm-', label='Val MSE', lw=2)
        axes[2].set_title('Validation Mean Squared Error', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('MSE')
        axes[2].grid(True, linestyle=':', alpha=0.6)
        axes[2].legend()
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    if device.type == 'cpu':
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        print(f"PyTorch CPU Multithreading enabled: {num_threads} worker threads.", flush=True)
        
    print("=" * 65, flush=True)
    print(f"  TopoUNet Training Pipeline (Phase 2)", flush=True)
    print(f"  Device:           {device}", flush=True)
    print(f"  Dataset:          {args.dataset}", flush=True)
    print(f"  Epochs:           {args.epochs}", flush=True)
    print(f"  Batch Size:       {args.batch_size}", flush=True)
    print(f"  Learning Rate:    {args.lr}", flush=True)
    print(f"  Volume Penalty a: {args.alpha_vf}", flush=True)
    print(f"  Binarization b:   {args.beta_bin}", flush=True)
    print("=" * 65, flush=True)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    train_loader, val_loader = get_dataloaders(
        data_path=args.dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        augment=not args.no_augment
    )
    
    print(f"DataLoaders ready: {len(train_loader.dataset)} training samples, "
          f"{len(val_loader.dataset) if val_loader else 0} validation samples.", flush=True)
    
    # Model
    model = TopoUNet(in_channels=5, out_channels=1, base_filters=args.base_filters)
    model.to(device)
    total_p, train_p = count_parameters(model)
    print(f"TopoUNet Architecture Initialized: {train_p:,} trainable parameters.", flush=True)
    
    # Loss & Optimizer
    criterion = TopoCompositeLoss(
        w_bce=args.w_bce,
        w_dice=args.w_dice,
        alpha_vf=args.alpha_vf,
        beta_bin=args.beta_bin
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    best_val_loss = float('inf')
    best_dice = 0.0
    history = []
    
    t_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        if val_loader is not None:
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device, epoch)
        else:
            val_loss, val_metrics = None, None
            
        scheduler.step()
        epoch_time = time.time() - t_epoch_start
        
        # Save metrics
        epoch_record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'lr': scheduler.get_last_lr()[0],
            'time_sec': epoch_time
        }
        history.append(epoch_record)
        
        # Terminal status output
        val_str = ""
        if val_loss and val_metrics:
            val_str = (f" | Val: total={val_loss['total']:.4f} "
                       f"Dice={val_metrics['dice']:.4f} MSE={val_metrics['mse']:.4f}")
            
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"({epoch_time:.1f}s) | Train: total={train_loss['total']:.4f} "
              f"BCE={train_loss['bce']:.4f} Vf_MSE={train_loss['vf_mse']:.5f}{val_str}", flush=True)
        
        # Checkpointing
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'val_loss': val_loss
        }
        
        # Save latest
        latest_path = os.path.join(args.save_dir, "latest_unet.pth")
        torch.save(checkpoint, latest_path)
        
        # Save best model
        current_loss = val_loss['total'] if val_loss else train_loss['total']
        if current_loss < best_val_loss:
            best_val_loss = current_loss
            best_path = os.path.join(args.save_dir, "best_unet.pth")
            torch.save(checkpoint, best_path)
            
    total_time = time.time() - t_start
    print("=" * 65)
    print(f"Training completed in {total_time:.1f}s ({total_time/60:.2f} min).")
    print(f"Best Model Checkpoint: {os.path.join(args.save_dir, 'best_unet.pth')}")
    
    # Save training curves & history log
    curves_path = os.path.join(args.save_dir, "training_curves.png")
    plot_training_curves(history, curves_path)
    print(f"Loss curves saved to: {curves_path}")
    
    history_path = os.path.join(args.save_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History log saved to: {history_path}")


def main():
    parser = argparse.ArgumentParser(description="Train TopoUNet for Real-Time Topology Optimization")
    parser.add_argument("--dataset", type=str, default="topo_dataset.npz", help="Path to .npz dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation set split fraction")
    parser.add_argument("--base_filters", type=int, default=64, help="Base filter width for U-Net")
    parser.add_argument("--w_bce", type=float, default=1.0, help="Weight for BCE loss")
    parser.add_argument("--w_dice", type=float, default=0.5, help="Weight for Dice loss")
    parser.add_argument("--alpha_vf", type=float, default=10.0, help="Weight for Volume Fraction penalty")
    parser.add_argument("--beta_bin", type=float, default=1e-3, help="Weight for Binarization penalty")
    parser.add_argument("--no_augment", action="store_true", help="Disable physics augmentations")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
