"""
train_unet_rl_ver.py - SIMP-Judge RL & Self-Adjoint Physics Fine-Tuning Pipeline for TopoUNet
Copy of train_unet.py upgraded with an RL-style SIMP Judge Environment:
  - Policy Network: TopoUNet (8.74M parameters, warm-started from checkpoints/best_unet_3000.pth)
  - Frozen Reference Policy: Keeps policy anchored to known-good topologies (PPO trust-region anchor)
  - Judge / Critic Environment: SIMPJudgeEnvironment solving 2D Q4 FEA (KU = F) in parallel across CPU cores,
    returning:
      1. Exact filtered self-adjoint compliance sensitivities dC/d_rho
      2. 1-step Optimality Criteria (OC) improved density targets rho_judge = OC_1(pred)
      3. Per-sample RL stability rewards R_i and advantage weights A_i
  - Thin-Strut Morphological Skeleton Weighting: 3.5x weight on thin internal web bars (fixes missing middle bars)

Hardware Footprint (NVIDIA GeForce RTX 3050 Laptop GPU + 12-Core i5-12450H CPU):
  - GPU VRAM: ~850 MB at batch_size=16 with AMP FP16 (fits comfortably in 4.0 GB VRAM)
  - CPU FEA Judge: 8 parallel worker threads via SciPy SuperLU/COLAMD (0 MB GPU VRAM)
"""

import argparse
import copy
import os
# Prevent BLAS/OpenMP thread oversubscription across the parallel FEA Judge worker threads
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from unet_model import TopoUNet, count_parameters
from losses_rl_ver import (
    TopoRLCompositeLoss,
    SIMPJudgeEnvironment,
    physics_aware_binarize
)
from dataset import get_dataloaders


def compute_metrics(preds, targets):
    """
    Computes structural topology metrics:
    - MSE (Mean Squared Error)
    - MAE (Mean Absolute Error)
    - Binary Accuracy (at threshold 0.35)
    - Binary Dice Score (at threshold 0.35)
    """
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()

    bin_preds = (preds >= 0.35).float()
    bin_targets = (targets >= 0.5).float()

    acc = (bin_preds == bin_targets).float().mean().item()

    intersection = (bin_preds * bin_targets).sum().item()
    total = bin_preds.sum().item() + bin_targets.sum().item()
    dice = (2.0 * intersection + 1e-6) / (total + 1e-6)

    return {'mse': mse, 'mae': mae, 'acc': acc, 'dice': dice}


def train_one_epoch_rl(model, ref_model, loader, criterion, judge_env, optimizer, scaler,
                       device, epoch, use_amp=True, judge_every_n=1):
    model.train()
    if ref_model is not None:
        ref_model.eval()

    running = {
        'total': 0.0, 'bce': 0.0, 'dice': 0.0, 'vf_mse': 0.0,
        'bin': 0.0, 'stiff': 0.0, 'edge': 0.0,
        'fea_judge': 0.0, 'judge_oc': 0.0, 'ref_anchor': 0.0,
        'fea_ratio': 0.0, 'reward': 0.0, 'batch_stable_pct': 0.0
    }
    judge_steps = 0

    for batch_idx, batch in enumerate(loader):
        if len(batch) == 3:
            inputs, targets, se = batch
            se = se.to(device, non_blocking=True)
        else:
            inputs, targets = batch
            se = None

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(inputs)
            if ref_model is not None:
                with torch.no_grad():
                    ref_preds = ref_model(inputs)
            else:
                ref_preds = None

        active_judge = judge_env if (judge_env is not None and (batch_idx % judge_every_n == 0)) else None

        loss, loss_dict = criterion(
            preds.float(),
            targets.float(),
            inputs.float(),
            epoch=epoch,
            strain_energy=se.float() if se is not None else None,
            raw_probs=model.last_raw_probs.float() if model.last_raw_probs is not None else None,
            judge_env=active_judge,
            ref_pred=ref_preds.float() if ref_preds is not None else None
        )

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        for k in ['total', 'bce', 'dice', 'vf_mse', 'bin', 'stiff', 'edge', 'fea_judge', 'judge_oc', 'ref_anchor']:
            running[k] += loss_dict.get(k, 0.0)

        if active_judge is not None:
            running['fea_ratio'] += loss_dict.get('fea_ratio', 1.0)
            running['reward'] += loss_dict.get('reward', 1.0)
            running['batch_stable_pct'] += loss_dict.get('batch_stable_pct', 100.0)
            judge_steps += 1

        if (batch_idx + 1) % 40 == 0 or (batch_idx + 1) == len(loader):
            n_j = max(1, judge_steps)
            print(f"    [Epoch {epoch:02d} | Batch {batch_idx+1:03d}/{len(loader):03d}] "
                  f"Loss={running['total']/(batch_idx+1):.4f} | "
                  f"Judge Ratio={running['fea_ratio']/n_j:.2f}x | "
                  f"Reward={running['reward']/n_j:.3f} | "
                  f"Batch Stable={running['batch_stable_pct']/n_j:.1f}%",
                  flush=True)

    n_batches = max(1, len(loader))
    n_judge = max(1, judge_steps)
    summary = {
        k: running[k] / n_batches
        for k in ['total', 'bce', 'dice', 'vf_mse', 'bin', 'stiff', 'edge', 'fea_judge', 'judge_oc', 'ref_anchor']
    }
    summary['fea_ratio'] = running['fea_ratio'] / n_judge
    summary['reward'] = running['reward'] / n_judge
    summary['batch_stable_pct'] = running['batch_stable_pct'] / n_judge
    return summary


def evaluate_rl(model, loader, criterion, judge_env, device, epoch, use_amp=True):
    model.eval()
    running = {
        'total': 0.0, 'bce': 0.0, 'dice': 0.0, 'vf_mse': 0.0,
        'bin': 0.0, 'stiff': 0.0, 'edge': 0.0,
        'fea_judge': 0.0, 'judge_oc': 0.0, 'ref_anchor': 0.0,
        'fea_ratio': 0.0, 'reward': 0.0, 'batch_stable_pct': 0.0
    }

    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 3:
                inputs, targets, se = batch
                se = se.to(device, non_blocking=True)
            else:
                inputs, targets = batch
                se = None

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preds = model(inputs)

            loss, loss_dict = criterion(
                preds.float(),
                targets.float(),
                inputs.float(),
                epoch=epoch,
                strain_energy=se.float() if se is not None else None,
                raw_probs=model.last_raw_probs.float() if model.last_raw_probs is not None else None,
                judge_env=judge_env,
                ref_pred=None
            )

            for k in running:
                running[k] += loss_dict.get(k, 0.0)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    n_batches = max(1, len(loader))
    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(preds_tensor, targets_tensor)

    loss_summary = {k: running[k] / n_batches for k in running}
    metrics['val_fea_ratio'] = loss_summary['fea_ratio']
    metrics['val_reward'] = loss_summary['reward']
    metrics['val_stable_pct'] = loss_summary['batch_stable_pct']
    return loss_summary, metrics


def plot_rl_training_curves(history, save_path):
    epochs = [h['epoch'] for h in history]
    train_total = [h['train_loss']['total'] for h in history]
    val_total = [h['val_loss']['total'] if h['val_loss'] else None for h in history]
    val_dice = [h['val_metrics']['dice'] if h['val_metrics'] else None for h in history]
    val_ratio = [h['val_metrics']['val_fea_ratio'] if h['val_metrics'] else None for h in history]
    val_stable = [h['val_metrics']['val_stable_pct'] if h['val_metrics'] else None for h in history]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), dpi=160)

    # 1. Total Loss
    axes[0].plot(epochs, train_total, 'b-o', label='Train RL Loss', lw=2)
    if val_total[0] is not None:
        axes[0].plot(epochs, val_total, 'r--s', label='Val RL Loss', lw=2)
    axes[0].set_title('1. SIMP-Judge RL Composite Loss', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    # 2. Validation FEA Compliance Ratio (Lower = Stiffer)
    if val_ratio[0] is not None:
        axes[1].plot(epochs, val_ratio, 'm-o', label='Val FEA Ratio ($C_{pred}/C^*$)', lw=2)
        axes[1].axhline(1.0, color='green', linestyle='--', lw=1.5, label='SIMP Parity (1.0x)')
        axes[1].set_title('2. Judge FEA Compliance Ratio', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Ratio $C_{pred} / C^*$')
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend()

    # 3. Validation Structural Stability Accuracy (%)
    if val_stable[0] is not None:
        axes[2].plot(epochs, val_stable, 'g-s', label='Val Stiffness Stability (%)', lw=2)
        axes[2].set_ylim(50, 102)
        axes[2].set_title('3. Validation Stability Pass Rate (%)', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Stable Structures (%)')
        axes[2].grid(True, linestyle=':', alpha=0.6)
        axes[2].legend()

    # 4. Validation Dice Score
    if val_dice[0] is not None:
        axes[3].plot(epochs, val_dice, 'c-^', label='Val Dice Overlap', lw=2)
        axes[3].set_ylim(0.5, 1.0)
        axes[3].set_title('4. Validation Dice Similarity', fontsize=11, fontweight='bold')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Dice Score')
        axes[3].grid(True, linestyle=':', alpha=0.6)
        axes[3].legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def train_rl(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    use_amp = (device.type == 'cuda') and not args.no_amp

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA GPU Enabled: {gpu_name} ({gpu_mem:.2f} GB VRAM) | AMP FP16: {use_amp}", flush=True)

    print("=" * 78, flush=True)
    print("  TopoUNet SIMP-Judge RL & Self-Adjoint Fine-Tuning Pipeline (RL Version)", flush=True)
    print(f"  Device:             {device}", flush=True)
    print(f"  Dataset:            {args.dataset}", flush=True)
    print(f"  Warm-Start Ckpt:    {args.resume}", flush=True)
    print(f"  Output Ckpt:        {os.path.join(args.save_dir, args.checkpoint_name)}", flush=True)
    print(f"  RL Epochs:          {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}", flush=True)
    print(f"  SIMP Judge Workers: {args.judge_workers} CPU threads | Judge Freq: every {args.judge_every_n} batch(es)", flush=True)
    print(f"  Weights:            w_fea={args.w_fea}, w_judge_oc={args.w_judge_oc}, w_thin={args.w_thin_strut}, beta_bin={args.beta_bin}", flush=True)
    print("=" * 78, flush=True)

    os.makedirs(args.save_dir, exist_ok=True)

    pin_mem = (device.type == 'cuda')
    train_loader, val_loader = get_dataloaders(
        data_path=args.dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        augment=not args.no_augment,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        return_strain_energy=True
    )

    n_train_samples = len(train_loader.dataset)
    n_val_samples = len(val_loader.dataset) if val_loader else 0
    print(f"DataLoaders ready: {n_train_samples} train, {n_val_samples} val samples.", flush=True)

    # Initialize policy model
    model = TopoUNet(in_channels=6, out_channels=1, base_filters=args.base_filters)
    if args.resume and os.path.exists(args.resume):
        print(f"Loading warm-start policy weights from: {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print("Notice: Warm-start checkpoint not found; initializing from scratch.", flush=True)

    model.to(device)
    _, train_p = count_parameters(model)

    # Create frozen reference policy for PPO-style trust-region regularization (~35 MB VRAM)
    ref_model = None
    if args.w_ref_anchor > 0.0 and args.resume and os.path.exists(args.resume):
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        print(f"Policy & Frozen Reference Policy initialized ({train_p:,} params).", flush=True)

    # Initialize parallel CPU SIMP-Judge Environment
    judge_env = SIMPJudgeEnvironment(
        nelx=120, nely=60, penal=3.0, num_workers=args.judge_workers
    )

    criterion = TopoRLCompositeLoss(
        w_bce=args.w_bce,
        w_dice=args.w_dice,
        alpha_vf=args.alpha_vf,
        beta_bin=args.beta_bin,
        w_stiff=args.w_stiff,
        w_edge=args.w_edge,
        w_fea=args.w_fea,
        w_judge_oc=args.w_judge_oc,
        w_ref_anchor=args.w_ref_anchor,
        w_thin_strut=args.w_thin_strut
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if device.type == 'cuda' else None

    best_score = float('inf')
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        train_loss = train_one_epoch_rl(
            model=model,
            ref_model=ref_model,
            loader=train_loader,
            criterion=criterion,
            judge_env=judge_env,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            judge_every_n=args.judge_every_n
        )

        if val_loader is not None:
            val_loss, val_metrics = evaluate_rl(
                model=model,
                loader=val_loader,
                criterion=criterion,
                judge_env=judge_env,
                device=device,
                epoch=epoch,
                use_amp=use_amp
            )
        else:
            val_loss, val_metrics = None, None

        scheduler.step()
        ep_time = time.time() - t_ep

        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'lr': scheduler.get_last_lr()[0],
            'time_sec': ep_time
        }
        history.append(record)

        val_str = ""
        if val_loss and val_metrics:
            val_str = (f" | Val: Ratio={val_metrics['val_fea_ratio']:.2f}x "
                       f"Stable={val_metrics['val_stable_pct']:.1f}% "
                       f"Dice={val_metrics['dice']:.4f} R={val_metrics['val_reward']:.3f}")

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({ep_time:.1f}s) | "
              f"Train: Loss={train_loss['total']:.4f} Ratio={train_loss['fea_ratio']:.2f}x "
              f"R={train_loss['reward']:.3f} Stable={train_loss['batch_stable_pct']:.1f}%{val_str}",
              flush=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'val_loss': val_loss
        }

        latest_rl_path = os.path.join(args.save_dir, "latest_unet_rl.pth")
        torch.save(checkpoint, latest_rl_path)

        # Selection criterion combines validation FEA compliance ratio and total RL loss
        current_score = (val_loss['total'] + 0.5 * val_metrics['val_fea_ratio']) if (val_loss and val_metrics) else train_loss['total']
        if current_score < best_score:
            best_score = current_score
            best_rl_path = os.path.join(args.save_dir, args.checkpoint_name)
            best_ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'val_loss': val_loss
            }
            torch.save(best_ckpt, best_rl_path)
            print(f"  --> [Saved New Best RL Model] {best_rl_path} (Score: {best_score:.4f})", flush=True)

    total_time = time.time() - t_start
    judge_env.close()
    print("=" * 78, flush=True)
    print(f"SIMP-Judge RL Fine-Tuning completed in {total_time:.1f}s ({total_time/60:.2f} min).", flush=True)
    print(f"Best RL Checkpoint: {os.path.join(args.save_dir, args.checkpoint_name)}", flush=True)

    curves_path = os.path.join(args.save_dir, "training_curves_rl.png")
    plot_rl_training_curves(history, curves_path)

    history_path = os.path.join(args.save_dir, "training_history_rl.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="SIMP-Judge RL Fine-Tuning for TopoUNet")
    parser.add_argument("--dataset", type=str, default="topo_dataset_3000.npz", help="Path to .npz dataset")
    parser.add_argument("--resume", type=str, default="checkpoints/best_unet_3000.pth", help="Warm-start checkpoint")
    parser.add_argument("--epochs", type=int, default=8, help="Number of RL fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size (16 uses <1 GB VRAM on RTX 3050)")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Fine-tuning learning rate")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--base_filters", type=int, default=64, help="Base filter width for U-Net")
    parser.add_argument("--w_bce", type=float, default=1.0, help="Weight for mechanics-weighted BCE")
    parser.add_argument("--w_dice", type=float, default=0.6, help="Weight for multi-scale eroded Dice")
    parser.add_argument("--alpha_vf", type=float, default=10.0, help="Weight for Volume Fraction penalty")
    parser.add_argument("--beta_bin", type=float, default=0.03, help="Weight for Binarization sharpness penalty")
    parser.add_argument("--w_stiff", type=float, default=0.6, help="Weight for SIMP cubic stiffness loss")
    parser.add_argument("--w_edge", type=float, default=0.25, help="Weight for Sobel edge alignment loss")
    parser.add_argument("--w_fea", type=float, default=0.35, help="Weight for Self-Adjoint FEA Judge gradient")
    parser.add_argument("--w_judge_oc", type=float, default=0.50, help="Weight for 1-step SIMP OC Judge distillation")
    parser.add_argument("--w_ref_anchor", type=float, default=0.20, help="Weight for PPO reference policy anchor")
    parser.add_argument("--w_thin_strut", type=float, default=3.5, help="Weight multiplier for thin internal web struts")
    parser.add_argument("--judge_workers", type=int, default=8, help="Parallel CPU threads for SIMP FEA Judge")
    parser.add_argument("--judge_every_n", type=int, default=1, help="Run FEA judge every N batches (1 = every batch)")
    parser.add_argument("--no_augment", action="store_true", help="Disable spatial augmentations")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP FP16 mixed precision")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint save directory")
    parser.add_argument("--checkpoint_name", type=str, default="best_unet_rl.pth", help="Output RL checkpoint filename")

    args = parser.parse_args()
    train_rl(args)


if __name__ == '__main__':
    main()
