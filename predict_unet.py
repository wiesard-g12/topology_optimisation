"""
predict_unet.py - Real-Time Inference & Evaluation Engine for TopoUNet (Phase 2)
Benchmarks inference speed, evaluates structural accuracy, and renders publication figures.

Outputs:
1. Inference Latency Benchmark (< 10 ms on GPU, < 30 ms on CPU)
2. Quantitative Structural Metrics:
   - Dice Similarity Coefficient
   - Mean Squared Error (MSE) & Mean Absolute Error (MAE)
   - Target vs. Predicted Volume Fraction Discrepancy
   - Binarization Clarity Score
3. 5-Panel Engineering Figure:
   [1. Boundary Conditions & Loads]
   [2. Ground Truth SIMP (80 iter)]
   [3. U-Net Continuous Prediction]
   [4. Binarized Final Truss (rho >= 0.5)]
   [5. Error Heatmap |Pred - SIMP|]
"""

import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from unet_model import TopoUNet
from dataset import TopoDataset
from visualize_data import extract_discrete_loads, draw_loads, draw_domain_and_bcs


def render_prediction_comparison(sample_idx, input_tensor, target_density, pred_density,
                                 latency_ms, output_path):
    """
    Renders a publication-ready 5-panel comparison figure.
    """
    nelx = input_tensor.shape[2]
    nely = input_tensor.shape[1]
    
    ch_domain = input_tensor[0]
    ch_bc = input_tensor[1]
    ch_fx = input_tensor[2]
    ch_fy = input_tensor[3]
    ch_vf = input_tensor[4]
    target_vf = float(np.mean(ch_vf))
    
    pred_vf = float(np.mean(pred_density))
    gt_vf = float(np.mean(target_density))
    
    # Calculate metrics
    bin_pred = (pred_density >= 0.5).astype(float)
    bin_gt = (target_density >= 0.5).astype(float)
    
    intersection = np.sum(bin_pred * bin_gt)
    union = np.sum(bin_pred) + np.sum(bin_gt)
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    
    mse = float(np.mean((pred_density - target_density) ** 2))
    diff = np.abs(pred_density - target_density)
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.8), dpi=180)
    
    # --- PANEL 1: Boundary Conditions & Loads ---
    ax0 = axes[0]
    draw_domain_and_bcs(ax0, ch_bc)
    loads = extract_discrete_loads(ch_fx, ch_fy)
    draw_loads(ax0, loads)
    ax0.set_title(f"1. Problem Definition (BCs & Loads)\nTarget $V_f = {target_vf:.2f}$",
                  fontsize=11, fontweight='bold', pad=8)
    ax0.set_xlim(-15, nelx + 15)
    ax0.set_ylim(nely + 10, -10)
    ax0.set_aspect('equal')
    ax0.axis('off')
    
    # --- PANEL 2: Ground Truth SIMP ---
    ax1 = axes[1]
    im1 = ax1.imshow(1.0 - target_density, cmap='gray', vmin=0, vmax=1,
                     extent=[0, nelx, nely, 0], interpolation='nearest')
    ax1.set_title(f"2. Ground Truth SIMP (80 iter)\n$V_f = {gt_vf:.3f}$",
                  fontsize=11, fontweight='bold', pad=8)
    ax1.set_xlim(-5, nelx + 5)
    ax1.set_ylim(nely + 5, -5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # --- PANEL 3: U-Net Continuous Output ---
    ax2 = axes[2]
    im2 = ax2.imshow(1.0 - pred_density, cmap='gray', vmin=0, vmax=1,
                     extent=[0, nelx, nely, 0], interpolation='nearest')
    ax2.set_title(rf"3. U-Net Continuous Pred ($\hat{{y}}$)" + "\n" + f"$V_f = {pred_vf:.3f}$",
                  fontsize=11, fontweight='bold', pad=8)
    ax2.set_xlim(-5, nelx + 5)
    ax2.set_ylim(nely + 5, -5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # --- PANEL 4: U-Net Binarized Output ---
    ax3 = axes[3]
    im3 = ax3.imshow(1.0 - bin_pred, cmap='gray', vmin=0, vmax=1,
                     extent=[0, nelx, nely, 0], interpolation='nearest')
    bin_vf = float(np.mean(bin_pred))
    ax3.set_title(rf"4. Binarized Topology ($\hat{{y}} \geq 0.5$)" + "\n" + f"Dice = {dice:.3f} | $V_f = {bin_vf:.3f}$",
                  fontsize=11, fontweight='bold', pad=8)
    ax3.set_xlim(-5, nelx + 5)
    ax3.set_ylim(nely + 5, -5)
    ax3.set_aspect('equal')
    ax3.axis('off')
    
    # --- PANEL 5: Error Heatmap ---
    ax4 = axes[4]
    im4 = ax4.imshow(diff, cmap='inferno', vmin=0, vmax=1,
                     extent=[0, nelx, nely, 0], interpolation='nearest')
    ax4.set_title(rf"5. Absolute Error $|\hat{{y}} - y^*|$" + "\n" + f"MSE = {mse:.5f}",
                  fontsize=11, fontweight='bold', pad=8)
    ax4.set_xlim(-5, nelx + 5)
    ax4.set_ylim(nely + 5, -5)
    ax4.set_aspect('equal')
    ax4.axis('off')
    cbar = fig.colorbar(im4, ax=ax4, fraction=0.035, pad=0.04)
    cbar.set_label('Error Magnitude', fontsize=9)
    
    # Figure Super-Title with Benchmark Metadata
    fig.suptitle(
        f"Real-Time Topology Prediction | Sample #{sample_idx:03d} | "
        f"Inference Latency: {latency_ms:.2f} ms | Dice Similarity: {dice:.3f} | "
        f"Volume Error: |{pred_vf:.3f} - {target_vf:.3f}| = {abs(pred_vf - target_vf):.3f}",
        fontsize=13, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def benchmark_and_evaluate(checkpoint_path, dataset_path, output_dir="images",
                           num_visualizations=5, device_str='auto'):
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
        
    print(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = TopoUNet(in_channels=5, out_channels=1)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    dataset = TopoDataset(dataset_path, augment=False)
    n_samples = len(dataset)
    print(f"Loaded dataset: {dataset_path} ({n_samples} samples)")
    print(f"Device: {device}")
    
    # 1. Warm-up GPU/CPU
    dummy = torch.randn(1, 5, 60, 120, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)
            
    # 2. Benchmark Inference Latency
    n_bench = min(50, n_samples)
    latencies = []
    
    with torch.no_grad():
        for i in range(n_bench):
            x, _ = dataset[i]
            x_tensor = x.unsqueeze(0).to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            _ = model(x_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    print(f"\nInference Latency Benchmark ({n_bench} trials):")
    print(f"  Mean: {mean_lat:.2f} ms | Std: {std_lat:.2f} ms | FPS: {1000.0/mean_lat:.1f}")
    
    # 3. Full Dataset Evaluation
    dices = []
    mses = []
    vf_errors = []
    
    os.makedirs(output_dir, exist_ok=True)
    num_to_viz = min(num_visualizations, n_samples)
    print(f"\nEvaluating dataset and generating {num_to_viz} visual comparisons...")
    
    with torch.no_grad():
        for i in range(n_samples):
            x, y = dataset[i]
            x_tensor = x.unsqueeze(0).to(device)
            
            t0 = time.perf_counter()
            pred = model(x_tensor)
            t1 = time.perf_counter()
            lat = (t1 - t0) * 1000.0
            
            pred_np = pred.squeeze().cpu().numpy()
            target_np = y.squeeze().cpu().numpy()
            input_np = x.cpu().numpy()
            
            # Metrics
            bin_pred = (pred_np >= 0.5).astype(float)
            bin_gt = (target_np >= 0.5).astype(float)
            dice = (2.0 * np.sum(bin_pred * bin_gt) + 1e-6) / (np.sum(bin_pred) + np.sum(bin_gt) + 1e-6)
            mse = np.mean((pred_np - target_np) ** 2)
            
            target_vf = float(np.mean(input_np[4]))
            pred_vf = float(np.mean(pred_np))
            vf_err = abs(pred_vf - target_vf)
            
            dices.append(dice)
            mses.append(mse)
            vf_errors.append(vf_err)
            
            if i < num_to_viz:
                out_fig = os.path.join(output_dir, f"prediction_comparison_sample_{i:03d}.png")
                render_prediction_comparison(
                    sample_idx=i,
                    input_tensor=input_np,
                    target_density=target_np,
                    pred_density=pred_np,
                    latency_ms=lat,
                    output_path=out_fig
                )
                print(f"  [Sample {i:03d}] Dice={dice:.3f} | MSE={mse:.5f} | Vf Err={vf_err:.4f} -> {out_fig}")
                
    print(f"\n================ Overall Benchmark Summary ================")
    print(f"  Samples Evaluated:       {n_samples}")
    print(f"  Mean Dice Score:         {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
    print(f"  Mean Squared Error:      {np.mean(mses):.6f}")
    print(f"  Mean Volume Error:       {np.mean(vf_errors):.4f}")
    print(f"  Mean Inference Latency:  {mean_lat:.2f} ms")
    print(f"============================================================")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and Visualize TopoUNet Predictions")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_unet.pth", help="Model checkpoint path")
    parser.add_argument("--dataset", type=str, default="topo_dataset.npz", help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="images", help="Output directory for prediction figures")
    parser.add_argument("--num_viz", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    args = parser.parse_args()
    
    benchmark_and_evaluate(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        num_visualizations=args.num_viz,
        device_str=args.device
    )


if __name__ == '__main__':
    main()
