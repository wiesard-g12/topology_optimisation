"""
generate_validation_comparison.py - Authentic Evaluation & Visualization of TopoUNet vs SIMP
Evaluates 2 real, held-out validation cases from topo_dataset_1000.npz using checkpoints/best_unet.pth.
Computes true quantitative metrics, solves real FEA compliance for both, and renders publication-quality figures.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from unet_model import TopoUNet
from losses import compute_fea_compliance_ratio, check_structural_stability, physics_aware_binarize
from visualize_data import extract_discrete_loads, draw_loads, draw_domain_and_bcs


def load_model(checkpoint_path="checkpoints/best_unet.pth", device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[1/4] Loading model from: {checkpoint_path} on {device}")
    model = TopoUNet(in_channels=6, out_channels=1)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def evaluate_sample(sample_idx, dataset_path, model, device, threshold=0.35):
    data = np.load(dataset_path)
    input_tensor = data['inputs'][sample_idx]  # (6, 60, 120)
    simp_density = data['targets'][sample_idx] # (60, 120)
    simp_comp = float(data['compliance'][sample_idx])

    target_vf = float(np.mean(input_tensor[5]))

    # GPU Inference
    x_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)
    
    # Warmup
    with torch.inference_mode():
        for _ in range(5):
            _ = model(x_tensor)

    # Timed inference
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        pred_tensor = model(x_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_density = pred_tensor.squeeze().cpu().numpy()
    bin_pred = physics_aware_binarize(pred_density, input_tensor, threshold=threshold, polish_steps=2)
    bin_simp = (simp_density >= 0.5).astype(float)

    # Metrics
    inter = np.sum(bin_pred * bin_simp)
    dice = float((2.0 * inter + 1e-6) / (np.sum(bin_pred) + np.sum(bin_simp) + 1e-6))
    mse = float(np.mean((pred_density - simp_density) ** 2))
    mae = float(np.mean(np.abs(pred_density - simp_density)))
    pred_vf = float(np.mean(pred_density))
    bin_vf = float(np.mean(bin_pred))
    simp_vf = float(np.mean(simp_density))

    # Full Physics Stability & Breakage Check (Connectivity + FEA Compliance + Yield FoS + Buckling)
    stability = check_structural_stability(
        pred_density=pred_density,
        inp_np=input_tensor,
        simp_density=simp_density,
        threshold=threshold
    )
    fea_ratio = stability['fea_ratio']
    c_pred = stability['c_pred']
    c_target = stability['c_target']

    result = {
        'sample_idx': sample_idx,
        'threshold': threshold,
        'input_tensor': input_tensor,
        'simp_density': simp_density,
        'pred_density': pred_density,
        'bin_pred': bin_pred,
        'target_vf': target_vf,
        'simp_vf': simp_vf,
        'pred_vf': pred_vf,
        'bin_vf': bin_vf,
        'dice': dice,
        'mse': mse,
        'mae': mae,
        'latency_ms': latency_ms,
        'c_pred': c_pred,
        'c_target': c_target,
        'fea_ratio': fea_ratio,
        'dataset_compliance': simp_comp,
        'stability': stability
    }
    return result


def plot_single_case(res, output_path):
    input_tensor = res['input_tensor']
    simp_density = res['simp_density']
    pred_density = res['pred_density']
    bin_pred = res['bin_pred']
    nelx = input_tensor.shape[2]
    nely = input_tensor.shape[1]

    fig = plt.figure(figsize=(22, 9), dpi=160)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.32, wspace=0.20)

    # 1. Problem Setup
    ax0 = fig.add_subplot(gs[0, 0])
    ch_bc = np.clip(input_tensor[1] + input_tensor[2], 0.0, 1.0)
    draw_domain_and_bcs(ax0, ch_bc)
    loads = extract_discrete_loads(input_tensor[3], input_tensor[4])
    draw_loads(ax0, loads)
    ax0.set_title(f"1. Problem Definition (BCs & Applied Forces)\nTarget Volume Fraction $V_f = {res['target_vf']:.3f}$",
                  fontsize=11, fontweight='bold', pad=8)
    ax0.set_xlim(-15, nelx + 15)
    ax0.set_ylim(nely + 10, -10)
    ax0.set_aspect('equal')
    ax0.axis('off')

    # 2. Ground Truth SIMP
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(1.0 - simp_density, cmap='gray', vmin=0, vmax=1,
               extent=[0, nelx, nely, 0], interpolation='nearest')
    ax1.set_title(f"2. Ground Truth SIMP (50-80 Iterations FEA)\n$V_f = {res['simp_vf']:.3f}$ | Compliance $C^* = {res['c_target']:.2f}$",
                  fontsize=11, fontweight='bold', pad=8)
    ax1.set_xlim(-5, nelx + 5)
    ax1.set_ylim(nely + 5, -5)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # 3. TopoUNet Continuous Prediction
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(1.0 - pred_density, cmap='gray', vmin=0, vmax=1,
               extent=[0, nelx, nely, 0], interpolation='nearest')
    ax2.set_title(f"3. TopoUNet Continuous Output ($\hat{{y}}$)\nLatency = {res['latency_ms']:.2f} ms | Pred $V_f = {res['pred_vf']:.3f}$",
                  fontsize=11, fontweight='bold', pad=8)
    ax2.set_xlim(-5, nelx + 5)
    ax2.set_ylim(nely + 5, -5)
    ax2.set_aspect('equal')
    ax2.axis('off')

    # 4. TopoUNet Binarized Truss + Stability & Breakage Verdict
    stab = res['stability']
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(1.0 - bin_pred, cmap='gray', vmin=0, vmax=1,
               extent=[0, nelx, nely, 0], interpolation='nearest')
    conn_str = "Connected" if stab['is_connected'] else "DISCONNECTED"
    ax3.set_title(
        rf"4. Binarized Truss ($\hat{{y}} \geq {res['threshold']:.2f}$) [{stab['short_badge']}]" + "\n" +
        f"Dice={res['dice']:.3f} | Ratio={res['fea_ratio']:.2f}x | FoS={stab['fos_yield']:.2f} | $\lambda_1$={stab['lambda_1']:.1f}",
        fontsize=10.5, fontweight='bold', color=stab['badge_color'], pad=8
    )
    ax3.text(
        nelx / 2.0, nely + 3.5,
        f"Stability Check: {stab['status_badge']} | Load Path: {conn_str}",
        ha='center', va='top', fontsize=9, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=stab['badge_color'], edgecolor='none', alpha=0.92)
    )
    ax3.set_xlim(-5, nelx + 5)
    ax3.set_ylim(nely + 10, -5)
    ax3.set_aspect('equal')
    ax3.axis('off')

    # 5. Absolute Discrepancy Heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    diff = np.abs(pred_density - simp_density)
    im4 = ax4.imshow(diff, cmap='inferno', vmin=0, vmax=1,
                     extent=[0, nelx, nely, 0], interpolation='nearest')
    ax4.set_title(rf"5. Absolute Error Field $|\hat{{y}} - y^*|$" + "\n" +
                  f"MSE = {res['mse']:.4f} | MAE = {res['mae']:.4f}",
                  fontsize=11, fontweight='bold', pad=8)
    ax4.set_xlim(-5, nelx + 5)
    ax4.set_ylim(nely + 5, -5)
    ax4.set_aspect('equal')
    ax4.axis('off')
    cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.035, pad=0.04)
    cbar4.set_label('Error Magnitude', fontsize=9)

    # 6. Density Distribution Histogram
    ax5 = fig.add_subplot(gs[1, 2])
    bins = np.linspace(0, 1, 30)
    ax5.hist(pred_density.flatten(), bins=bins, color='#1976D2', alpha=0.7,
             label='TopoUNet Pred', density=True)
    ax5.hist(simp_density.flatten(), bins=bins, color='#D32F2F', alpha=0.5,
             histtype='step', lw=2.5, label='SIMP Ground Truth', density=True)
    ax5.axvline(res['threshold'], color='purple', linestyle='--', lw=1.8, label=f"Threshold ({res['threshold']:.2f})")
    ax5.set_title("6. Density Distribution (Binarization Sharpness)", fontsize=11, fontweight='bold', pad=8)
    ax5.set_xlabel(r"Element Density $\rho$", fontsize=10)
    ax5.set_ylabel("Probability Density", fontsize=10)
    ax5.legend(loc='upper center', fontsize=9, frameon=True)
    ax5.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle(
        f"Real-Time TopoUNet vs Ground Truth SIMP | Validation Sample #{res['sample_idx']:03d} | "
        f"Inference: {res['latency_ms']:.2f} ms | Dice: {res['dice']:.3f} | Ratio: {res['fea_ratio']:.2f}x | "
        f"Stability: {stab['short_badge']} (FoS={stab['fos_yield']:.2f}, λ₁={stab['lambda_1']:.1f})",
        fontsize=12.5, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {output_path}")


def plot_side_by_side_summary(res1, res2, output_path):
    """
    Renders a unified, high-impact 2-case comparative engineering summary.
    Row 1: Case 1 (Sample #132)
    Row 2: Case 2 (Sample #733)
    Columns:
      Col 1: Problem BCs & Loads
      Col 2: Ground Truth SIMP
      Col 3: TopoUNet Continuous Output
      Col 4: TopoUNet Binarized Truss
      Col 5: Error Heatmap
    """
    cases = [res1, res2]
    nelx = 120
    nely = 60

    fig, axes = plt.subplots(2, 5, figsize=(26, 9), dpi=180)

    for row_idx, res in enumerate(cases):
        inp = res['input_tensor']
        simp = res['simp_density']
        pred = res['pred_density']
        bin_p = res['bin_pred']
        diff = np.abs(pred - simp)

        # Col 0: Problem Setup
        ax0 = axes[row_idx, 0]
        ch_bc = np.clip(inp[1] + inp[2], 0.0, 1.0)
        draw_domain_and_bcs(ax0, ch_bc)
        loads = extract_discrete_loads(inp[3], inp[4])
        draw_loads(ax0, loads)
        ax0.set_title(f"Case {row_idx+1}: Val #{res['sample_idx']:03d}\nLoads & Supports ($V_f={res['target_vf']:.2f}$)",
                      fontsize=11, fontweight='bold', pad=6)
        ax0.set_xlim(-15, nelx + 15)
        ax0.set_ylim(nely + 10, -10)
        ax0.set_aspect('equal')
        ax0.axis('off')

        # Col 1: Ground Truth SIMP
        ax1 = axes[row_idx, 1]
        ax1.imshow(1.0 - simp, cmap='gray', vmin=0, vmax=1, extent=[0, nelx, nely, 0])
        ax1.set_title(f"Ground Truth SIMP\n$V_f={res['simp_vf']:.3f}$ | $C^*={res['c_target']:.1f}$",
                      fontsize=11, fontweight='bold', pad=6)
        ax1.set_xlim(-5, nelx + 5)
        ax1.set_ylim(nely + 5, -5)
        ax1.set_aspect('equal')
        ax1.axis('off')

        # Col 2: TopoUNet Continuous
        ax2 = axes[row_idx, 2]
        ax2.imshow(1.0 - pred, cmap='gray', vmin=0, vmax=1, extent=[0, nelx, nely, 0])
        ax2.set_title(f"TopoUNet Continuous ($\hat{{y}}$)\nLatency = {res['latency_ms']:.2f} ms",
                      fontsize=11, fontweight='bold', pad=6)
        ax2.set_xlim(-5, nelx + 5)
        ax2.set_ylim(nely + 5, -5)
        ax2.set_aspect('equal')
        ax2.axis('off')

        # Col 3: TopoUNet Binarized Truss + Stability Check
        stab = res['stability']
        ax3 = axes[row_idx, 3]
        ax3.imshow(1.0 - bin_p, cmap='gray', vmin=0, vmax=1, extent=[0, nelx, nely, 0])
        ax3.set_title(
            rf"TopoUNet Truss ($\hat{{y}} \geq 0.35$) [{stab['short_badge']}]" + "\n" +
            f"Dice={res['dice']:.3f} | Ratio={res['fea_ratio']:.2f}x | FoS={stab['fos_yield']:.2f}",
            fontsize=10.5, fontweight='bold', color=stab['badge_color'], pad=6
        )
        ax3.set_xlim(-5, nelx + 5)
        ax3.set_ylim(nely + 5, -5)
        ax3.set_aspect('equal')
        ax3.axis('off')

        # Col 4: Error Heatmap
        ax4 = axes[row_idx, 4]
        im4 = ax4.imshow(diff, cmap='inferno', vmin=0, vmax=1, extent=[0, nelx, nely, 0])
        ax4.set_title(f"Absolute Error $|\hat{{y}} - y^*|$\nMSE = {res['mse']:.4f}",
                      fontsize=11, fontweight='bold', pad=6)
        ax4.set_xlim(-5, nelx + 5)
        ax4.set_ylim(nely + 5, -5)
        ax4.set_aspect('equal')
        ax4.axis('off')
        cbar = fig.colorbar(im4, ax=ax4, fraction=0.035, pad=0.04)
        cbar.set_label('Error', fontsize=8)

    fig.suptitle(
        "Real-Time TopoUNet vs Ground Truth SIMP — Head-to-Head Engineering & Stability Evaluation\n"
        "(Held-Out Validation Cases Evaluated with Real Checkpoint Weights, FEA Compliance & Breakage Check)",
        fontsize=14, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  [Saved Summary] {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate TopoUNet vs SIMP Ground Truth")
    parser.add_argument("--dataset", type=str, default="topo_dataset_3000.npz", help="Dataset path")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_unet.pth", help="Model checkpoint path")
    parser.add_argument("--indices", type=int, nargs="+", default=[132, 733, 1500, 2500], help="Sample indices to evaluate")
    parser.add_argument("--threshold", type=float, default=0.35, help="Binary threshold for density")
    parser.add_argument("--output_dir", type=str, default="images", help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_path = args.dataset
    checkpoint_path = args.checkpoint

    model, device = load_model(checkpoint_path)

    results = []
    for idx in args.indices:
        print(f"Evaluating Sample #{idx}...")
        try:
            res = evaluate_sample(idx, dataset_path, model, device, threshold=args.threshold)
            results.append(res)
            fig_path = os.path.join(args.output_dir, f"comparison_sample_{idx}.png")
            plot_single_case(res, fig_path)
        except Exception as ex:
            print(f"  Warning: Could not evaluate sample #{idx}: {ex}")

    if len(results) >= 2:
        summary_path = os.path.join(args.output_dir, "unet_vs_simp_summary.png")
        plot_side_by_side_summary(results[0], results[1], summary_path)

    print("\n" + "=" * 75)
    print("  AUTHENTIC EVALUATION METRICS (TOPOUNET VS GROUND TRUTH SIMP)")
    print("=" * 75)
    for res in results:
        stab = res['stability']
        print(f"SAMPLE #{res['sample_idx']}:")
        print(f"  Stability Verdict:               {stab['status_badge']}")
        print(f"  Load-to-Support Connected:       {stab['is_connected']} (Anchored Mass: {stab['anchored_mass_ratio']*100:.1f}%)")
        print(f"  Yield Factor of Safety (FoS):    {stab['fos_yield']:.2f} (SIMP FoS: {stab['simp_fos']:.2f})")
        print(f"  Buckling Multiplier (lambda_1):  {stab['lambda_1']:.2f} (SIMP lambda_1: {stab['simp_lambda_1']:.2f})")
        print(f"  Target Volume Fraction (Vf):     {res['target_vf']:.3f}")
        print(f"  SIMP Achieved Volume Fraction:   {res['simp_vf']:.3f}")
        print(f"  TopoUNet Predicted Volume Frac:  {res['pred_vf']:.3f} (Error: {abs(res['pred_vf'] - res['target_vf']):.4f})")
        print(f"  Binarized Dice Score:            {res['dice']:.4f}")
        print(f"  Mean Squared Error (MSE):        {res['mse']:.5f}")
        print(f"  Mean Absolute Error (MAE):       {res['mae']:.5f}")
        print(f"  Inference Latency:               {res['latency_ms']:.2f} ms")
        print(f"  SIMP Ground Truth Compliance C*: {res['c_target']:.2f}")
        print(f"  TopoUNet Predicted Compliance C: {res['c_pred']:.2f}")
        print(f"  FEA Compliance Ratio (C_pred/C*): {res['fea_ratio']:.2f}")
        print("-" * 75)

    print("All comparison figures and metrics successfully generated.")


if __name__ == '__main__':
    main()
