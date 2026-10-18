"""
visualize_unet.py - Comprehensive Engineering Visualization Suite for TopoUNet / PI-UNet
Provides multi-aspect visualization, sensitivity sweeps, stress recovery, and statistical scorecards.

Visualization Modes:
  1. --mode comparison (default): Publication-quality 6-panel evaluation
     (Problem BCs, SIMP Ground Truth, U-Net Continuous, Binarized Truss, Error Map, Density Histogram)
  2. --mode sweep_vf: Real-time volume fraction response sweep (Vf = 0.25 -> 0.55)
  3. --mode sweep_threshold: Cutoff sensitivity analysis (tau = 0.2 -> 0.8) with Dice & Vf curve
  4. --mode stress: Physical stress recovery & compliance verification (Von Mises stress heatmaps)
  5. --mode batch: Statistical scorecard across multiple dataset samples (Dice, MSE, Vf error, Latency)

Usage:
  python visualize_unet.py --case cantilever --volfrac 0.40
  python visualize_unet.py --sample_idx 1
  python visualize_unet.py --mode sweep_vf --case mbb
  python visualize_unet.py --mode sweep_threshold --sample_idx 1
  python visualize_unet.py --mode stress --sample_idx 1 --material steel --force 1500
  python visualize_unet.py --mode batch --dataset topo_dataset_1000.npz --num_samples 50
"""

import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from unet_model import TopoUNet
from interactive_demo import build_benchmark_problem
from visualize_data import extract_discrete_loads, draw_loads, draw_domain_and_bcs
from simp_solver import MATERIALS, analyze_structural_integrity
from dataset_generator import precompute_mesh_and_filter


def load_model(checkpoint_path="checkpoints/best_unet.pth", device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    model = TopoUNet(in_channels=6, out_channels=1)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Notice: Loading checkpoint with strict=False ({e})")
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, device


def get_sample_data(sample_idx=None, case_name='cantilever', volfrac=0.40, dataset_path="topo_dataset_1000.npz"):
    if sample_idx is not None:
        if not os.path.exists(dataset_path):
            dataset_path = "topo_dataset.npz"
        data = np.load(dataset_path)
        input_tensor = data['inputs'][sample_idx]
        if input_tensor.shape[0] == 5:
            input_tensor = np.concatenate([input_tensor[0:1], input_tensor[1:2], input_tensor[1:2], input_tensor[2:5]], axis=0)
        simp_density = data['targets'][sample_idx]
        title = f"Dataset Sample #{sample_idx:03d}"
        target_vf = float(np.mean(input_tensor[-1]))
        meta = None
    else:
        input_tensor, meta = build_benchmark_problem(case_name, volfrac=volfrac)
        simp_density = None
        title = f"Benchmark: {case_name.upper()}"
        target_vf = volfrac
    return input_tensor, simp_density, title, target_vf, meta


# =========================================================================
# MODE 1: 6-PANEL ENGINEERING COMPARISON
# =========================================================================
def visualize_comparison(input_tensor, simp_density, title, target_vf, model, device,
                         threshold=0.5, output_path="images/unet_viz_comparison.png"):
    nelx = input_tensor.shape[2]
    nely = input_tensor.shape[1]

    tensor_x = torch.from_numpy(input_tensor).unsqueeze(0).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred_tensor = model(tensor_x)
    pred_density = pred_tensor.squeeze().cpu().numpy()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_vf = float(np.mean(pred_density))
    bin_pred = (pred_density >= threshold).astype(float)
    bin_vf = float(np.mean(bin_pred))

    dice = None
    mse = None
    diff = None
    if simp_density is not None:
        bin_simp = (simp_density >= 0.5).astype(float)
        inter = np.sum(bin_pred * bin_simp)
        dice = (2.0 * inter + 1e-6) / (np.sum(bin_pred) + np.sum(bin_simp) + 1e-6)
        mse = float(np.mean((pred_density - simp_density) ** 2))
        diff = np.abs(pred_density - simp_density)

    fig = plt.figure(figsize=(22, 9), dpi=160)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.32, wspace=0.20)

    # 1. Problem Setup
    ax0 = fig.add_subplot(gs[0, 0])
    if input_tensor.shape[0] == 6:
        ch_bc = np.clip(input_tensor[1] + input_tensor[2], 0.0, 1.0)
        ch_fx = input_tensor[3]
        ch_fy = input_tensor[4]
    else:
        ch_bc = input_tensor[1]
        ch_fx = input_tensor[2]
        ch_fy = input_tensor[3]

    draw_domain_and_bcs(ax0, ch_bc)
    loads = extract_discrete_loads(ch_fx, ch_fy)
    draw_loads(ax0, loads)
    ax0.set_title(f"1. Problem Definition (BCs & Loads)\nTarget $V_f = {target_vf:.2f}$",
                  fontsize=11, fontweight='bold', pad=8)
    ax0.set_xlim(-15, nelx + 15)
    ax0.set_ylim(nely + 10, -10)
    ax0.set_aspect('equal')
    ax0.axis('off')

    # 2. Ground Truth SIMP
    ax1 = fig.add_subplot(gs[0, 1])
    if simp_density is not None:
        ax1.imshow(1.0 - simp_density, cmap='gray', vmin=0, vmax=1,
                   extent=[0, nelx, nely, 0], interpolation='nearest')
        ax1.set_title(f"2. Ground Truth SIMP (80 iter)\n$V_f = {float(np.mean(simp_density)):.3f}$",
                      fontsize=11, fontweight='bold', pad=8)
    else:
        ax1.text(0.5, 0.5, "Benchmark Run\n(SIMP not precomputed)", ha='center', va='center',
                 transform=ax1.transAxes, fontsize=12, color='gray')
        ax1.set_title("2. Ground Truth SIMP", fontsize=11, fontweight='bold', pad=8)
    ax1.set_xlim(-5, nelx + 5)
    ax1.set_ylim(nely + 5, -5)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # 3. U-Net Continuous Output
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(1.0 - pred_density, cmap='gray', vmin=0, vmax=1,
                     extent=[0, nelx, nely, 0], interpolation='nearest')
    ax2.set_title(rf"3. TopoUNet Continuous ($\hat{{y}}$)" + "\n" +
                  f"Latency: {latency_ms:.1f} ms | $V_f = {pred_vf:.3f}$",
                  fontsize=11, fontweight='bold', pad=8)
    ax2.set_xlim(-5, nelx + 5)
    ax2.set_ylim(nely + 5, -5)
    ax2.set_aspect('equal')
    ax2.axis('off')

    # 4. Binarized Topology
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(1.0 - bin_pred, cmap='gray', vmin=0, vmax=1,
               extent=[0, nelx, nely, 0], interpolation='nearest')
    dice_str = f" | Dice = {dice:.3f}" if dice is not None else ""
    ax3.set_title(rf"4. Binarized Truss ($\hat{{y}} \geq {threshold:.2f}$)" + "\n" +
                  f"$V_f = {bin_vf:.3f}${dice_str}",
                  fontsize=11, fontweight='bold', pad=8)
    ax3.set_xlim(-5, nelx + 5)
    ax3.set_ylim(nely + 5, -5)
    ax3.set_aspect('equal')
    ax3.axis('off')

    # 5. Error Heatmap or Continuous Colormap
    ax4 = fig.add_subplot(gs[1, 1])
    if diff is not None:
        im4 = ax4.imshow(diff, cmap='inferno', vmin=0, vmax=1,
                         extent=[0, nelx, nely, 0], interpolation='nearest')
        ax4.set_title(rf"5. Absolute Error $|\hat{{y}} - y^*|$" + "\n" + f"MSE = {mse:.5f}",
                      fontsize=11, fontweight='bold', pad=8)
        cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.035, pad=0.04)
        cbar4.set_label('Error', fontsize=9)
    else:
        im4 = ax4.imshow(pred_density, cmap='viridis', vmin=0, vmax=1,
                         extent=[0, nelx, nely, 0], interpolation='nearest')
        ax4.set_title(rf"5. Material Concentration Density ($\hat{{y}}$)" + "\n" + f"Volume Error = {abs(pred_vf - target_vf):.4f}",
                      fontsize=11, fontweight='bold', pad=8)
        cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.035, pad=0.04)
        cbar4.set_label(r'Density $\rho$', fontsize=9)
    ax4.set_xlim(-5, nelx + 5)
    ax4.set_ylim(nely + 5, -5)
    ax4.set_aspect('equal')
    ax4.axis('off')

    # 6. Density Distribution & Binarization Histogram
    ax5 = fig.add_subplot(gs[1, 2])
    bins = np.linspace(0, 1, 30)
    ax5.hist(pred_density.flatten(), bins=bins, color='#1976D2', alpha=0.7,
             label='TopoUNet Pred', density=True)
    if simp_density is not None:
        ax5.hist(simp_density.flatten(), bins=bins, color='#D32F2F', alpha=0.5,
                 histtype='step', lw=2.5, label='SIMP Ground Truth', density=True)
    ax5.axvline(threshold, color='purple', linestyle='--', lw=1.8, label=f'Threshold ({threshold})')
    ax5.set_title("6. Density Distribution (Binarization Quality)", fontsize=11, fontweight='bold', pad=8)
    ax5.set_xlabel(r"Element Density $\rho$", fontsize=10)
    ax5.set_ylabel("Probability Density", fontsize=10)
    ax5.legend(loc='upper center', fontsize=9, frameon=True)
    ax5.grid(True, linestyle=':', alpha=0.6)

    dice_title = f" | Dice Score: {dice:.3f}" if dice is not None else ""
    fig.suptitle(
        f"TopoUNet Engineering Inspection | {title} | "
        f"Inference Latency: {latency_ms:.2f} ms | Target $V_f$: {target_vf:.2f} "
        f"$\to$ Pred $V_f$: {pred_vf:.3f}{dice_title}",
        fontsize=13, fontweight='bold', y=0.98
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[Comparison Visualizer] Saved: {os.path.abspath(output_path)}")
    return output_path, latency_ms, dice


# =========================================================================
# MODE 2: VOLUME FRACTION SENSITIVITY SWEEP (Vf Response Curve)
# =========================================================================
def visualize_vf_sweep(case_name='cantilever', vfs=(0.25, 0.35, 0.45, 0.55),
                       model=None, device=None, output_path="images/unet_viz_vf_sweep.png"):
    if model is None:
        model, device = load_model()

    fig, axes = plt.subplots(len(vfs), 3, figsize=(16, 3.6 * len(vfs)), dpi=160)

    nelx, nely = 120, 60
    pred_vfs = []
    latencies = []

    for idx, vf in enumerate(vfs):
        inp_tensor, _ = build_benchmark_problem(case_name, volfrac=vf, nelx=nelx, nely=nely)
        tensor_x = torch.from_numpy(inp_tensor).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            pred_tensor = model(tensor_x)
        pred = pred_tensor.squeeze().cpu().numpy()
        lat = (time.perf_counter() - t0) * 1000.0

        pvf = float(np.mean(pred))
        bvf = float(np.mean(pred >= 0.5))
        pred_vfs.append(pvf)
        latencies.append(lat)

        # Panel 1: BCs & Load
        ax0 = axes[idx, 0]
        if inp_tensor.shape[0] == 6:
            ch_bc = np.clip(inp_tensor[1] + inp_tensor[2], 0.0, 1.0)
            ch_fx = inp_tensor[3]
            ch_fy = inp_tensor[4]
        else:
            ch_bc = inp_tensor[1]
            ch_fx = inp_tensor[2]
            ch_fy = inp_tensor[3]

        draw_domain_and_bcs(ax0, ch_bc)
        loads = extract_discrete_loads(ch_fx, ch_fy)
        draw_loads(ax0, loads)
        ax0.set_title(f"Target $V_f = {vf:.2f}$ (Setup)", fontsize=10, fontweight='bold')
        ax0.set_xlim(-15, nelx + 15)
        ax0.set_ylim(nely + 10, -10)
        ax0.set_aspect('equal')
        ax0.axis('off')

        # Panel 2: Continuous Prediction
        ax1 = axes[idx, 1]
        ax1.imshow(1.0 - pred, cmap='gray', vmin=0, vmax=1,
                   extent=[0, nelx, nely, 0], interpolation='nearest')
        ax1.set_title(rf"Continuous Pred ($\hat{{y}}$) | Latency: {lat:.1f} ms" + "\n" +
                      f"Pred $V_f = {pvf:.3f}$ (Err: {abs(pvf-vf):.4f})",
                      fontsize=10, fontweight='bold')
        ax1.set_xlim(-5, nelx + 5)
        ax1.set_ylim(nely + 5, -5)
        ax1.set_aspect('equal')
        ax1.axis('off')

        # Panel 3: Binarized Truss
        ax2 = axes[idx, 2]
        bin_img = (pred >= 0.5).astype(float)
        ax2.imshow(1.0 - bin_img, cmap='gray', vmin=0, vmax=1,
                   extent=[0, nelx, nely, 0], interpolation='nearest')
        ax2.set_title(rf"Binarized Truss ($\hat{{y}} \geq 0.5$)" + "\n" +
                      f"Truss $V_f = {bvf:.3f}$",
                      fontsize=10, fontweight='bold')
        ax2.set_xlim(-5, nelx + 5)
        ax2.set_ylim(nely + 5, -5)
        ax2.set_aspect('equal')
        ax2.axis('off')

    fig.suptitle(
        f"TopoUNet Volume Fraction Sensitivity Sweep | Benchmark: {case_name.upper()} | "
        f"Mean Latency: {np.mean(latencies):.1f} ms",
        fontsize=13, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[Vf Sweep Visualizer] Saved: {os.path.abspath(output_path)}")
    return output_path


# =========================================================================
# MODE 3: BINARIZATION THRESHOLD SENSITIVITY SWEEP
# =========================================================================
def visualize_threshold_sweep(sample_idx=1, case_name='cantilever', volfrac=0.40,
                              model=None, device=None, output_path="images/unet_viz_threshold_sweep.png"):
    if model is None:
        model, device = load_model()

    inp_tensor, simp_density, title, target_vf, _ = get_sample_data(sample_idx, case_name, volfrac)
    nelx = inp_tensor.shape[2]
    nely = inp_tensor.shape[1]

    tensor_x = torch.from_numpy(inp_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_tensor = model(tensor_x)
    pred = pred_tensor.squeeze().cpu().numpy()

    thresholds = [0.20, 0.35, 0.50, 0.65, 0.80]

    fig = plt.figure(figsize=(20, 8), dpi=160)
    gs = fig.add_gridspec(2, len(thresholds), height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.15)

    truss_vfs = []
    dices = []

    for i, tau in enumerate(thresholds):
        bin_p = (pred >= tau).astype(float)
        bvf = float(np.mean(bin_p))
        truss_vfs.append(bvf)

        dice_val = None
        if simp_density is not None:
            bin_s = (simp_density >= 0.5).astype(float)
            inter = np.sum(bin_p * bin_s)
            dice_val = (2.0 * inter + 1e-6) / (np.sum(bin_p) + np.sum(bin_s) + 1e-6)
            dices.append(dice_val)

        ax = fig.add_subplot(gs[0, i])
        ax.imshow(1.0 - bin_p, cmap='gray', vmin=0, vmax=1,
                  extent=[0, nelx, nely, 0], interpolation='nearest')
        d_str = f"\nDice = {dice_val:.3f}" if dice_val is not None else ""
        ax.set_title(rf"$\tau = {tau:.2f}$" + f"\n$V_f = {bvf:.3f}$" + d_str,
                     fontsize=10, fontweight='bold')
        ax.set_xlim(-5, nelx + 5)
        ax.set_ylim(nely + 5, -5)
        ax.set_aspect('equal')
        ax.axis('off')

    # Curve plot on lower half
    ax_curve = fig.add_subplot(gs[1, :])
    fine_taus = np.linspace(0.1, 0.9, 50)
    curve_vfs = [float(np.mean(pred >= t)) for t in fine_taus]

    ax_curve.plot(fine_taus, curve_vfs, 'b-o', lw=2.2, label=r'Achieved Volume Fraction $V_f(\tau)$')
    ax_curve.axhline(target_vf, color='gray', linestyle='--', label=f'Target $V_f = {target_vf:.2f}$')

    if simp_density is not None:
        bin_s = (simp_density >= 0.5).astype(float)
        curve_dices = []
        for t in fine_taus:
            bp = (pred >= t).astype(float)
            inter = np.sum(bp * bin_s)
            d = (2.0 * inter + 1e-6) / (np.sum(bp) + np.sum(bin_s) + 1e-6)
            curve_dices.append(d)
        ax_dice = ax_curve.twinx()
        ax_dice.plot(fine_taus, curve_dices, 'r-s', lw=2.2, label=r'Dice Similarity Score')
        ax_dice.set_ylabel("Dice Score (0 to 1)", color='r', fontsize=11)
        ax_dice.tick_params(axis='y', labelcolor='r')
        ax_dice.set_ylim(0, 1.0)

    ax_curve.set_xlabel(r"Binarization Cutoff Threshold $\tau$", fontsize=11)
    ax_curve.set_ylabel(r"Resulting Volume Fraction $V_f$", color='b', fontsize=11)
    ax_curve.tick_params(axis='y', labelcolor='b')
    ax_curve.set_title("Binarization Operating Curve: Volume Fraction & Overlap vs Cutoff", fontsize=11, fontweight='bold')
    ax_curve.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle(f"TopoUNet Threshold Sweep & Operating Curve | {title}", fontsize=13, fontweight='bold', y=0.98)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[Threshold Sweep Visualizer] Saved: {os.path.abspath(output_path)}")
    return output_path


# =========================================================================
# MODE 4: REAL-WORLD STRESS RECOVERY & PHYSICAL COMPLIANCE
# =========================================================================
def visualize_stress_recovery(sample_idx=1, material='steel', force_mag=1500.0,
                              model=None, device=None, output_path="images/unet_viz_stress.png"):
    if model is None:
        model, device = load_model()

    mat = MATERIALS.get(material.lower(), MATERIALS['steel'])

    # Load dataset sample
    dataset_path = "topo_dataset_1000.npz" if os.path.exists("topo_dataset_1000.npz") else "topo_dataset.npz"
    data = np.load(dataset_path)
    input_tensor = data['inputs'][sample_idx]
    if input_tensor.shape[0] == 5:
        input_tensor = np.concatenate([input_tensor[0:1], input_tensor[1:2], input_tensor[1:2], input_tensor[2:5]], axis=0)
    simp_density = data['targets'][sample_idx]

    nelx = input_tensor.shape[2]
    nely = input_tensor.shape[1]

    # U-Net prediction
    tensor_x = torch.from_numpy(input_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_tensor = model(tensor_x)
    pred_density = pred_tensor.squeeze().cpu().numpy()
    bin_pred = (pred_density >= 0.5).astype(float)

    # Reconstruct load candidates and fixtures from input channels
    edofMat, iK, jK, H, Hs = precompute_mesh_and_filter(nelx, nely, 1.5)

    # Extract fixed DOFs from channel 1 (bc_x) and channel 2 (bc_y)
    ch_bc_x = input_tensor[1]
    ch_bc_y = input_tensor[2]
    fixed_dofs = []
    for y in range(nely):
        for x in range(nelx):
            nodes = [
                x * (nely + 1) + y,
                (x + 1) * (nely + 1) + y,
                (x + 1) * (nely + 1) + y + 1,
                x * (nely + 1) + y + 1
            ]
            if ch_bc_x[y, x] > 0.05:
                for n in nodes:
                    fixed_dofs.append(2*n)
            if ch_bc_y[y, x] > 0.05:
                for n in nodes:
                    fixed_dofs.append(2*n + 1)
    fixed_dofs = np.unique(fixed_dofs) if len(fixed_dofs) > 0 else np.array([0, 1], dtype=int)

    # Extract loads from channel 3 & 4
    ch_fx = input_tensor[3]
    ch_fy = input_tensor[4]
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof)
    for y in range(nely):
        for x in range(nelx):
            if abs(ch_fx[y, x]) > 1e-4 or abs(ch_fy[y, x]) > 1e-4:
                nodes = [
                    x * (nely + 1) + y,
                    (x + 1) * (nely + 1) + y,
                    (x + 1) * (nely + 1) + y + 1,
                    x * (nely + 1) + y + 1
                ]
                for n in nodes:
                    F[2*n] += 0.25 * ch_fx[y, x] * force_mag
                    F[2*n + 1] += 0.25 * ch_fy[y, x] * force_mag

    # Solve FEA & Stress for SIMP
    res_simp = analyze_structural_integrity(
        density=simp_density,
        fixed_dofs=fixed_dofs, F=F,
        edofMat=edofMat,
        nelx=nelx, nely=nely,
        material_name=material,
        force_real_N=force_mag
    )

    # Solve FEA & Stress for U-Net Binarized
    res_unet = analyze_structural_integrity(
        density=bin_pred,
        fixed_dofs=fixed_dofs, F=F,
        edofMat=edofMat,
        nelx=nelx, nely=nely,
        material_name=material,
        force_real_N=force_mag
    )

    if res_simp is None or res_unet is None:
        print("Warning: FEA matrix singular. Scaling supports...")
        return None

    vm_simp = res_simp['von_mises_grid']
    vm_unet = res_unet['von_mises_grid']

    peak_vm_simp = res_simp['peak_vm_stress']
    peak_vm_unet = res_unet['peak_vm_stress']

    fos_simp = res_simp['fos_yield']
    fos_unet = res_unet['fos_yield']

    defl_simp = res_simp['max_deflection_mm']
    defl_unet = res_unet['max_deflection_mm']

    # 4-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=160)

    # 1. SIMP Topology & Stress
    ax0 = axes[0, 0]
    ax0.imshow(1.0 - simp_density, cmap='gray', extent=[0, nelx, nely, 0], alpha=0.3)
    im0 = ax0.imshow(vm_simp, cmap='jet', extent=[0, nelx, nely, 0], alpha=0.85,
                     vmin=0, vmax=max(peak_vm_simp, peak_vm_unet)*0.8)
    ax0.set_title(f"1. Ground Truth SIMP: Von Mises Stress\nMax Deflection: {defl_simp:.4f} mm | Peak Stress: {peak_vm_simp:.1f} MPa",
                  fontsize=11, fontweight='bold', pad=8)
    ax0.axis('off')
    fig.colorbar(im0, ax=ax0, fraction=0.035, pad=0.04, label='Stress (MPa)')

    # 2. U-Net Topology & Stress
    ax1 = axes[0, 1]
    ax1.imshow(1.0 - bin_pred, cmap='gray', extent=[0, nelx, nely, 0], alpha=0.3)
    im1 = ax1.imshow(vm_unet, cmap='jet', extent=[0, nelx, nely, 0], alpha=0.85,
                     vmin=0, vmax=max(peak_vm_simp, peak_vm_unet)*0.8)
    ax1.set_title(f"2. TopoUNet Binarized: Von Mises Stress\nMax Deflection: {defl_unet:.4f} mm | Peak Stress: {peak_vm_unet:.1f} MPa",
                  fontsize=11, fontweight='bold', pad=8)
    ax1.axis('off')
    fig.colorbar(im1, ax=ax1, fraction=0.035, pad=0.04, label='Stress (MPa)')

    # 3. Maximum Structural Deflection Comparison
    defl_delta = abs(defl_unet - defl_simp) / (defl_simp + 1e-9) * 100.0
    ax2 = axes[1, 0]
    bars = ax2.bar(['SIMP (80 iter)', 'TopoUNet (< 50ms)'], [defl_simp, defl_unet],
                   color=['#1976D2', '#388E3C'], width=0.45)
    ax2.set_ylabel("Max Deflection (mm) [Lower is Stiffer]", fontsize=10)
    ax2.set_title(f"3. Maximum Deflection (Stiffness Verification)\nDeflection Discrepancy: {defl_delta:.1f}%",
                  fontsize=11, fontweight='bold', pad=8)
    ax2.grid(True, linestyle=':', alpha=0.6)
    for b in bars:
        h = b.get_height()
        ax2.annotate(f"{h:.4f} mm", (b.get_x() + b.get_width() / 2, h / 2),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=11)

    # 4. Factor of Safety & Engineering Verdict
    ax3 = axes[1, 1]
    fos_bars = ax3.bar(['SIMP FoS', 'TopoUNet FoS'], [fos_simp, fos_unet],
                       color=['#0288D1', '#7B1FA2'], width=0.45)
    ax3.axhline(1.0, color='red', linestyle='--', lw=2, label='Yield Limit (FoS = 1.0)')
    ax3.set_ylabel("Factor of Safety (Yielding)", fontsize=10)
    verdict = "PASS" if fos_unet >= 1.0 else "FAIL"
    verdict_color = "green" if fos_unet >= 1.0 else "red"
    ax3.set_title(f"4. Factor of Safety (Material: {mat['name']})\nVerdict: {verdict} (FoS = {fos_unet:.2f})",
                  fontsize=11, fontweight='bold', pad=8, color=verdict_color)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.6)
    for b in fos_bars:
        h = b.get_height()
        ax3.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h / 2),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=11)

    fig.suptitle(
        f"TopoUNet Physics & Stress Verification | Sample #{sample_idx:03d} | "
        f"Material: {mat['name']} | Working Load: {force_mag:.0f} N | Deflection Delta: {defl_delta:.1f}%",
        fontsize=13, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[Stress Recovery Visualizer] Saved: {os.path.abspath(output_path)}")
    return output_path


# =========================================================================
# MODE 5: MULTI-SAMPLE STATISTICAL SCORECARD
# =========================================================================
def visualize_batch_scorecard(dataset_path="topo_dataset_1000.npz", num_samples=50,
                              model=None, device=None, output_path="images/unet_viz_scorecard.png"):
    if model is None:
        model, device = load_model()

    if not os.path.exists(dataset_path):
        dataset_path = "topo_dataset.npz"

    data = np.load(dataset_path)
    inputs = data['inputs'][:num_samples]
    targets = data['targets'][:num_samples]

    dices, mses, vf_targets, vf_preds, latencies = [], [], [], [], []

    print(f"Evaluating scorecard across {len(inputs)} samples...")
    for i in range(len(inputs)):
        inp = inputs[i]
        if inp.shape[0] == 5:
            inp = np.concatenate([inp[0:1], inp[1:2], inp[1:2], inp[2:5]], axis=0)
        tgt = targets[i]

        tx = torch.from_numpy(inp).unsqueeze(0).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            pt = model(tx)
        pred = pt.squeeze().cpu().numpy()
        lat = (time.perf_counter() - t0) * 1000.0

        bin_p = (pred >= 0.5).astype(float)
        bin_t = (tgt >= 0.5).astype(float)

        inter = np.sum(bin_p * bin_t)
        dice = (2.0 * inter + 1e-6) / (np.sum(bin_p) + np.sum(bin_t) + 1e-6)
        mse = float(np.mean((pred - tgt) ** 2))

        tvf = float(np.mean(inp[-1]))
        pvf = float(np.mean(pred))

        dices.append(dice)
        mses.append(mse)
        vf_targets.append(tvf)
        vf_preds.append(pvf)
        latencies.append(lat)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=160)

    # 1. Dice Score Distribution
    ax0 = axes[0, 0]
    ax0.hist(dices, bins=15, color='#1976D2', edgecolor='white', alpha=0.85)
    ax0.axvline(np.mean(dices), color='red', linestyle='--', lw=2,
                label=f"Mean: {np.mean(dices):.3f} +/- {np.std(dices):.3f}")
    ax0.set_title("1. Dice Similarity Score Distribution", fontsize=11, fontweight='bold', pad=8)
    ax0.set_xlabel("Dice Score (0 to 1)", fontsize=10)
    ax0.set_ylabel("Count", fontsize=10)
    ax0.legend(fontsize=10)
    ax0.grid(True, linestyle=':', alpha=0.6)

    # 2. Volume Fraction Parity Plot
    ax1 = axes[0, 1]
    ax1.scatter(vf_targets, vf_preds, color='#388E3C', alpha=0.75, s=40, edgecolors='k', lw=0.5)
    ax1.plot([0.2, 0.6], [0.2, 0.6], 'r--', lw=2, label='Ideal Parity (Pred = Target)')
    ax1.set_title(f"2. Volume Fraction Parity\nMean Error: {np.mean(np.abs(np.array(vf_preds)-np.array(vf_targets))):.4f}",
                  fontsize=11, fontweight='bold', pad=8)
    ax1.set_xlabel(r"Target $V_f$", fontsize=10)
    ax1.set_ylabel(r"Predicted $V_f$", fontsize=10)
    ax1.set_xlim(0.2, 0.6)
    ax1.set_ylim(0.2, 0.6)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 3. MSE Distribution
    ax2 = axes[1, 0]
    ax2.hist(mses, bins=15, color='#7B1FA2', edgecolor='white', alpha=0.85)
    ax2.axvline(np.mean(mses), color='yellow', linestyle='--', lw=2,
                label=f"Mean MSE: {np.mean(mses):.5f}")
    ax2.set_title("3. Mean Squared Error (MSE) Distribution", fontsize=11, fontweight='bold', pad=8)
    ax2.set_xlabel("MSE", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 4. Inference Latency Distribution
    ax3 = axes[1, 1]
    ax3.hist(latencies, bins=15, color='#E65100', edgecolor='white', alpha=0.85)
    ax3.axvline(np.mean(latencies), color='blue', linestyle='--', lw=2,
                label=f"Mean: {np.mean(latencies):.2f} ms ({1000/np.mean(latencies):.1f} FPS)")
    ax3.set_title(f"4. Real-Time Latency Benchmark ({device})\nTarget: < 100 ms",
                  fontsize=11, fontweight='bold', pad=8)
    ax3.set_xlabel("Latency (ms)", fontsize=10)
    ax3.set_ylabel("Count", fontsize=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle(
        f"TopoUNet Comprehensive Statistical Scorecard | {len(inputs)} Samples Tested | "
        f"Dataset: {os.path.basename(dataset_path)}",
        fontsize=13, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[Scorecard Visualizer] Saved: {os.path.abspath(output_path)}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="TopoUNet Advanced Engineering Visualizer")
    parser.add_argument("--mode", type=str, default="comparison",
                        choices=["comparison", "sweep_vf", "sweep_threshold", "stress", "batch"],
                        help="Visualization mode")
    parser.add_argument("--case", type=str, default="cantilever", choices=["cantilever", "mbb", "bridge", "crane"])
    parser.add_argument("--sample_idx", type=int, default=None, help="Dataset sample index")
    parser.add_argument("--volfrac", type=float, default=0.40, help="Target volume fraction")
    parser.add_argument("--threshold", type=float, default=0.50, help="Binarization cutoff threshold")
    parser.add_argument("--material", type=str, default="steel", help="Material preset for stress mode")
    parser.add_argument("--force", type=float, default=1500.0, help="Force magnitude (N) for stress mode")
    parser.add_argument("--dataset", type=str, default="topo_dataset_1000.npz", help="Dataset path")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples for batch scorecard")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_unet.pth", help="Model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output image file path")
    args = parser.parse_args()

    model, device = load_model(args.checkpoint)

    if args.mode == "comparison":
        inp, simp, title, tvf, _ = get_sample_data(args.sample_idx, args.case, args.volfrac, args.dataset)
        out = args.output or f"images/unet_comparison_{'sample_' + str(args.sample_idx) if args.sample_idx is not None else args.case}.png"
        visualize_comparison(inp, simp, title, tvf, model, device, threshold=args.threshold, output_path=out)

    elif args.mode == "sweep_vf":
        out = args.output or f"images/unet_sweep_vf_{args.case}.png"
        visualize_vf_sweep(case_name=args.case, model=model, device=device, output_path=out)

    elif args.mode == "sweep_threshold":
        out = args.output or f"images/unet_sweep_threshold_{'sample_' + str(args.sample_idx) if args.sample_idx is not None else args.case}.png"
        visualize_threshold_sweep(sample_idx=args.sample_idx, case_name=args.case, volfrac=args.volfrac,
                                  model=model, device=device, output_path=out)

    elif args.mode == "stress":
        s_idx = 1 if args.sample_idx is None else args.sample_idx
        out = args.output or f"images/unet_stress_sample_{s_idx}_{args.material}.png"
        visualize_stress_recovery(sample_idx=s_idx, material=args.material, force_mag=args.force,
                                  model=model, device=device, output_path=out)

    elif args.mode == "batch":
        out = args.output or "images/unet_batch_scorecard.png"
        visualize_batch_scorecard(dataset_path=args.dataset, num_samples=args.num_samples,
                                  model=model, device=device, output_path=out)


if __name__ == '__main__':
    main()
