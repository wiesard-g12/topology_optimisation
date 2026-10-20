"""
interactive_demo.py - Interactive CLI & Visual Demonstration of TopoUNet (Phase 2)
Allows instantaneous real-time prediction on standard benchmark structural cases
(Cantilever, MBB Beam, Bridge Truss, Crane) or any sample from topo_dataset.npz.

Usage:
  python interactive_demo.py --case cantilever --volfrac 0.40
  python interactive_demo.py --case mbb --volfrac 0.45
  python interactive_demo.py --case bridge --volfrac 0.35
  python interactive_demo.py --sample_idx 3
  python interactive_demo.py --case cantilever --volfrac 0.40 --compare_simp
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
from dataset_generator import precompute_mesh_and_filter, solve_simp
from visualize_data import extract_discrete_loads, draw_loads, draw_domain_and_bcs


def build_benchmark_problem(case_name, volfrac=0.40, nelx=120, nely=60, load_angle=None):
    """
    Builds a standardized 6-channel physical tensor for standard structural benchmarks:
      Channel 0: domain (solid mask)
      Channel 1: bc_x (Ux = 0 constraint)
      Channel 2: bc_y (Uy = 0 constraint)
      Channel 3: Fx
      Channel 4: Fy
      Channel 5: Vf
    """
    ch_domain = np.ones((nely, nelx), dtype=np.float32)
    fixed_grid_x = np.zeros((nely + 1, nelx + 1), dtype=np.float32)
    fixed_grid_y = np.zeros((nely + 1, nelx + 1), dtype=np.float32)
    Fx_grid = np.zeros((nely + 1, nelx + 1), dtype=np.float32)
    Fy_grid = np.zeros((nely + 1, nelx + 1), dtype=np.float32)

    fixed_dofs = []
    def get_nid(x, y):
        return x * (nely + 1) + y

    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof)

    case_name = case_name.lower()

    if case_name == 'cantilever':
        # Top-left and bottom-left discrete pin supports (fix X and Y)
        p1 = (0, 0)
        p2 = (0, nely)
        fixed_grid_x[p1[1], p1[0]] = 1.0
        fixed_grid_y[p1[1], p1[0]] = 1.0
        fixed_grid_x[p2[1], p2[0]] = 1.0
        fixed_grid_y[p2[1], p2[0]] = 1.0
        n1, n2 = get_nid(*p1), get_nid(*p2)
        fixed_dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]

        # Downward load at middle right edge
        lx, ly = nelx, nely // 2
        ln = get_nid(lx, ly)
        angle = -np.pi / 2 if load_angle is None else np.radians(load_angle)
        fx, fy = float(np.cos(angle)), float(np.sin(angle))
        Fx_grid[ly, lx] = fx
        Fy_grid[ly, lx] = fy
        F[2*ln] = fx
        F[2*ln + 1] = fy

    elif case_name == 'mbb':
        # Left wall symmetry roller (fix X), bottom-right vertical roller (fix Y)
        for y in [0, nely // 2, nely]:
            fixed_grid_x[y, 0] = 1.0
            n = get_nid(0, y)
            fixed_dofs.append(2*n) # fix X

        fixed_grid_y[nely, nelx] = 1.0
        n_br = get_nid(nelx, nely)
        fixed_dofs.append(2*n_br + 1) # fix Y

        # Downward load at top-left corner
        lx, ly = 0, 0
        ln = get_nid(lx, ly)
        angle = -np.pi / 2 if load_angle is None else np.radians(load_angle)
        fx, fy = float(np.cos(angle)), float(np.sin(angle))
        Fx_grid[ly, lx] = fx
        Fy_grid[ly, lx] = fy
        F[2*ln] = fx
        F[2*ln + 1] = fy

    elif case_name == 'bridge':
        # Bottom-left pin (fix X, Y), bottom-right roller (fix Y)
        p1 = (0, nely)
        p2 = (nelx, nely)
        fixed_grid_x[p1[1], p1[0]] = 1.0
        fixed_grid_y[p1[1], p1[0]] = 1.0
        fixed_grid_y[p2[1], p2[0]] = 1.0
        n1, n2 = get_nid(*p1), get_nid(*p2)
        fixed_dofs = [2*n1, 2*n1+1, 2*n2+1]

        # Downward load at center top / deck
        lx, ly = nelx // 2, 0
        ln = get_nid(lx, ly)
        angle = -np.pi / 2 if load_angle is None else np.radians(load_angle)
        fx, fy = float(np.cos(angle)), float(np.sin(angle))
        Fx_grid[ly, lx] = fx
        Fy_grid[ly, lx] = fy
        F[2*ln] = fx
        F[2*ln + 1] = fy

    elif case_name == 'crane':
        # Top-left and bottom-left pin supports (fix X, Y)
        p1 = (0, 0)
        p2 = (0, nely)
        fixed_grid_x[p1[1], p1[0]] = 1.0
        fixed_grid_y[p1[1], p1[0]] = 1.0
        fixed_grid_x[p2[1], p2[0]] = 1.0
        fixed_grid_y[p2[1], p2[0]] = 1.0
        n1, n2 = get_nid(*p1), get_nid(*p2)
        fixed_dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]

        # Load at bottom-right corner
        lx, ly = nelx, nely
        ln = get_nid(lx, ly)
        angle = -np.pi / 2 if load_angle is None else np.radians(load_angle)
        fx, fy = float(np.cos(angle)), float(np.sin(angle))
        Fx_grid[ly, lx] = fx
        Fy_grid[ly, lx] = fy
        F[2*ln] = fx
        F[2*ln + 1] = fy
    else:
        raise ValueError(f"Unknown case: {case_name}. Available: cantilever, mbb, bridge, crane")

    # Map vertices to cells (60, 120)
    ch_bc_x = 0.25 * (
        fixed_grid_x[:-1, :-1] + fixed_grid_x[1:, :-1] +
        fixed_grid_x[:-1, 1:] + fixed_grid_x[1:, 1:]
    )
    ch_bc_y = 0.25 * (
        fixed_grid_y[:-1, :-1] + fixed_grid_y[1:, :-1] +
        fixed_grid_y[:-1, 1:] + fixed_grid_y[1:, 1:]
    )
    ch_fx = 0.25 * (
        Fx_grid[:-1, :-1] + Fx_grid[1:, :-1] +
        Fx_grid[:-1, 1:] + Fx_grid[1:, 1:]
    )
    ch_fy = 0.25 * (
        Fy_grid[:-1, :-1] + Fy_grid[1:, :-1] +
        Fy_grid[:-1, 1:] + Fy_grid[1:, 1:]
    )
    ch_vf = np.full((nely, nelx), volfrac, dtype=np.float32)

    # 6-channel dense input tensor
    input_tensor = np.stack([ch_domain, ch_bc_x, ch_bc_y, ch_fx, ch_fy, ch_vf], axis=0).astype(np.float32)
    meta = {
        'fixed_dofs': np.unique(fixed_dofs),
        'F': F,
        'volfrac': volfrac,
        'nelx': nelx,
        'nely': nely,
        'case_name': case_name
    }
    return input_tensor, meta


def run_demo(case_name='cantilever', volfrac=0.40, sample_idx=None, dataset_path="topo_dataset_1000.npz",
             checkpoint_path="checkpoints/best_unet.pth", compare_simp=False, simp_iters=40,
             output_path=None, threshold=0.35):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Model
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

    simp_density = None
    simp_time = 0.0

    # 2. Prepare Input Problem
    if sample_idx is not None:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset {dataset_path} not found!")
        data = np.load(dataset_path)
        input_tensor = data['inputs'][sample_idx]
        if input_tensor.shape[0] == 5:
            # Upgrade on-the-fly
            input_tensor = np.concatenate([input_tensor[0:1], input_tensor[1:2], input_tensor[1:2], input_tensor[2:5]], axis=0)

        simp_density = data['targets'][sample_idx]
        title_case = f"Dataset Sample #{sample_idx:03d}"
        volfrac = float(np.mean(input_tensor[-1]))
        nelx, nely = input_tensor.shape[2], input_tensor.shape[1]
    else:
        input_tensor, meta = build_benchmark_problem(case_name, volfrac=volfrac)
        title_case = f"Benchmark Case: {case_name.upper()}"
        nelx, nely = meta['nelx'], meta['nely']

        if compare_simp:
            print(f"Running SIMP solver ({simp_iters} iterations) for ground-truth comparison...")
            t0 = time.time()
            edofMat, iK, jK, H, Hs = precompute_mesh_and_filter(nelx, nely, rmin=1.5)
            simp_density = solve_simp(
                nelx, nely, volfrac, penal=3.0,
                fixeddofs=meta['fixed_dofs'], F=meta['F'],
                edofMat=edofMat, iK=iK, jK=jK, H=H, Hs=Hs,
                max_iter=simp_iters
            )
            simp_time = time.time() - t0
            print(f"SIMP solved in {simp_time:.2f}s.")

    # 3. Real-Time PI-UNet Inference
    tensor_x = torch.from_numpy(input_tensor).unsqueeze(0).to(device)

    # Measure latency
    t_start = time.perf_counter()
    with torch.no_grad():
        pred_tensor = model(tensor_x)
    pred_density = pred_tensor.squeeze().cpu().numpy()
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    pred_vf = float(np.mean(pred_density))
    bin_pred = (pred_density >= threshold).astype(float)
    bin_vf = float(np.mean(bin_pred))

    print("=" * 65)
    print(f"  TopoUNet / PI-UNet Prediction Demo: {title_case}")
    print(f"  Target Volume Fraction Vf: {volfrac:.3f}")
    print(f"  Predicted Volume Fraction: {pred_vf:.3f} (Error: {abs(pred_vf - volfrac):.4f})")
    print(f"  Binarized Volume Fraction: {bin_vf:.3f} (Threshold: {threshold})")
    print(f"  Inference Latency:         {latency_ms:.2f} ms ({device})")
    if simp_density is not None:
        bin_simp = (simp_density >= 0.5).astype(float)
        inter = np.sum(bin_pred * bin_simp)
        dice = (2.0 * inter + 1e-6) / (np.sum(bin_pred) + np.sum(bin_simp) + 1e-6)
        print(f"  Dice Similarity with SIMP: {dice:.3f}")
        if simp_time > 0:
            speedup = (simp_time * 1000.0) / latency_ms
            print(f"  Instant Speedup vs SIMP:   {speedup:.0f}x faster!")
    print("=" * 65)

    # 4. Render Visual Figure
    n_cols = 4 if simp_density is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.5), dpi=180)

    # Extract BC and loads for drawing
    if input_tensor.shape[0] == 6:
        ch_bc = np.clip(input_tensor[1] + input_tensor[2], 0.0, 1.0)
        ch_fx = input_tensor[3]
        ch_fy = input_tensor[4]
    else:
        ch_bc = input_tensor[1]
        ch_fx = input_tensor[2]
        ch_fy = input_tensor[3]

    # Panel 1: Problem Boundary Conditions & Loads
    ax0 = axes[0]
    draw_domain_and_bcs(ax0, ch_bc)
    loads = extract_discrete_loads(ch_fx, ch_fy)
    draw_loads(ax0, loads)
    ax0.set_title(f"1. Problem Setup\nTarget $V_f = {volfrac:.2f}$", fontsize=11, fontweight='bold', pad=8)
    ax0.set_xlim(-15, nelx + 15)
    ax0.set_ylim(nely + 10, -10)
    ax0.set_aspect('equal')
    ax0.axis('off')

    # Panel 2: U-Net Continuous Output
    ax1 = axes[1]
    ax1.imshow(1.0 - pred_density, cmap='gray', vmin=0, vmax=1,
               extent=[0, nelx, nely, 0], interpolation='nearest')
    ax1.set_title(rf"2. PI-UNet Continuous ($\hat{{y}}$)" + "\n" + f"Latency: {latency_ms:.1f} ms | $V_f = {pred_vf:.3f}$",
                  fontsize=11, fontweight='bold', pad=8)
    ax1.set_xlim(-5, nelx + 5)
    ax1.set_ylim(nely + 5, -5)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Panel 3: Binarized Structural Topology
    ax2 = axes[2]
    ax2.imshow(1.0 - bin_pred, cmap='gray', vmin=0, vmax=1,
               extent=[0, nelx, nely, 0], interpolation='nearest')
    ax2.set_title(rf"3. Binarized Truss ($\hat{{y}} \geq {threshold}$)" + "\n" + f"$V_f = {bin_vf:.3f}$",
                  fontsize=11, fontweight='bold', pad=8)
    ax2.set_xlim(-5, nelx + 5)
    ax2.set_ylim(nely + 5, -5)
    ax2.set_aspect('equal')
    ax2.axis('off')

    # Panel 4 (Optional): SIMP Comparison
    if simp_density is not None:
        ax3 = axes[3]
        ax3.imshow(1.0 - simp_density, cmap='gray', vmin=0, vmax=1,
                   extent=[0, nelx, nely, 0], interpolation='nearest')
        ax3.set_title(f"4. Ground Truth SIMP\n$V_f = {float(np.mean(simp_density)):.3f}$ | Dice: {dice:.3f}",
                      fontsize=11, fontweight='bold', pad=8)
        ax3.set_xlim(-5, nelx + 5)
        ax3.set_ylim(nely + 5, -5)
        ax3.set_aspect('equal')
        ax3.axis('off')

    fig.suptitle(
        f"Real-Time Deep Learning Topology Optimization | {title_case} | "
        f"Inference: {latency_ms:.2f} ms",
        fontsize=13, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if output_path is None:
        os.makedirs("images", exist_ok=True)
        tag = f"sample_{sample_idx:03d}" if sample_idx is not None else case_name
        output_path = os.path.join("images", f"demo_{tag}.png")

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved demo figure: {os.path.abspath(output_path)}")
    return output_path, latency_ms, pred_vf


def main():
    parser = argparse.ArgumentParser(description="Real-Time TopoUNet Interactive Demo")
    parser.add_argument("--case", type=str, default="cantilever", choices=["cantilever", "mbb", "bridge", "crane"],
                        help="Benchmark structural case")
    parser.add_argument("--volfrac", type=float, default=0.40, help="Target volume fraction (0.20 to 0.60)")
    parser.add_argument("--sample_idx", type=int, default=None, help="Inspect dataset sample directly by index")
    parser.add_argument("--dataset", type=str, default="topo_dataset_1000.npz", help="Dataset path")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_unet.pth", help="Checkpoint path")
    parser.add_argument("--compare_simp", action="store_true", help="Run SIMP solver side-by-side for verification")
    parser.add_argument("--simp_iters", type=int, default=40, help="SIMP iterations if compare_simp is enabled")
    parser.add_argument("--threshold", type=float, default=0.35, help="Binarization cutoff threshold")
    parser.add_argument("--output", type=str, default=None, help="Output plot path")
    args = parser.parse_args()

    run_demo(
        case_name=args.case,
        volfrac=args.volfrac,
        sample_idx=args.sample_idx,
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        compare_simp=args.compare_simp,
        simp_iters=args.simp_iters,
        output_path=args.output,
        threshold=args.threshold
    )


if __name__ == '__main__':
    main()
