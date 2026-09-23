"""
visualize_evolution.py - SIMP Topology Optimization Evolution Visualizer
Renders the design progression every 10 iterations (e.g. Iter 0, 10, 20, ..., 80).
Generates:
1. High-resolution multi-panel static figure showing the optimization snapshots
   and the compliance / density change convergence curves.
2. An animated GIF showing the frame-by-frame structural evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import argparse
import os
from PIL import Image

import dataset_generator as dg
from visualize_data import extract_discrete_loads, draw_domain_and_bcs, draw_loads


def run_and_visualize_evolution(case="cantilever", interval=10, max_iter=80,
                               output_dir="images", make_gif=True, dpi=200):
    os.makedirs(output_dir, exist_ok=True)
    
    # Map case string to benchmark ID or random
    bm_map = {
        "cantilever": 0,
        "mbb": 1,
        "bridge": 2,
        "crane": 3,
        "random": None
    }
    bm_type = bm_map.get(case.lower(), 0)
    case_name = case.upper() if case.lower() != "random" else "RANDOM POISSON FIXTURES"
    
    print(f"\n========================================================")
    print(f" Running SIMP Optimization Evolution for '{case_name}'")
    print(f" Recording snapshots every {interval} iterations up to {max_iter}...")
    print(f"========================================================")
    
    # Sample problem with history recording enabled
    res = dg.sample_problem(
        sample_id=42 if bm_type is not None else 105,
        nelx=120,
        nely=60,
        penal=3.0,
        rmin=1.5,
        max_iter=max_iter,
        record_history=True,
        history_interval=interval,
        benchmark_type=bm_type
    )
    
    if res is None:
        print("Optimization failed or returned singular system.")
        return
        
    input_tensor, final_density, history = res
    nelx = 120
    nely = 60
    
    ch_domain, ch_bc, ch_fx, ch_fy, ch_vf = input_tensor
    volfrac = float(ch_vf[0, 0])
    loads = extract_discrete_loads(ch_fx, ch_fy)
    
    iterations = [h['iteration'] for h in history]
    compliances = [h['compliance'] for h in history if h['compliance'] is not None]
    comp_iters = [h['iteration'] for h in history if h['compliance'] is not None]
    changes = [h['change'] for h in history]
    
    print(f"Captured {len(history)} snapshots at iterations: {iterations}")
    if compliances:
        print(f"Initial compliance (Iter {comp_iters[0]}): {compliances[0]:.2f}")
        print(f"Final compliance   (Iter {comp_iters[-1]}): {compliances[-1]:.2f}")
        print(f"Compliance reduction: {((compliances[0] - compliances[-1]) / compliances[0]) * 100:.1f}%")
        
    # --- 1. MULTI-PANEL EVOLUTION FIGURE ---
    # Setup layout:
    # Top: Problem Specification (BCs & Loads)
    # Middle: 2 rows of snapshots (e.g. Iter 0, 10, 20, 30, 40 on row 1; Iter 50, 60, 70, 80 on row 2)
    # Bottom: Convergence curves (Compliance & Change)
    
    num_snaps = len(history)
    ncols = 5
    nrows_snaps = int(np.ceil(num_snaps / ncols))
    
    fig = plt.figure(figsize=(18, 3.5 * nrows_snaps + 6), facecolor="#F8F9FA")
    gs = GridSpec(nrows_snaps + 2, ncols, figure=fig, height_ratios=[1.2] + [1.0]*nrows_snaps + [1.1], hspace=0.35, wspace=0.2)
    
    fig.suptitle(
        f"SIMP Topology Optimization Progression (Every {interval} Iterations)\n"
        f"Problem: {case_name} | Grid: {nelx}x{nely} | Target Vf = {volfrac:.2f} | Filter Radius r_min = 1.5",
        fontsize=15, fontweight="bold", y=0.98, color="#1A237E"
    )
    
    # --- Top Banner: Problem Boundary Conditions & Loads ---
    ax_setup = fig.add_subplot(gs[0, :3])
    ax_setup.set_title("1. Problem Physics: Boundary Supports & Applied Loads", fontsize=11, fontweight="bold", pad=32)
    draw_domain_and_bcs(ax_setup, ch_bc, nelx, nely)
    draw_loads(ax_setup, loads)
    ax_setup.set_xlim(-15, nelx + 15)
    ax_setup.set_ylim(nely + 24, -22)
    ax_setup.axis("off")
    
    # Information Box
    ax_info = fig.add_subplot(gs[0, 3:])
    ax_info.axis("off")
    info_text = (
        f"OPTIMIZATION PARAMETERS:\n"
        f"• Algorithm: Solid Isotropic Material with Penalization (SIMP)\n"
        f"• Penalization power (p): 3.0\n"
        f"• Volume constraint (Vf): {volfrac:.2f} ({volfrac*100:.1f}% solid material)\n"
        f"• Filter: Sigmund Conical Sensitivity Filter (r_min = 1.5 elements)\n"
        f"• Optimizer: Optimality Criteria (OC) with 1D Bisection Search\n"
        f"• Convergence criteria: max(|Δρ|) <= 0.01 or max_iter = {max_iter}\n"
        f"• Final Status: {'Converged' if changes[-1] <= 0.01 else 'Reached Max Iterations'} "
        f"({iterations[-1]} iterations)"
    )
    ax_info.text(0.05, 0.5, info_text, transform=ax_info.transAxes,
                 fontsize=9.5, fontfamily="monospace", va="center",
                 bbox=dict(boxstyle="round,pad=0.8", facecolor="#E8EAF6", edgecolor="#3F51B5", lw=1.5))
    
    # --- Snapshots Grid ---
    for idx, snap in enumerate(history):
        row = 1 + idx // ncols
        col = idx % ncols
        ax_snap = fig.add_subplot(gs[row, col])
        
        it = snap['iteration']
        dens = snap['density']
        c_val = snap['compliance']
        ch_val = snap['change']
        
        # Invert grayscale so 1 (solid) is black and 0 (void) is white
        ax_snap.imshow(1.0 - dens, cmap='gray', extent=[0, nelx, nely, 0], vmin=0, vmax=1)
        rect = mpatches.Rectangle((0, 0), nelx, nely, fill=False, edgecolor='#333333', lw=1.2)
        ax_snap.add_patch(rect)
        
        # Subplot title
        c_str = f"c={c_val:.1f}" if c_val is not None else "c=N/A"
        ax_snap.set_title(f"Iter {it:02d} ({c_str})", fontsize=10, fontweight="bold",
                          color="#0D47A1" if it > 0 else "#424242")
        ax_snap.set_xlabel(f"Δρ = {ch_val:.3f}", fontsize=8.5, color="#555555")
        ax_snap.set_xticks([])
        ax_snap.set_yticks([])
        
    # Blank any unused snapshot cells in grid
    for idx in range(num_snaps, nrows_snaps * ncols):
        row = 1 + idx // ncols
        col = idx % ncols
        ax_empty = fig.add_subplot(gs[row, col])
        ax_empty.axis("off")
        
    # --- Bottom: Convergence History Curves ---
    ax_curve1 = fig.add_subplot(gs[nrows_snaps + 1, :])
    ax_curve2 = ax_curve1.twinx()
    
    # Plot compliance
    if compliances:
        l1 = ax_curve1.plot(comp_iters, compliances, color="#D32F2F", lw=2.2, marker="o", markersize=4, label="Compliance c (Strain Energy)")
        ax_curve1.set_ylabel("Compliance (Lower = Stiffer)", color="#D32F2F", fontsize=10, fontweight="bold")
        ax_curve1.tick_params(axis='y', labelcolor="#D32F2F")
    
    # Plot max density change
    l2 = ax_curve2.plot(iterations[1:], changes[1:], color="#1976D2", lw=2.0, linestyle="--", marker="s", markersize=4, label="Max Density Change Δρ")
    ax_curve2.axhline(0.01, color="#388E3C", linestyle=":", lw=1.5, label="Convergence Threshold (0.01)")
    ax_curve2.set_ylabel("Max Density Change Δρ", color="#1976D2", fontsize=10, fontweight="bold")
    ax_curve2.tick_params(axis='y', labelcolor="#1976D2")
    
    ax_curve1.set_xlabel("Iteration Number", fontsize=11, fontweight="bold")
    ax_curve1.set_xticks(iterations)
    ax_curve1.grid(True, linestyle="--", alpha=0.5)
    ax_curve1.set_title("Optimization Convergence History", fontsize=11, fontweight="bold")
    
    # Combined legend
    lines = (l1 if compliances else []) + l2 + [ax_curve2.lines[-1]]
    labels = [l.get_label() for l in lines]
    ax_curve1.legend(lines, labels, loc="upper right", framealpha=0.9, fontsize=9)
    
    output_png = os.path.join(output_dir, f"evolution_{case.lower()}.png")
    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Multi-panel evolution figure saved: {output_png}")
    
    # --- 2. ANIMATED GIF (FRAME BY FRAME) ---
    if make_gif:
        gif_frames = []
        for idx, snap in enumerate(history):
            fig_frame, ax_f = plt.subplots(figsize=(8, 4.5), dpi=120, facecolor="#FFFFFF")
            it = snap['iteration']
            dens = snap['density']
            c_val = snap['compliance']
            ch_val = snap['change']
            
            # Show density field
            ax_f.imshow(1.0 - dens, cmap='gray', extent=[0, nelx, nely, 0], vmin=0, vmax=1)
            rect = mpatches.Rectangle((0, 0), nelx, nely, fill=False, edgecolor='#222222', lw=2.0)
            ax_f.add_patch(rect)
            
            # Overlay boundary condition dots
            bc_y, bc_x = np.where(ch_bc > 0.05)
            ax_f.scatter(bc_x, bc_y, color='#D32F2F', s=18, alpha=0.7, zorder=5, label="Supports")
            
            # Title banner
            c_txt = f"{c_val:.2f}" if c_val is not None else "Initial"
            title_txt = f"SIMP Optimization | Iteration: {it:02d} / {max_iter}\nCompliance: {c_txt} | Max Δρ: {ch_val:.4f} | Target Vf: {volfrac:.2f}"
            ax_f.set_title(title_txt, fontsize=11, fontweight="bold", color="#1A237E")
            ax_f.set_xlim(-5, nelx + 5)
            ax_f.set_ylim(nely + 5, -5)
            ax_f.axis("off")
            
            fig_frame.canvas.draw()
            frame_img = Image.frombuffer(
                "RGBA",
                fig_frame.canvas.get_width_height(),
                fig_frame.canvas.buffer_rgba()
            )
            gif_frames.append(frame_img.convert("RGB"))
            plt.close(fig_frame)
            
        # Hold last frame longer
        if gif_frames:
            gif_frames += [gif_frames[-1]] * 4
            output_gif = os.path.join(output_dir, f"evolution_{case.lower()}.gif")
            gif_frames[0].save(
                output_gif,
                save_all=True,
                append_images=gif_frames[1:],
                duration=600,  # 600ms per frame
                loop=0
            )
            print(f"[SUCCESS] Animated evolution GIF saved: {output_gif}")
            
    return output_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIMP Optimization Evolution Visualizer")
    parser.add_argument("--case", type=str, default="cantilever",
                        choices=["cantilever", "mbb", "bridge", "crane", "random", "all"],
                        help="Structural benchmark case to run and visualize")
    parser.add_argument("--interval", type=int, default=10, help="Iteration interval to record")
    parser.add_argument("--max_iter", type=int, default=80, help="Maximum SIMP iterations")
    parser.add_argument("--output_dir", type=str, default="images", help="Output directory for plots")
    parser.add_argument("--no_gif", action="store_true", help="Disable GIF generation")
    args = parser.parse_args()
    
    cases_to_run = ["cantilever", "mbb"] if args.case == "all" else [args.case]
    for c in cases_to_run:
        run_and_visualize_evolution(
            case=c,
            interval=args.interval,
            max_iter=args.max_iter,
            output_dir=args.output_dir,
            make_gif=not args.no_gif
        )
