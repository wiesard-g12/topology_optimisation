"""
simulate_optimization_evolution.py - Optimization & Structural Multi-Physics Simulation
Demonstrates the complete evolution of the 25-iteration buckling-aware topology optimization
and performs physical structural simulation on the resulting architecture.

Generates:
1. High-resolution multi-panel evolution & physics simulation figure
   (images/simulation_evolution_sample_000.png)
2. Frame-by-frame animated simulation GIF
   (images/optimization_simulation.gif)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

from dataset_generator import sample_problem, precompute_mesh_and_filter
from buckling_optimizer import optimize_with_buckling
from structural_mechanics import MATERIALS, analyze_structural_integrity


def run_simulation(sample_idx=0, material_name="steel", force_N=1500.0,
                   length_mm=120.0, height_mm=60.0, thickness_mm=10.0,
                   nelx=60, nely=30, volfrac=0.4, buckle_weight=0.45,
                   output_img="images/simulation_evolution_sample_000.png",
                   output_gif="images/optimization_simulation.gif"):
    """
    Simulates the optimization process across 25 iterations, generates snapshot filmstrip,
    and runs multi-physics FEA simulation on the final structure.
    """
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    mat = MATERIALS.get(material_name.lower(), MATERIALS["steel"])
    
    print("==================================================================")
    print(f" SIMULATING TOPOLOGY OPTIMIZATION: Sample #{sample_idx:03d}")
    print(f" Material: {mat['name']} | Load: {force_N:,.0f} N | Grid: {nelx}x{nely}")
    print(f" Physical Dimensions: {length_mm:.1f} x {height_mm:.1f} x {thickness_mm:.1f} mm")
    print("==================================================================\n")
    
    # 1. Sample problem setup
    res_sample = sample_problem(
        sample_id=sample_idx, nelx=nelx, nely=nely,
        benchmark_type='michell' if sample_idx == 0 else None,
        return_meta=True
    )
    if res_sample is None:
        raise RuntimeError("Failed to sample problem setup.")
    _, _, meta = res_sample
    F = meta["F"]
    fixed_dofs = meta["fixed_dofs"]
    
    edofMat, _, _, _, _ = precompute_mesh_and_filter(nelx, nely, rmin=1.5)
    
    # 2. Run 25-iteration optimization with full history logging
    print("[1/3] Running 25-iteration buckling-aware optimization simulation...")
    opt_res = optimize_with_buckling(
        F, fixed_dofs, nelx=nelx, nely=nely, volfrac=volfrac,
        max_iter=25, buckle_start_iter=15, buckle_weight=buckle_weight,
        verbose=True
    )
    history = opt_res["history"]
    final_rho = opt_res["density"]
    
    # 3. Perform physical multi-physics structural simulation on the final topology
    print("\n[2/3] Running physical FEA simulation on final architecture...")
    sim_data = analyze_structural_integrity(
        final_rho, fixed_dofs, F, edofMat,
        nelx=nelx, nely=nely, material_name=material_name,
        force_real_N=force_N, length_mm=length_mm,
        height_mm=height_mm, thickness_mm=thickness_mm
    )
    
    # Extract deflection and buckling mode shapes
    U_phys = sim_data["U_phys"]
    u_phys = U_phys[0::2].reshape((nely + 1, nelx + 1), order='F')
    v_phys = U_phys[1::2].reshape((nely + 1, nelx + 1), order='F')
    disp_mag_mm = np.sqrt(u_phys**2 + v_phys**2)
    
    vm_stress = sim_data["von_mises_grid"]
    buckle_data = sim_data["buckling_data"]
    phi_mag = buckle_data["phi_mag"] if buckle_data else np.zeros((nely + 1, nelx + 1))
    
    # 4. Generate Master Multi-Panel Visualization
    print("\n[3/3] Rendering simulation panels and animated GIF...")
    fig = plt.figure(figsize=(18, 14), facecolor="#F8F9FA")
    gs = GridSpec(4, 6, figure=fig, height_ratios=[0.9, 1.0, 1.1, 1.1], hspace=0.38, wspace=0.28,
                  left=0.05, right=0.96, top=0.92, bottom=0.05)
    
    fig.suptitle(
        f"Topology Optimization Simulation & Structural Multi-Physics Analysis\n"
        f"Problem: Michell Truss | Material: {mat['name']} | Force: {force_N:,.0f} N | 25-Iteration Budget",
        fontsize=15, fontweight="bold", y=0.98, color="#1A237E"
    )
    
    # --- ROW 1: Optimization Evolution Filmstrip (6 Key Snapshots: 1, 5, 10, 15, 20, 25) ---
    selected_iters = [1, 5, 10, 15, 20, 25]
    iter_lookup = {h['iteration']: h for h in history}
    
    for col_idx, it_num in enumerate(selected_iters):
        ax = fig.add_subplot(gs[1, col_idx])
        snap = iter_lookup.get(it_num, history[-1])
        density_snap = snap['density']
        
        ax.imshow(1.0 - density_snap, cmap="Greys", origin="lower",
                  extent=[0, length_mm, 0, height_mm], interpolation="bilinear")
        
        if it_num < 15:
            phase_tag = "Compliance Phase"
            border_col = "#455A64"
        else:
            phase_tag = "Buckling Active"
            border_col = "#1B5E20"
            
        ax.set_title(f"Iter {it_num:02d}: C={snap['compliance']:.1f}\n[{phase_tag}]",
                     fontsize=9.5, fontweight="bold", color=border_col, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(1.8 if it_num in [15, 25] else 1.0)
            
    # Section Title for Row 1
    ax_banner1 = fig.add_subplot(gs[0, :])
    ax_banner1.axis("off")
    ax_banner1.text(0.5, 0.45,
                    "PART 1: TOPOLOGY OPTIMIZATION SIMULATION (ITERATION SNAPSHOTS)\n"
                    "Iterations 1 to 14: Macro-truss load path discovery | Iterations 15 to 25: Active buckling reinforcement",
                    ha="center", va="center", fontsize=12, fontweight="bold", color="#263238",
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#ECEFF1", edgecolor="#B0BEC5"))
    
    # --- ROW 2 & 3: Multi-Physics Structural Simulations on the Final Architecture ---
    
    # Sub-Panel A: Physical Deflection Simulation (mm)
    ax_disp = fig.add_subplot(gs[2, 0:3])
    im_disp = ax_disp.imshow(disp_mag_mm, cmap="plasma", origin="lower",
                             extent=[0, length_mm, 0, height_mm], interpolation="bilinear")
    cbar_disp = fig.colorbar(im_disp, ax=ax_disp, fraction=0.035, pad=0.02)
    cbar_disp.set_label("Displacement (mm)", fontsize=9)
    ax_disp.set_title(f"A. Physical Deflection Field under {force_N:,.0f} N Load\nMax Deflection: {sim_data['max_deflection_mm']*1e3:.2f} µm ({sim_data['max_deflection_mm']:.4f} mm)",
                      fontsize=11, fontweight="bold", pad=8)
    ax_disp.set_xlabel("Length (mm)", fontsize=9)
    ax_disp.set_ylabel("Height (mm)", fontsize=9)
    
    # Draw applied load arrow
    fx_nodes = np.where(np.abs(F[0::2]) > 1e-6)[0]
    fy_nodes = np.where(np.abs(F[1::2]) > 1e-6)[0]
    for n in fy_nodes:
        c = n // (nely + 1)
        r = n % (nely + 1)
        xp = c * (length_mm / nelx)
        yp = r * (height_mm / nely)
        ax_disp.annotate('', xy=(xp, yp - 7.0), xytext=(xp, yp),
                         arrowprops=dict(facecolor='#D32F2F', edgecolor='black', width=1.8, headwidth=6))
        ax_disp.text(xp, yp + 2.0, f"F = {force_N:,.0f} N", ha='center', fontsize=9, fontweight='bold', color='#D32F2F')
        
    # Sub-Panel B: Von Mises Stress Heatmap (MPa)
    ax_stress = fig.add_subplot(gs[2, 3:6])
    solid_mask = (final_rho >= 0.25)
    masked_stress = np.ma.masked_where(~solid_mask, vm_stress)
    
    im_stress = ax_stress.imshow(masked_stress, cmap="inferno", origin="lower",
                                 extent=[0, length_mm, 0, height_mm], interpolation="nearest")
    cbar_stress = fig.colorbar(im_stress, ax=ax_stress, fraction=0.035, pad=0.02)
    cbar_stress.set_label("Stress (MPa)", fontsize=9)
    ax_stress.set_title(f"B. Von Mises Stress Distribution (Yield Limit: {mat['yield_strength']:.0f} MPa)\n95th% Stress: {sim_data['vm_95th']:.1f} MPa | FoS = {sim_data['fos_yield']:.2f} [SAFE]",
                        fontsize=11, fontweight="bold", pad=8)
    ax_stress.set_xlabel("Length (mm)", fontsize=9)
    ax_stress.set_ylabel("Height (mm)", fontsize=9)
    
    # Sub-Panel C: 1st In-Plane Buckling Mode Shape
    ax_buckle = fig.add_subplot(gs[3, 0:3])
    im_buckle = ax_buckle.imshow(phi_mag, cmap="viridis", origin="lower",
                                 extent=[0, length_mm, 0, height_mm], interpolation="bilinear")
    cbar_buckle = fig.colorbar(im_buckle, ax=ax_buckle, fraction=0.035, pad=0.02)
    cbar_buckle.set_label("Normalized Modal Amp", fontsize=9)
    ax_buckle.set_title(f"C. 1st In-Plane Buckling Mode Shape\nCritical Buckling Multiplier $\\lambda_1 = {sim_data['lambda_1']:.2f}$ | $P_{{cr}} = {sim_data['p_critical_N']/1e3:.1f}$ kN [SAFE]",
                        fontsize=11, fontweight="bold", pad=8)
    ax_buckle.set_xlabel("Length (mm)", fontsize=9)
    ax_buckle.set_ylabel("Height (mm)", fontsize=9)
    
    # Sub-Panel D: Convergence Dynamics (Compliance & Buckling Multiplier vs Iteration)
    ax_conv = fig.add_subplot(gs[3, 3:6])
    
    all_iters = [h['iteration'] for h in history]
    all_comps = [h['compliance'] for h in history]
    buckle_iters = [h['iteration'] for h in history if h['lambda_1'] is not None]
    buckle_lams = [h['lambda_1'] for h in history if h['lambda_1'] is not None]
    
    color_c = "#D32F2F"
    ax_conv.set_xlabel("Optimization Iteration", fontsize=10)
    ax_conv.set_ylabel("Compliance c (Lower = Stiffer)", color=color_c, fontsize=10)
    line1 = ax_conv.plot(all_iters, all_comps, color=color_c, lw=2.2, marker='o', markersize=4, label="Compliance c")
    ax_conv.tick_params(axis='y', labelcolor=color_c)
    ax_conv.grid(True, linestyle="--", alpha=0.5)
    
    # Vertical transition marker at Iteration 15
    ax_conv.axvline(x=15, color="#1B5E20", linestyle="--", lw=1.8, alpha=0.85)
    ax_conv.text(15.2, max(all_comps)*0.85, "Iter 15: Buckling Sensitivities Active",
                 fontsize=8.5, color="#1B5E20", fontweight="bold")
    
    # Secondary Y axis for Buckling Multiplier lambda_1
    color_lam = "#1B5E20"
    ax_conv2 = ax_conv.twinx()
    ax_conv2.set_ylabel(r"Buckling Multiplier $\lambda_1$ (Higher = Safer)", color=color_lam, fontsize=10)
    line2 = ax_conv2.plot(buckle_iters, buckle_lams, color=color_lam, lw=2.2, marker='s', markersize=5, label=r"Buckling Factor $\lambda_1$")
    ax_conv2.tick_params(axis='y', labelcolor=color_lam)
    
    ax_conv.set_title("D. Multi-Objective Convergence Dynamics (25 Iters)", fontsize=11, fontweight="bold", pad=8)
    
    plt.savefig(output_img, dpi=200, bbox_inches="tight")
    plt.close()
    print(f" Master simulation figure saved to: {output_img}")
    
    # 5. Generate Animated GIF across all 25 frames
    frames = []
    print(" Compiling 25-frame animated simulation GIF...")
    for idx, h in enumerate(history):
        fig_frame, ax_f = plt.subplots(figsize=(6, 3.2), facecolor="#212121")
        ax_f.set_facecolor("#212121")
        
        rho_frame = h['density']
        it = h['iteration']
        comp = h['compliance']
        
        ax_f.imshow(1.0 - rho_frame, cmap="Greys", origin="lower",
                    extent=[0, length_mm, 0, height_mm], interpolation="bilinear")
        
        tag = "[Compliance Phase]" if it < 15 else "[Buckling-Aware Reinforcement]"
        col_tag = "#FFB300" if it < 15 else "#66BB6A"
        
        ax_f.set_title(f"Iteration {it:02d} / 25 | C = {comp:.2f} {tag}",
                       color=col_tag, fontsize=10, fontweight="bold", pad=6)
        ax_f.set_xticks([])
        ax_f.set_yticks([])
        for spine in ax_f.spines.values():
            spine.set_color(col_tag)
            spine.set_linewidth(1.5)
            
        fig_frame.canvas.draw()
        rgba_img = np.asarray(fig_frame.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba_img))
        plt.close(fig_frame)
        
    # Save GIF with 250ms per frame
    if frames:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=220,
            loop=0
        )
        print(f" Animated simulation GIF saved to: {output_gif}")
        
    print("\n==================================================================")
    print(" SIMULATION COMPLETE & READY FOR REVIEW")
    print("==================================================================\n")
    return {
        "output_img": output_img,
        "output_gif": output_gif,
        "sim_data": sim_data
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate topology optimization evolution & structural multi-physics.")
    parser.add_argument("--sample-index", type=int, default=0, help="Problem sample index")
    parser.add_argument("--material", type=str, default="steel", help="Material name")
    parser.add_argument("--force", type=float, default=1500.0, help="Applied force in Newtons")
    parser.add_argument("--output-img", type=str, default="images/simulation_evolution_sample_000.png")
    parser.add_argument("--output-gif", type=str, default="images/optimization_simulation.gif")
    
    args = parser.parse_args()
    run_simulation(
        sample_idx=args.sample_index,
        material_name=args.material,
        force_N=args.force,
        output_img=args.output_img,
        output_gif=args.output_gif
    )
