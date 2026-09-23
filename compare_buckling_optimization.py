"""
compare_buckling_optimization.py - Head-to-Head Comparison Script
Compares:
1. Standard SIMP (25 iterations, pure compliance minimization)
2. Buckling-Aware SIMP (25 iterations, hybrid compliance + buckling reinforcement)

Generates a 4-panel visual comparison showing topology adaptation, density redistribution,
and quantitative metrics (compliance, stress, Factor of Safety, and critical buckling multiplier).
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm

from dataset_generator import sample_problem
from buckling_optimizer import optimize_with_buckling
from structural_mechanics import MATERIALS, analyze_structural_integrity, solve_inplane_buckling


def run_comparison(sample_idx=0, material_name="steel", force_N=1500.0,
                   length_mm=120.0, height_mm=60.0, thickness_mm=10.0,
                   nelx=60, nely=30, volfrac=0.4, buckle_weight=0.45,
                   output_path="images/buckling_comparison_sample_000.png"):
    """
    Executes both optimizers on the same problem setup and generates a comparative certificate.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mat = MATERIALS.get(material_name.lower(), MATERIALS["steel"])
    
    print(f"==================================================================")
    print(f" BUCKLING OPTIMIZATION COMPARISON: Sample #{sample_idx:03d} (25 ITERS)")
    print(f" Material: {mat['name']} | Force: {force_N:,.1f} N | Grid: {nelx}x{nely}")
    print(f" Domain: {length_mm:.1f} x {height_mm:.1f} x {thickness_mm:.1f} mm | Buckle Weight: {buckle_weight}")
    print(f"==================================================================\n")
    
    # 1. Sample problem setup (BCs, forces, mesh)
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
    
    from dataset_generator import precompute_mesh_and_filter
    edofMat, _, _, _, _ = precompute_mesh_and_filter(nelx, nely, rmin=1.5)
    
    # 2. Run Standard SIMP (w = 0.0)
    print("[1/2] Running Pure Compliance SIMP (25 iterations)...")
    res_std = optimize_with_buckling(
        F, fixed_dofs, nelx=nelx, nely=nely, volfrac=volfrac,
        max_iter=25, buckle_weight=0.0, verbose=True
    )
    
    # 3. Run Buckling-Aware SIMP (w = buckle_weight)
    print("\n[2/2] Running Buckling-Aware SIMP (25 iterations, buckle start @ iter 15)...")
    res_buckle = optimize_with_buckling(
        F, fixed_dofs, nelx=nelx, nely=nely, volfrac=volfrac,
        max_iter=25, buckle_start_iter=15, buckle_weight=buckle_weight, verbose=True
    )
    
    # 4. Perform Real-World Physical Structural Integrity Analysis
    print("\n[Analysis] Evaluating real-world physical stress & buckling safety...")
    ana_std = analyze_structural_integrity(
        res_std["density"], fixed_dofs, F, edofMat,
        nelx=nelx, nely=nely, material_name=material_name,
        force_real_N=force_N, length_mm=length_mm,
        height_mm=height_mm, thickness_mm=thickness_mm
    )
    
    ana_buckle = analyze_structural_integrity(
        res_buckle["density"], fixed_dofs, F, edofMat,
        nelx=nelx, nely=nely, material_name=material_name,
        force_real_N=force_N, length_mm=length_mm,
        height_mm=height_mm, thickness_mm=thickness_mm
    )
    
    rho_std = res_std["density"]
    rho_bck = res_buckle["density"]
    delta_rho = rho_bck - rho_std  # Positive = added material, Negative = removed
    
    # 5. Render Professional 4-Panel Comparison
    fig = plt.figure(figsize=(16, 11), facecolor="#F8F9FA")
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22,
                          left=0.06, right=0.96, top=0.91, bottom=0.06)
    
    # --- PANEL 1: Standard Pure Compliance Topology ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#EEEEEE")
    ax1.imshow(1.0 - rho_std, cmap="Greys", origin="lower",
               extent=[0, length_mm, 0, height_mm], interpolation="bilinear")
    ax1.set_title(f"A. Standard SIMP (Pure Compliance)\nCompliance: {res_std['final_compliance']:.2f} | $\\lambda_1$: {ana_std['lambda_1']:.2f}",
                  fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("Length (mm)", fontsize=10)
    ax1.set_ylabel("Height (mm)", fontsize=10)
    
    # Draw load location
    fx_nodes = np.where(np.abs(F[0::2]) > 1e-6)[0]
    fy_nodes = np.where(np.abs(F[1::2]) > 1e-6)[0]
    for n in fy_nodes:
        col = n // (nely + 1)
        row = n % (nely + 1)
        x_pt = col * (length_mm / nelx)
        y_pt = row * (height_mm / nely)
        ax1.annotate('', xy=(x_pt, y_pt - 8.0), xytext=(x_pt, y_pt),
                     arrowprops=dict(facecolor='#D32F2F', edgecolor='black', width=1.5, headwidth=6))
        ax1.text(x_pt, y_pt + 2.0, f"{force_N:,.0f} N", ha='center', fontsize=9, fontweight='bold', color='#D32F2F')
        
    # --- PANEL 2: Buckling-Aware Topology ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#EEEEEE")
    ax2.imshow(1.0 - rho_bck, cmap="Greys", origin="lower",
               extent=[0, length_mm, 0, height_mm], interpolation="bilinear")
    
    gain_pct = ((ana_buckle['lambda_1'] - ana_std['lambda_1']) / max(1e-6, ana_std['lambda_1'])) * 100.0
    sign = "+" if gain_pct >= 0 else ""
    ax2.set_title(f"B. Buckling-Aware SIMP (w = {buckle_weight})\nCompliance: {res_buckle['final_compliance']:.2f} | $\\lambda_1$: {ana_buckle['lambda_1']:.2f} ({sign}{gain_pct:.1f}%)",
                  fontsize=12, fontweight="bold", pad=8, color="#1B5E20")
    ax2.set_xlabel("Length (mm)", fontsize=10)
    ax2.set_ylabel("Height (mm)", fontsize=10)
    
    # --- PANEL 3: Material Redistribution Map (Delta Rho) ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#FFFFFF")
    
    # Two-slope diverging colormap: Blue = shed, Red = reinforced
    norm = TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6)
    im3 = ax3.imshow(delta_rho, cmap="coolwarm", norm=norm, origin="lower",
                     extent=[0, length_mm, 0, height_mm], interpolation="nearest")
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.035, pad=0.03)
    cbar3.set_label(r"Density Shift $\Delta \rho = \rho_{\mathrm{buckle}} - \rho_{\mathrm{std}}$", fontsize=9)
    ax3.set_title(r"C. Material Redistribution ($\Delta \rho$)" + "\nRed = Struts Reinforced against Buckling | Blue = Shed Tension Material",
                  fontsize=11, fontweight="bold", pad=8)
    ax3.set_xlabel("Length (mm)", fontsize=10)
    ax3.set_ylabel("Height (mm)", fontsize=10)
    
    # --- PANEL 4: Quantitative Dashboard & Executive Comparison ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    
    std_pass = ana_std["overall_pass"]
    bck_pass = ana_buckle["overall_pass"]
    
    table_data = [
        ["Metric", "Standard SIMP", "Buckling-Aware", "Impact"],
        ["Iterations", f"{res_std['history'][-1]['iteration']}", f"{res_buckle['history'][-1]['iteration']}", "25 (budget)"],
        ["Solve Time", f"{res_std['elapsed_time']:.2f} s", f"{res_buckle['elapsed_time']:.2f} s", f"+{res_buckle['elapsed_time']-res_std['elapsed_time']:.2f} s"],
        ["Compliance c", f"{res_std['final_compliance']:.2f}", f"{res_buckle['final_compliance']:.2f}", f"+{((res_buckle['final_compliance']-res_std['final_compliance'])/res_std['final_compliance'])*100:+.1f}% trade-off"],
        ["Buckling Factor lambda_1", f"{ana_std['lambda_1']:.2f}", f"{ana_buckle['lambda_1']:.2f}", f"{sign}{gain_pct:.1f}% gain"],
        ["Critical Load P_cr", f"{ana_std['p_critical_N']/1e3:.1f} kN", f"{ana_buckle['p_critical_N']/1e3:.1f} kN", f"{(ana_buckle['p_critical_N']-ana_std['p_critical_N'])/1e3:+.1f} kN"],
        ["95th% Stress", f"{ana_std['vm_95th']:.1f} MPa", f"{ana_buckle['vm_95th']:.1f} MPa", f"Yield: {mat['yield_strength']:.0f} MPa"],
        ["Yield Safety FoS", f"{ana_std['fos_yield']:.2f}", f"{ana_buckle['fos_yield']:.2f}", "Factor of Safety"],
        ["Overall Health", "[PASS] SAFE" if std_pass else "[FAIL]", "[PASS] REINFORCED" if bck_pass else "[FAIL]", "Certified"]
    ]
    
    t = ax4.table(cellText=table_data, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    t.scale(1.0, 1.45)
    
    # Style table header and cells
    for j in range(4):
        cell = t[(0, j)]
        cell.set_facecolor("#263238")
        cell.set_text_props(color="white", weight="bold")
    
    # Highlight buckling factor row
    for j in range(4):
        cell = t[(4, j)]
        cell.set_facecolor("#E8F5E9")
        cell.set_text_props(weight="bold", color="#1B5E20")
        
    ax4.set_title("D. Comparative Structural Performance", fontsize=12, fontweight="bold", pad=12)
    
    # Super Title
    fig.suptitle(
        f"25-Iteration Topology Optimization: Pure Compliance vs Buckling-Aware\n"
        f"Material: {mat['name']} | Load: {force_N:,.0f} N | Active Reinforcement Weight: {buckle_weight}",
        fontsize=14, fontweight="bold", y=0.98
    )
    
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"\n==================================================================")
    print(f" SUCCESS: Comparison report exported to: {output_path}")
    print(f" Standard lambda_1: {ana_std['lambda_1']:.2f} -> Buckling-Aware lambda_1: {ana_buckle['lambda_1']:.2f} ({sign}{gain_pct:.1f}%)")
    print(f" Solve time: Standard {res_std['elapsed_time']:.2f}s | Buckling-Aware {res_buckle['elapsed_time']:.2f}s")
    print(f"==================================================================\n")
    
    return {
        "res_std": res_std,
        "res_buckle": res_buckle,
        "ana_std": ana_std,
        "ana_buckle": ana_buckle,
        "gain_pct": gain_pct,
        "output_path": output_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 25-iteration standard vs buckling-aware topology optimization comparison.")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index / problem preset")
    parser.add_argument("--material", type=str, default="steel", help="Material name (steel, aluminum, titanium, petg)")
    parser.add_argument("--force", type=float, default=1500.0, help="Applied force magnitude in Newtons")
    parser.add_argument("--length", type=float, default=120.0, help="Physical length in mm")
    parser.add_argument("--height", type=float, default=60.0, help="Physical height in mm")
    parser.add_argument("--thickness", type=float, default=10.0, help="Physical thickness in mm")
    parser.add_argument("--nelx", type=int, default=60, help="Elements along X")
    parser.add_argument("--nely", type=int, default=30, help="Elements along Y")
    parser.add_argument("--volfrac", type=float, default=0.4, help="Target volume fraction")
    parser.add_argument("--buckle-weight", type=float, default=0.45, help="Buckling sensitivity blending weight")
    parser.add_argument("--output", type=str, default="images/buckling_comparison_sample_000.png", help="Output comparison path")
    
    args = parser.parse_args()
    run_comparison(
        sample_idx=args.sample_index,
        material_name=args.material,
        force_N=args.force,
        length_mm=args.length,
        height_mm=args.height,
        thickness_mm=args.thickness,
        nelx=args.nelx,
        nely=args.nely,
        volfrac=args.volfrac,
        buckle_weight=args.buckle_weight,
        output_path=args.output
    )
