"""
generate_iteration_visualizations.py - Generate 3-Panel Visualizations for Every 10 Iterations
Produces the exact format as sample_000.png for Iterations 10, 20, 30, 40, 50, 60, 70, 80.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import argparse
from PIL import Image

import dataset_generator as dg
from visualize_data import extract_discrete_loads, draw_domain_and_bcs, draw_loads


def generate_3panel_iterations(sample_id=0, output_dir="images", max_iter=80, interval=10):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"============================================================")
    print(f" Generating 3-Panel Iteration Visualizations for Sample {sample_id:03d}")
    print(f" Snapshots: Every {interval} iterations up to {max_iter}")
    print(f"============================================================")
    
    # Run SIMP with iteration recording
    res = dg.sample_problem(
        sample_id=sample_id,
        nelx=120,
        nely=60,
        penal=3.0,
        rmin=1.5,
        max_iter=max_iter,
        record_history=True,
        history_interval=interval
    )
    
    if res is None:
        print("Error: SIMP run failed.")
        return []
        
    input_tensor, final_density, history = res
    nelx = 120
    nely = 60
    margin_x = 22
    margin_y = 20
    
    ch_domain, ch_bc, ch_fx, ch_fy, ch_vf = input_tensor
    target_vf = float(ch_vf[0, 0])
    loads = extract_discrete_loads(ch_fx, ch_fy)
    
    generated_files = []
    gif_frames = []
    
    # Filter for iterations 10, 20, 30, 40, 50, 60, 70, 80
    target_iterations = [h for h in history if h['iteration'] > 0 and (h['iteration'] % interval == 0 or h['iteration'] == max_iter)]
    
    for snap in target_iterations:
        it = snap['iteration']
        dens = snap['density']
        c_val = snap['compliance']
        ch_val = snap['change']
        actual_vf = float(np.mean(dens))
        vf_error = abs(actual_vf - target_vf)
        
        fig, axes = plt.subplots(1, 3, figsize=(22, 6.5),
                                 gridspec_kw={'width_ratios': [1.1, 0.4, 1.1]})
        
        # ============================================================
        # Panel 1: Problem Setup (Supports + Loads on domain)
        # ============================================================
        ax0 = axes[0]
        draw_domain_and_bcs(ax0, ch_bc, nelx, nely, is_optimized=False)
        draw_loads(ax0, loads, arrow_len=18)
        
        ax0.set_xlim(-margin_x, nelx + margin_x)
        ax0.set_ylim(nely + margin_y, -margin_y)
        ax0.set_aspect('equal')
        ax0.set_title(f"Input Conditions (Problem Specification)\nTarget Volume Fraction Vf = {target_vf:.3f}",
                      fontsize=12, fontweight='bold', pad=10)
        ax0.set_xlabel("X (elements)", fontsize=10)
        ax0.set_ylabel("Y (elements)", fontsize=10)
        
        leg_items = [
            mpatches.Patch(facecolor='#D32F2F', edgecolor='#B71C1C', label='Boundary Support (Fixed/Pin)'),
            mpatches.Patch(facecolor='#1565C0', edgecolor='#0D47A1', label='Applied Load Vector'),
            mpatches.Patch(facecolor='#FF6F00', edgecolor='black', label='Application Node')
        ]
        ax0.legend(handles=leg_items, loc='upper right', fontsize=8.5, framealpha=0.9)
        
        # ============================================================
        # Panel 2: Center Schematic (SIMP Process + Current Metrics)
        # ============================================================
        ax1 = axes[1]
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.annotate('', xy=(8.5, 5), xytext=(1.5, 5),
                     arrowprops=dict(arrowstyle='->', color='#1A237E', lw=4,
                                     mutation_scale=28))
        ax1.text(5, 7.2, "SIMP\nOptimization", ha='center', va='center',
                 fontsize=14, fontweight='bold', color='#1A237E')
                 
        comp_str = f"{c_val:.2f}" if c_val is not None else "N/A"
        ax1.text(5, 2.7,
                 f"• Objective: Min Compliance\n"
                 f"• Penalization: p = 3.0\n"
                 f"• Sensitivity Filter: R = 1.5\n"
                 f"• Iteration: {it:02d} / {max_iter}\n"
                 f"• Compliance: {comp_str}\n"
                 f"• Max Δρ: {ch_val:.4f}\n"
                 f"• Self-Weight: Excluded",
                 ha='center', va='center', fontsize=9.5, color='#37474F',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECEFF1', edgecolor='#B0BEC5', alpha=0.9))
        ax1.axis('off')
        
        # ============================================================
        # Panel 3: Evolving Topology with Overlaid Boundary Conditions
        # ============================================================
        ax2 = axes[2]
        draw_domain_and_bcs(ax2, ch_bc, nelx, nely, is_optimized=True, target=dens)
        draw_loads(ax2, loads, arrow_len=18)
        
        ax2.set_xlim(-margin_x, nelx + margin_x)
        ax2.set_ylim(nely + margin_y, -margin_y)
        ax2.set_aspect('equal')
        
        panel3_title = f"Optimized Structure (Iteration {it:02d} / {max_iter})\nActual Vf = {actual_vf:.3f} (Error: {vf_error:.4f})"
        ax2.set_title(panel3_title, fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel("X (elements)", fontsize=10)
        ax2.set_ylabel("Y (elements)", fontsize=10)
        
        fig.suptitle(f"Sample {sample_id:03d} — SIMP Optimization Evolution (Iteration {it:02d} / {max_iter})",
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        out_filename = f"sample_{sample_id:03d}_iter_{it:03d}.png"
        out_path = os.path.join(output_dir, out_filename)
        plt.savefig(out_path, dpi=160, bbox_inches='tight')
        print(f" [SAVED] Iteration {it:02d} -> {out_path}")
        generated_files.append(out_path)
        
        # Render frame for animated GIF
        fig.canvas.draw()
        frame = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        gif_frames.append(frame.convert("RGB"))
        plt.close(fig)
        
    # Save combined animated GIF
    if gif_frames:
        # Pause longer on the final converged iteration
        extended_frames = gif_frames + [gif_frames[-1]] * 4
        gif_path = os.path.join(output_dir, f"sample_{sample_id:03d}_evolution_3panel.gif")
        extended_frames[0].save(
            gif_path,
            save_all=True,
            append_images=extended_frames[1:],
            duration=700,
            loop=0
        )
        print(f" [SAVED] 3-Panel Animated GIF -> {gif_path}")
        
    return generated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3-Panel Iteration Visualizations")
    parser.add_argument("--sample_id", type=int, default=0, help="Sample ID to generate")
    parser.add_argument("--output_dir", type=str, default="images", help="Output directory")
    parser.add_argument("--interval", type=int, default=10, help="Iteration interval")
    parser.add_argument("--max_iter", type=int, default=80, help="Max iterations")
    args = parser.parse_args()
    
    generate_3panel_iterations(
        sample_id=args.sample_id,
        output_dir=args.output_dir,
        max_iter=args.max_iter,
        interval=args.interval
    )
