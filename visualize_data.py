"""
visualize_data.py - Professional Engineering Visualization of SIMP Dataset
Clearly displays:
1. Boundary conditions / supports (with standard engineering symbols & labels)
2. Applied load vectors (unclipped, with arrows, application points, magnitudes, angles)
3. Target volume fraction
4. SIMP-optimized topology with all BCs and loads overlaid
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os
from scipy.ndimage import label, center_of_mass


def extract_discrete_loads(ch_fx, ch_fy, threshold=0.01):
    """
    Groups adjacent interpolated element load cells to find the exact
    discrete load application points and vectors.
    """
    mag = np.sqrt(ch_fx**2 + ch_fy**2)
    mask = mag > threshold
    if not np.any(mask):
        return []

    labeled, num_features = label(mask)
    loads = []
    for f in range(1, num_features + 1):
        idx = np.where(labeled == f)
        # Weight centroid by magnitude
        weights = mag[idx]
        total_w = np.sum(weights)
        c_y = np.sum(idx[0] * weights) / total_w
        c_x = np.sum(idx[1] * weights) / total_w

        # Take peak or average force vector in this cluster
        peak_i = np.argmax(weights)
        r_p, c_p = idx[0][peak_i], idx[1][peak_i]
        fx = ch_fx[r_p, c_p]
        fy = ch_fy[r_p, c_p]
        f_mag = np.sqrt(fx**2 + fy**2)
        loads.append((c_x + 0.5, c_y + 0.5, fx, fy, f_mag))
    return loads


def draw_domain_and_bcs(ax, ch_bc, nelx=120, nely=60, is_optimized=False, target=None):
    """Draws the domain background, borders, and standard engineering support symbols."""
    # Domain background
    if is_optimized and target is not None:
        struct_rgb = np.ones((nely, nelx, 3), dtype=np.float32)
        for c in range(3):
            struct_rgb[:, :, c] = 1.0 - 0.92 * target
        ax.imshow(struct_rgb, extent=[0, nelx, nely, 0], zorder=1)
    else:
        domain_bg = np.ones((nely, nelx, 3), dtype=np.float32) * 0.96
        ax.imshow(domain_bg, extent=[0, nelx, nely, 0], zorder=1)

    # Clean domain border
    domain_border = mpatches.Rectangle((0, 0), nelx, nely, fill=False,
                                       edgecolor='#444444', lw=2.0, zorder=5)
    ax.add_patch(domain_border)

    # Detect BC type from ch_bc
    bc_rows, bc_cols = np.where(ch_bc > 0.05)
    left_clamped = np.sum(bc_cols <= 1) > 20
    right_clamped = np.sum(bc_cols >= nelx - 2) > 20
    bottom_corners = (np.any((bc_rows >= nely - 3) & (bc_cols <= 3)) and
                      np.any((bc_rows >= nely - 3) & (bc_cols >= nelx - 4)))

    support_patches = []

    # 1. Left Clamped Wall
    if left_clamped:
        # Thick red wall
        wall = mpatches.Rectangle((-3.5, 0), 3.5, nely, facecolor='#D32F2F',
                                  edgecolor='#B71C1C', lw=1.5, zorder=6)
        ax.add_patch(wall)
        # Engineering hatch lines
        for yy in range(2, nely, 5):
            ax.plot([-3.5, -6.5], [yy, yy + 3], color='#B71C1C', lw=1.5, zorder=6)
        ax.text(-8.0, nely / 2, 'CLAMPED\nWALL', color='#B71C1C',
                fontsize=9, fontweight='bold', ha='right', va='center', rotation=90)

    # 2. Right Clamped Wall
    if right_clamped:
        wall = mpatches.Rectangle((nelx, 0), 3.5, nely, facecolor='#D32F2F',
                                  edgecolor='#B71C1C', lw=1.5, zorder=6)
        ax.add_patch(wall)
        for yy in range(2, nely, 5):
            ax.plot([nelx + 3.5, nelx + 6.5], [yy, yy + 3], color='#B71C1C', lw=1.5, zorder=6)
        ax.text(nelx + 8.0, nely / 2, 'CLAMPED\nWALL', color='#B71C1C',
                fontsize=9, fontweight='bold', ha='left', va='center', rotation=-90)

    # 3. Bottom Corner Supports (Bridge / MBB / Truss)
    if not left_clamped and not right_clamped and len(bc_rows) > 0:
        # Pin at bottom-left
        if np.any((bc_rows >= nely - 4) & (bc_cols <= 5)):
            # Pin triangle
            tri = mpatches.Polygon([[0, nely], [-4, nely + 6], [4, nely + 6]],
                                   closed=True, facecolor='#D32F2F', edgecolor='#B71C1C', lw=1.5, zorder=7)
            ax.add_patch(tri)
            # Ground line & hatch
            ax.plot([-7, 7], [nely + 6, nely + 6], color='#B71C1C', lw=2, zorder=7)
            for hx in range(-6, 7, 3):
                ax.plot([hx, hx - 2], [nely + 6, nely + 9], color='#B71C1C', lw=1.2, zorder=7)
            ax.text(0, nely + 13, 'PIN', color='#B71C1C', fontsize=9,
                    fontweight='bold', ha='center', va='top')

        # Roller/Pin at bottom-right
        if np.any((bc_rows >= nely - 4) & (bc_cols >= nelx - 5)):
            tri = mpatches.Polygon([[nelx, nely], [nelx - 4, nely + 6], [nelx + 4, nely + 6]],
                                   closed=True, facecolor='#D32F2F', edgecolor='#B71C1C', lw=1.5, zorder=7)
            ax.add_patch(tri)
            # Roller wheels
            c1 = mpatches.Circle((nelx - 2, nely + 7.5), 1.2, facecolor='white', edgecolor='#B71C1C', lw=1.2, zorder=7)
            c2 = mpatches.Circle((nelx + 2, nely + 7.5), 1.2, facecolor='white', edgecolor='#B71C1C', lw=1.2, zorder=7)
            ax.add_patch(c1); ax.add_patch(c2)
            # Ground line & hatch
            ax.plot([nelx - 7, nelx + 7], [nely + 9, nely + 9], color='#B71C1C', lw=2, zorder=7)
            for hx in range(int(nelx - 6), int(nelx + 7), 3):
                ax.plot([hx, hx - 2], [nely + 9, nely + 12], color='#B71C1C', lw=1.2, zorder=7)
            ax.text(nelx, nely + 15, 'ROLLER / PIN', color='#B71C1C', fontsize=9,
                    fontweight='bold', ha='center', va='top')


def draw_loads(ax, loads, arrow_len=18):
    """Draws big, unambiguous load arrows with labels."""
    for i, (lx, ly, fx, fy, mag) in enumerate(loads):
        # Mark application node with a bright orange dot
        ax.plot(lx, ly, marker='o', markersize=8, color='#FF6F00',
                markeredgecolor='black', markeredgewidth=1.2, zorder=12)

        # Vector normalized
        u_x = fx / mag
        u_y = fy / mag

        # Arrow points from outside (lx - u_x*arrow_len, ly - u_y*arrow_len) towards (lx, ly)
        # OR from (lx, ly) along (u_x, u_y).
        # In engineering, an arrow pointing *towards* the contact node is universally recognized.
        # Let's draw the arrow starting at (lx, ly) and extending along (u_x, u_y):
        start_x = lx
        start_y = ly
        end_x = lx + u_x * arrow_len
        end_y = ly + u_y * arrow_len

        ax.annotate(
            '', xy=(end_x, end_y), xytext=(start_x, start_y),
            arrowprops=dict(
                arrowstyle='->,head_width=0.6,head_length=0.8',
                color='#1565C0',
                lw=3.5
            ),
            zorder=15
        )

        # Label with force magnitude and angle
        deg = np.degrees(np.arctan2(fy, fx))
        label_text = f"Load #{i+1}\n|F|={mag:.2f}\nθ={deg:.0f}°"
        
        # Position label offset from arrow end
        lbl_x = end_x + u_x * 5
        lbl_y = end_y + u_y * 5
        ax.text(
            lbl_x, lbl_y, label_text,
            fontsize=8.5, fontweight='bold', color='#0D47A1',
            ha='center', va='center', zorder=16,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1565C0', alpha=0.95, lw=1.2)
        )


def visualize_dataset(npz_path='topo_dataset.npz', output_dir='images', max_samples=10):
    if not os.path.exists(npz_path):
        print(f"Error: Dataset file '{npz_path}' not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = np.load(npz_path)
    inputs = data['inputs']
    targets = data['targets']

    total_samples = inputs.shape[0]
    num_to_plot = min(total_samples, max_samples)
    print(f"Visualizing {num_to_plot} of {total_samples} samples from '{npz_path}'...")

    nely, nelx = 60, 120
    margin_x = 22
    margin_y = 20

    for i in range(num_to_plot):
        target = targets[i]
        ch_bc = inputs[i, 1]
        ch_fx = inputs[i, 2]
        ch_fy = inputs[i, 3]
        ch_vf = inputs[i, 4]

        target_vf = ch_vf[0, 0]
        actual_vf = np.mean(target)
        loads = extract_discrete_loads(ch_fx, ch_fy)

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

        # Custom clear legend
        leg_items = [
            mpatches.Patch(facecolor='#D32F2F', edgecolor='#B71C1C', label='Boundary Support (Fixed/Pin)'),
            mpatches.Patch(facecolor='#1565C0', edgecolor='#0D47A1', label='Applied Load Vector'),
            mpatches.Patch(facecolor='#FF6F00', edgecolor='black', label='Application Node')
        ]
        ax0.legend(handles=leg_items, loc='upper right', fontsize=8.5, framealpha=0.9)

        # ============================================================
        # Panel 2: Center Schematic (SIMP Process)
        # ============================================================
        ax1 = axes[1]
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.annotate('', xy=(8.5, 5), xytext=(1.5, 5),
                     arrowprops=dict(arrowstyle='->', color='#1A237E', lw=4,
                                     mutation_scale=28))
        ax1.text(5, 7.2, "SIMP\nOptimization", ha='center', va='center',
                 fontsize=14, fontweight='bold', color='#1A237E')
        ax1.text(5, 2.8,
                 f"• Objective: Min Compliance\n"
                 f"• Penalization: p = 3.0\n"
                 f"• Sensitivity Filter: R = 1.5\n"
                 f"• Max Iterations: 80\n"
                 f"• Self-Weight: Excluded",
                 ha='center', va='center', fontsize=9.5, color='#37474F',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECEFF1', edgecolor='#B0BEC5', alpha=0.9))
        ax1.axis('off')

        # ============================================================
        # Panel 3: SIMP-Optimized Topology with Overlaid Boundary Conditions
        # ============================================================
        ax2 = axes[2]
        draw_domain_and_bcs(ax2, ch_bc, nelx, nely, is_optimized=True, target=target)
        draw_loads(ax2, loads, arrow_len=18)

        ax2.set_xlim(-margin_x, nelx + margin_x)
        ax2.set_ylim(nely + margin_y, -margin_y)
        ax2.set_aspect('equal')
        ax2.set_title(f"Optimized Structure (Ground Truth Target)\nActual Vf = {actual_vf:.3f} (Error: {abs(actual_vf - target_vf):.4f})",
                      fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel("X (elements)", fontsize=10)
        ax2.set_ylabel("Y (elements)", fontsize=10)

        fig.suptitle(f"Sample {i:03d} — Training Pair Visualization (Input Specification → Optimal Topology)",
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        out_file = os.path.join(output_dir, f'sample_{i:03d}.png')
        plt.savefig(out_file, dpi=160, bbox_inches='tight')
        plt.close()

    print(f"Generated {num_to_plot} images in '{os.path.abspath(output_dir)}'.")


def main():
    parser = argparse.ArgumentParser(description="Visualize SIMP Topology Optimization Dataset")
    parser.add_argument("--dataset", type=str, default="topo_dataset.npz", help="Path to .npz dataset file")
    parser.add_argument("--output_dir", type=str, default="images", help="Output directory for PNG visualizations")
    parser.add_argument("--max_samples", type=int, default=10, help="Maximum number of samples to visualize")
    args = parser.parse_args()

    visualize_dataset(npz_path=args.dataset, output_dir=args.output_dir, max_samples=args.max_samples)


if __name__ == '__main__':
    main()
