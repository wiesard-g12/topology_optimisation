"""
visualize_data.py - Professional Engineering Visualization of SIMP Topology Optimization
Unified 2D Continuum SIMP Pipeline (With & Without Buckling)

Features:
1. Standardized discrete support symbols (Pin, Roller, Clamped Wall) with standard engineering hatch lines.
2. Unambiguous load vector arrows (application node, magnitude, angle, Cartesian components).
3. Initial Structure rendering before optimization (draw_initial_structure_and_bcs) displaying
   uniform density block rho^(0) = Vf with domain boundary and boundary conditions.
4. 3-Panel Before & After Sample Visualizer (render_before_after_sample):
   - Panel 1: BEFORE Optimization (Initial Uniform Block with supports & applied forces)
   - Panel 2: Carried Variables Scorecard (Domain, Mode, Target Vf, Reduction %, Supports, Loads, Compliance, Buckling)
   - Panel 3: AFTER Optimization (Converged SIMP Topology with overlaid supports & applied forces)
5. 4-Panel Head-to-Head Buckling Visualizer (render_buckling_head_to_head):
   - Panel 1: BEFORE (Initial Block & Problem Specification)
   - Panel 2: AFTER Pure Compliance SIMP (Without Buckling)
   - Panel 3: AFTER Buckling-Aware SIMP (With Buckling + % gain in lambda_1)
   - Panel 4: Material Redistribution Delta rho = rho_buckle - rho_pure (coolwarm heatmap)
6. Backward compatibility with extract_discrete_loads, draw_domain_and_bcs, draw_loads, and visualize_dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os
from scipy.ndimage import label


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
        weights = mag[idx]
        total_w = np.sum(weights)
        c_y = np.sum(idx[0] * weights) / total_w
        c_x = np.sum(idx[1] * weights) / total_w

        peak_i = np.argmax(weights)
        r_p, c_p = idx[0][peak_i], idx[1][peak_i]
        fx = float(ch_fx[r_p, c_p])
        fy = float(ch_fy[r_p, c_p])
        f_mag = float(np.sqrt(fx**2 + fy**2))
        if f_mag < 0.55:
            fx *= 4.0
            fy *= 4.0
            f_mag *= 4.0
        loads.append((c_x + 0.5, c_y + 0.5, fx, fy, f_mag))
    return loads


def extract_supports_from_bcs(ch_bc_x, ch_bc_y, nelx=120, nely=60, threshold=0.04):
    """
    Extracts support locations and types (Pin, X-Roller, Y-Roller, Clamped Wall)
    from continuous raster boundary condition channels.
    """
    if ch_bc_x is None and ch_bc_y is None:
        return []
    if ch_bc_x is None:
        ch_bc_x = np.zeros_like(ch_bc_y)
    if ch_bc_y is None:
        ch_bc_y = np.zeros_like(ch_bc_x)

    ch_combined = np.maximum(ch_bc_x, ch_bc_y)
    mask = ch_combined > threshold
    if not np.any(mask):
        return []

    # Check for wall clamping along left edge
    left_mask = mask[:, :2]
    if np.sum(left_mask) > 20:
        has_y = np.any(ch_bc_y[:, :2] > threshold)
        stype = 'Clamped Wall' if has_y else 'Roller (X fixed)'
        mask_rest = mask.copy()
        mask_rest[:, :2] = False
        supports = [{
            'node_id': 0,
            'x': 0,
            'y': nely // 2,
            'type': stype,
            'is_wall': True
        }]
    else:
        mask_rest = mask
        supports = []

    if np.any(mask_rest):
        labeled, num_features = label(mask_rest)
        for f in range(1, num_features + 1):
            idx = np.where(labeled == f)
            weights = ch_combined[idx]
            total_w = np.sum(weights)
            c_y = np.sum(idx[0] * weights) / total_w
            c_x = np.sum(idx[1] * weights) / total_w

            snap_x = 0 if c_x < 3 else (nelx if c_x > nelx - 3 else int(round(c_x)))
            snap_y = 0 if c_y < 3 else (nely if c_y > nely - 3 else int(round(c_y)))

            max_x = float(np.max(ch_bc_x[idx]))
            max_y = float(np.max(ch_bc_y[idx]))

            has_x = max_x > threshold
            has_y = max_y > threshold

            if has_x and has_y:
                stype = 'Pin (X,Y fixed)'
            elif has_x:
                stype = 'Roller (X fixed)'
            elif has_y:
                stype = 'Roller (Y fixed)'
            else:
                stype = 'Support'

            supports.append({
                'node_id': snap_x * (nely + 1) + snap_y,
                'x': snap_x,
                'y': snap_y,
                'type': stype,
                'is_wall': False
            })

    return supports


def draw_supports(ax, ch_bc=None, nelx=120, nely=60, support_coords=None, ch_bc_x=None, ch_bc_y=None):
    """
    Renders standardized engineering support symbols:
    - Clamped walls with 45-deg ground hatches
    - Pinned supports (triangle with fixed ground line & hatches)
    - Roller supports (triangle with dual roller bearings & ground line)
    Handles discrete node coordinates list (support_coords) and raster BC masks (ch_bc_x, ch_bc_y, ch_bc).
    """
    if support_coords is None or len(support_coords) == 0:
        if ch_bc_x is not None or ch_bc_y is not None:
            support_coords = extract_supports_from_bcs(ch_bc_x, ch_bc_y, nelx, nely)
        elif ch_bc is not None:
            support_coords = extract_supports_from_bcs(ch_bc, ch_bc, nelx, nely)

    if support_coords is not None and len(support_coords) > 0:
        for s in support_coords:
            x = float(s['x'])
            y = float(s['y'])
            stype = str(s.get('type', 'Pin')).lower()
            is_roller = ('roller' in stype)

            if s.get('is_wall', False):
                if x <= 1.0:
                    wall = mpatches.Rectangle((-3.5, 0), 3.5, nely, facecolor='#D32F2F',
                                              edgecolor='#B71C1C', lw=1.5, zorder=6)
                    ax.add_patch(wall)
                    if is_roller:
                        for yy in range(2, nely, 6):
                            c = mpatches.Circle((-5.0, yy + 2), 1.2, facecolor='white', edgecolor='#B71C1C', lw=1.2, zorder=7)
                            ax.add_patch(c)
                            ax.plot([-6.5, -6.5], [yy, yy + 4], color='#B71C1C', lw=1.5, zorder=7)
                        ax.text(-9.5, nely / 2, 'SYMMETRY\nROLLER', color='#B71C1C',
                                fontsize=8.5, fontweight='bold', ha='right', va='center', rotation=90)
                    else:
                        for yy in range(2, nely, 5):
                            ax.plot([-3.5, -6.5], [yy, yy + 3], color='#B71C1C', lw=1.5, zorder=6)
                        ax.text(-8.0, nely / 2, 'CLAMPED\nWALL', color='#B71C1C',
                                fontsize=9, fontweight='bold', ha='right', va='center', rotation=90)
                continue

            # Determine support orientation based on domain boundary
            if x <= 1.0:  # Left edge
                if is_roller:
                    lbl = 'X-ROLLER' if 'x fixed' in stype else 'ROLLER'
                    tri = mpatches.Polygon([[0, y], [-4, y - 3.5], [-4, y + 3.5]],
                                           closed=True, facecolor='#D32F2F', edgecolor='#B71C1C', lw=1.5, zorder=7)
                    ax.add_patch(tri)
                    c1 = mpatches.Circle((-5.5, y - 2), 1.1, facecolor='white', edgecolor='#B71C1C', lw=1.2, zorder=7)
                    c2 = mpatches.Circle((-5.5, y + 2), 1.1, facecolor='white', edgecolor='#B71C1C', lw=1.2, zorder=7)
                    ax.add_patch(c1); ax.add_patch(c2)
                    ax.plot([-7, -7], [y - 5.5, y + 5.5], color='#B71C1C', lw=2, zorder=7)
                    for dy in [-3.5, 0, 3.5]:
                        ax.plot([-7, -10], [y + dy, y + dy + 2.5], color='#B71C1C', lw=1.2, zorder=7)
                    ax.text(-12, y, lbl, color='#B71C1C', fontsize=8, fontweight='bold', ha='right', va='center')
                else:
                    tri = mpatches.Polygon([[0, y], [-5, y - 4], [-5, y + 4]],
                                           closed=True, facecolor='#D32F2F', edgecolor='#B71C1C', lw=1.5, zorder=7)
                    ax.add_patch(tri)
                    ax.plot([-5, -5], [y - 6, y + 6], color='#B71C1C', lw=2, zorder=7)
                    for dy in [-4, 0, 4]:
                        ax.plot([-5, -8], [y + dy, y + dy + 2.5], color='#B71C1C', lw=1.2, zorder=7)
                    ax.text(-10, y, 'PIN', color='#B71C1C', fontsize=8, fontweight='bold', ha='right', va='center')

            elif x >= nelx - 1.0:  # Right edge
                if is_roller:
                    lbl = 'Y-ROLLER' if 'y fixed' in stype else ('X-ROLLER' if 'x fixed' in stype else 'ROLLER')
                    tri = mpatches.Polygon([[nelx, y], [nelx + 4, y - 3.5], [nelx + 4, y + 3.5]],
                                           closed=True, facecolor='#D32F2F', edgecolor='#B71C1C', lw=1.5, zorder=7)
                    ax.add_patch(tri)
                    c1 = mpatches.Circle((nelx + 5.5, y - 2), 1.1, facecolor='white', edgecolor='#B71C1C', lw=1.2, zorder=7)
                    c2 = mpatches.Circle((nelx + 5.5, y + 2), 1.1, facecolor='white', edgecolor='#B71C1C', lw=1.2, zorder=7)
                    ax.add_patch(c1); ax.add_patch(c2)
                    ax.plot([nelx + 7, nelx + 7], [y - 5.5, y + 5.5], color='#B71C1C', lw=2, zorder=7)
                    for dy in [-3.5, 0, 3.5]:
                        ax.plot([nelx + 7, nelx + 10], [y + dy, y + dy + 2.5], color='#B71C1C', lw=1.2, zorder=7)
                    ax.text(nelx + 12, y, lbl, color='#B71C1C', fontsize=8, fontweight='bold', ha='left', va='center')
                else:
                    tri = mpatches.Polygon([[nelx, y], [nelx + 5, y - 4], [nelx + 5, y + 4]],
                                           closed=True, facecolor='#D32F2F', edgecolor='#B71C1C', lw=1.5, zorder=7)
                    ax.add_patch(tri)
                    ax.plot([nelx + 5, nelx + 5], [y - 6, y + 6], color='#B71C1C', lw=2, zorder=7)
                    for dy in [-4, 0, 4]:
                        ax.plot([nelx + 5, nelx + 8], [y + dy, y + dy + 2.5], color='#B71C1C', lw=1.2, zorder=7)
                    ax.text(nelx + 10, y, 'PIN', color='#B71C1C', fontsize=8, fontweight='bold', ha='left', va='center')

            elif y >= nely - 1.0:  # Bottom edge
                if is_roller:
                    lbl = 'Y-ROLLER' if 'y fixed' in stype else 'ROLLER'
                    tri = mpatches.Polygon([[x, nely], [x - 4, nely + 5], [x + 4, nely + 5]],
                                           closed=True, facecolor='#D32F2F', edgecolor='#B71C1C', lw=1.5, zorder=7)
                    ax.add_patch(tri)
                    c1 = mpatches.Circle((x - 2, nely + 6.5), 1.2, facecolor='white', edgecolor='#B71C1C', lw=1.2, zorder=7)
                    c2 = mpatches.Circle((x + 2, nely + 6.5), 1.2, facecolor='white', edgecolor='#B71C1C', lw=1.2, zorder=7)
                    ax.add_patch(c1); ax.add_patch(c2)
                    ax.plot([x - 6.5, x + 6.5], [nely + 8, nely + 8], color='#B71C1C', lw=2, zorder=7)
                    for hx in np.linspace(x - 5, x + 5, 4):
                        ax.plot([hx, hx - 2], [nely + 8, nely + 11], color='#B71C1C', lw=1.2, zorder=7)
                    ax.text(x, nely + 14, lbl, color='#B71C1C', fontsize=8, fontweight='bold', ha='center', va='top')
                else:
                    tri = mpatches.Polygon([[x, nely], [x - 4, nely + 6], [x + 4, nely + 6]],
                                           closed=True, facecolor='#D32F2F', edgecolor='#B71C1C', lw=1.5, zorder=7)
                    ax.add_patch(tri)
                    ax.plot([x - 6.5, x + 6.5], [nely + 6, nely + 6], color='#B71C1C', lw=2, zorder=7)
                    for hx in np.linspace(x - 5, x + 5, 4):
                        ax.plot([hx, hx - 2], [nely + 6, nely + 9], color='#B71C1C', lw=1.2, zorder=7)
                    ax.text(x, nely + 12, 'PIN', color='#B71C1C', fontsize=8, fontweight='bold', ha='center', va='top')

            elif y <= 1.0:  # Top edge
                if is_roller:
                    lbl = 'Y-ROLLER' if 'y fixed' in stype else 'ROLLER'
                    tri = mpatches.Polygon([[x, 0], [x - 4, -5], [x + 4, -5]],
                                           closed=True, facecolor='#D32F2F', edgecolor='#B71C1C', lw=1.5, zorder=7)
                    ax.add_patch(tri)
                    c1 = mpatches.Circle((x - 2, -6.5), 1.2, facecolor='white', edgecolor='#B71C1C', lw=1.2, zorder=7)
                    c2 = mpatches.Circle((x + 2, -6.5), 1.2, facecolor='white', edgecolor='#B71C1C', lw=1.2, zorder=7)
                    ax.add_patch(c1); ax.add_patch(c2)
                    ax.plot([x - 6.5, x + 6.5], [-8, -8], color='#B71C1C', lw=2, zorder=7)
                    for hx in np.linspace(x - 5, x + 5, 4):
                        ax.plot([hx, hx - 2], [-8, -11], color='#B71C1C', lw=1.2, zorder=7)
                    ax.text(x, -14, lbl, color='#B71C1C', fontsize=8, fontweight='bold', ha='center', va='bottom')
                else:
                    tri = mpatches.Polygon([[x, 0], [x - 4, -6], [x + 4, -6]],
                                           closed=True, facecolor='#D32F2F', edgecolor='#B71C1C', lw=1.5, zorder=7)
                    ax.add_patch(tri)
                    ax.plot([x - 6.5, x + 6.5], [-6, -6], color='#B71C1C', lw=2, zorder=7)
                    for hx in np.linspace(x - 5, x + 5, 4):
                        ax.plot([hx, hx - 2], [-6, -9], color='#B71C1C', lw=1.2, zorder=7)
                    ax.text(x, -12, 'PIN', color='#B71C1C', fontsize=8, fontweight='bold', ha='center', va='bottom')
            else:
                # Internal support
                ax.plot(x, y, marker='s', markersize=9, color='#D32F2F', markeredgecolor='black', lw=1.5, zorder=7)
                ax.text(x, y - 4, 'SUPPORT', color='#B71C1C', fontsize=8, fontweight='bold', ha='center', va='bottom')
        return

    # Fallback to raster ch_bc detection
    if ch_bc is not None:
        bc_rows, bc_cols = np.where(ch_bc > 0.05)
        left_clamped = np.sum(bc_cols <= 1) > 20
        right_clamped = np.sum(bc_cols >= nelx - 2) > 20

        if left_clamped:
            wall = mpatches.Rectangle((-3.5, 0), 3.5, nely, facecolor='#D32F2F',
                                      edgecolor='#B71C1C', lw=1.5, zorder=6)
            ax.add_patch(wall)
            for yy in range(2, nely, 5):
                ax.plot([-3.5, -6.5], [yy, yy + 3], color='#B71C1C', lw=1.5, zorder=6)
            ax.text(-8.0, nely / 2, 'CLAMPED\nWALL', color='#B71C1C',
                    fontsize=9, fontweight='bold', ha='right', va='center', rotation=90)

        if right_clamped:
            wall = mpatches.Rectangle((nelx, 0), 3.5, nely, facecolor='#D32F2F',
                                      edgecolor='#B71C1C', lw=1.5, zorder=6)
            ax.add_patch(wall)
            for yy in range(2, nely, 5):
                ax.plot([nelx + 3.5, nelx + 6.5], [yy, yy + 3], color='#B71C1C', lw=1.5, zorder=6)
            ax.text(nelx + 8.0, nely / 2, 'CLAMPED\nWALL', color='#B71C1C',
                    fontsize=9, fontweight='bold', ha='left', va='center', rotation=-90)

        if not left_clamped and not right_clamped and len(bc_rows) > 0:
            if np.any((bc_rows >= nely - 4) & (bc_cols <= 5)):
                tri = mpatches.Polygon([[0, nely], [-4, nely + 6], [4, nely + 6]],
                                       closed=True, facecolor='#D32F2F', edgecolor='#B71C1C', lw=1.5, zorder=7)
                ax.add_patch(tri)
                ax.plot([-7, 7], [nely + 6, nely + 6], color='#B71C1C', lw=2, zorder=7)
                for hx in range(-6, 7, 3):
                    ax.plot([hx, hx - 2], [nely + 6, nely + 9], color='#B71C1C', lw=1.2, zorder=7)
                ax.text(0, nely + 13, 'PIN', color='#B71C1C', fontsize=8.5,
                        fontweight='bold', ha='center', va='top')

            if np.any((bc_rows >= nely - 4) & (bc_cols >= nelx - 5)):
                tri = mpatches.Polygon([[nelx, nely], [nelx - 4, nely + 5], [nelx + 4, nely + 5]],
                                       closed=True, facecolor='#D32F2F', edgecolor='#B71C1C', lw=1.5, zorder=7)
                ax.add_patch(tri)
                c1 = mpatches.Circle((nelx - 2, nely + 6.5), 1.2, facecolor='white', edgecolor='#B71C1C', lw=1.2, zorder=7)
                c2 = mpatches.Circle((nelx + 2, nely + 6.5), 1.2, facecolor='white', edgecolor='#B71C1C', lw=1.2, zorder=7)
                ax.add_patch(c1); ax.add_patch(c2)
                ax.plot([nelx - 7, nelx + 7], [nely + 8, nely + 8], color='#B71C1C', lw=2, zorder=7)
                for hx in range(int(nelx - 6), int(nelx + 7), 3):
                    ax.plot([hx, hx - 2], [nely + 8, nely + 11], color='#B71C1C', lw=1.2, zorder=7)
                ax.text(nelx, nely + 14, 'ROLLER / PIN', color='#B71C1C', fontsize=8.5,
                        fontweight='bold', ha='center', va='top')


def draw_domain_and_bcs(ax, ch_bc, nelx=120, nely=60, is_optimized=False, target=None, support_coords=None, ch_bc_x=None, ch_bc_y=None):
    """Draws domain background, borders, and standard engineering support symbols."""
    if is_optimized and target is not None:
        struct_rgb = np.ones((nely, nelx, 3), dtype=np.float32)
        for c in range(3):
            struct_rgb[:, :, c] = 1.0 - 0.92 * np.clip(target, 0.0, 1.0)
        ax.imshow(struct_rgb, extent=[0, nelx, nely, 0], zorder=1)
    else:
        domain_bg = np.ones((nely, nelx, 3), dtype=np.float32) * 0.96
        ax.imshow(domain_bg, extent=[0, nelx, nely, 0], zorder=1)

    domain_border = mpatches.Rectangle((0, 0), nelx, nely, fill=False,
                                       edgecolor='#333333', lw=2.0, zorder=5)
    ax.add_patch(domain_border)

    draw_supports(ax, ch_bc, nelx, nely, support_coords=support_coords, ch_bc_x=ch_bc_x, ch_bc_y=ch_bc_y)


def draw_initial_structure_and_bcs(ax, ch_bc, initial_density, nelx=120, nely=60, support_coords=None, ch_bc_x=None, ch_bc_y=None):
    """
    Renders the Initial Structure before optimization:
    A uniform solid continuum block with shading proportional to initial_density = volfrac,
    domain border, and support pins/rollers/walls so the engineer can visually see
    how the initial block looked before carving out material!
    """
    if isinstance(initial_density, np.ndarray):
        init_arr = initial_density
    else:
        init_arr = np.full((nely, nelx), float(initial_density), dtype=np.float32)

    struct_rgb = np.ones((nely, nelx, 3), dtype=np.float32)
    # Shading proportional to density: solid 1.0 is dark gray (~0.12), 0.40 is medium gray (~0.65)
    for c in range(3):
        struct_rgb[:, :, c] = 1.0 - 0.88 * np.clip(init_arr, 0.0, 1.0)
    ax.imshow(struct_rgb, extent=[0, nelx, nely, 0], zorder=1)

    domain_border = mpatches.Rectangle((0, 0), nelx, nely, fill=False,
                                       edgecolor='#333333', lw=2.0, zorder=5)
    ax.add_patch(domain_border)

    draw_supports(ax, ch_bc, nelx, nely, support_coords=support_coords, ch_bc_x=ch_bc_x, ch_bc_y=ch_bc_y)


def draw_loads(ax, loads, arrow_len=18):
    """
    Draws big, unambiguous load arrows with application nodes, vectors, magnitudes, and angles.
    Accepts both list of tuples (lx, ly, fx, fy, mag) and list of dicts from meta['load_details'].
    """
    if not loads:
        return

    for i, item in enumerate(loads):
        if isinstance(item, dict):
            lx, ly = item['x'], item['y']
            fx, fy = item['Fx'], item['Fy']
            mag = item['magnitude']
            deg = item.get('angle_deg', float(np.degrees(np.arctan2(fy, fx))))
        else:
            lx, ly, fx, fy, mag = item
            deg = float(np.degrees(np.arctan2(fy, fx)))

        if mag <= 1e-4:
            continue

        # Application node marker
        ax.plot(lx, ly, marker='o', markersize=8, color='#FF6F00',
                markeredgecolor='black', markeredgewidth=1.2, zorder=12)

        # Normalized vector
        u_x = fx / mag
        u_y = fy / mag

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

        label_text = f"Load #{i+1}\n|F|={mag:.2f}\nθ={deg:.0f}°"
        lbl_x = end_x + u_x * 5.5
        lbl_y = end_y + u_y * 5.5
        ax.text(
            lbl_x, lbl_y, label_text,
            fontsize=8.0, fontweight='bold', color='#0D47A1',
            ha='center', va='center', zorder=16,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#E3F2FD', edgecolor='#1565C0', alpha=0.95, lw=1.2)
        )


def render_before_after_sample(input_tensor, meta, output_path):
    """
    Creates a 3-panel Before & After engineering figure:
      Panel 1: BEFORE Optimization (Initial Uniform Block with supports & applied forces)
      Panel 2: Carried Variables Scorecard (Domain, Mode, Target Vf, Reduction %, Supports, Loads, Compliances, Buckling)
      Panel 3: AFTER Optimization (Converged SIMP Topology with overlaid supports & applied forces)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    nelx = meta.get('nelx', 120)
    nely = meta.get('nely', 60)
    ndof = meta.get('ndof', 2 * (nelx + 1) * (nely + 1))
    case_name = meta.get('case_name', 'Benchmark').upper()
    use_buckling = meta.get('use_buckling', False)
    buckle_weight = meta.get('buckle_weight', 0.45)
    mode_str = f"WITH BUCKLING (wb = {buckle_weight:.2f})" if use_buckling else "WITHOUT BUCKLING (Pure Compliance)"

    if input_tensor.shape[0] == 6:
        ch_domain = input_tensor[0]
        ch_bc_x = input_tensor[1]
        ch_bc_y = input_tensor[2]
        ch_fx = input_tensor[3]
        ch_fy = input_tensor[4]
        ch_vf = input_tensor[5]
        ch_bc = np.maximum(ch_bc_x, ch_bc_y)
        volfrac = meta.get('volfrac', float(ch_vf[0, 0]))
    else:
        ch_domain = input_tensor[0]
        ch_bc = input_tensor[1]
        ch_fx = input_tensor[2]
        ch_fy = input_tensor[3]
        ch_vf = input_tensor[4]
        ch_bc_x = ch_bc
        ch_bc_y = ch_bc
        volfrac = meta.get('volfrac', float(ch_vf[0, 0]))

    vol_red_pct = meta.get('volume_reduction_pct', (1.0 - volfrac) * 100.0)
    actual_vf = meta.get('actual_vf', volfrac)

    initial_c = meta.get('initial_compliance', 0.0)
    final_c = meta.get('final_compliance', 0.0)
    c_red_pct = meta.get('compliance_reduction_pct', 0.0)
    lambda_1 = meta.get('final_lambda_1', 999.0)
    lambda_str = f"{lambda_1:.2f}" if (0.0 < lambda_1 < 900) else "N/A (Pure Comp)"

    supp_coords = meta.get('support_coords', None)
    load_details = meta.get('load_details', None)
    if load_details is not None and len(load_details) > 0:
        loads = load_details
    else:
        loads = extract_discrete_loads(ch_fx, ch_fy)

    initial_density = meta.get('initial_density', volfrac)
    final_density = meta.get('final_density', None)

    fig, axes = plt.subplots(1, 3, figsize=(23, 7.2),
                             gridspec_kw={'width_ratios': [1.15, 0.75, 1.15]})

    margin_x = 24
    margin_y = 22

    # ============================================================
    # Panel 1: BEFORE Optimization (Initial Uniform Block)
    # ============================================================
    ax0 = axes[0]
    draw_initial_structure_and_bcs(ax0, ch_bc, initial_density, nelx, nely, support_coords=supp_coords, ch_bc_x=ch_bc_x, ch_bc_y=ch_bc_y)
    draw_loads(ax0, loads, arrow_len=18)

    ax0.set_xlim(-margin_x, nelx + margin_x)
    ax0.set_ylim(nely + margin_y, -margin_y)
    ax0.set_aspect('equal')
    ax0.set_title(
        f"1. BEFORE Optimization: Initial Uniform Block\n"
        f"Vf = {volfrac:.2f} (Target Reduction = {vol_red_pct:.1f}%) | Initial C₀ = {initial_c:.2f}",
        fontsize=11, fontweight='bold', pad=10
    )
    ax0.set_xlabel("X (elements)", fontsize=9.5)
    ax0.set_ylabel("Y (elements)", fontsize=9.5)

    leg_items_0 = [
        mpatches.Patch(facecolor='#888888', edgecolor='#333333', label=f'Initial Solid Block (ρ⁽⁰⁾ = {volfrac:.2f})'),
        mpatches.Patch(facecolor='#D32F2F', edgecolor='#B71C1C', label='Boundary Support (Pin/Roller)'),
        mpatches.Patch(facecolor='#1565C0', edgecolor='#0D47A1', label='Applied Load Vector'),
        mpatches.Patch(facecolor='#FF6F00', edgecolor='black', label='Application Node')
    ]
    ax0.legend(handles=leg_items_0, loc='upper right', fontsize=8.2, framealpha=0.92)

    # ============================================================
    # Panel 2: Carried Variables Scorecard
    # ============================================================
    ax1 = axes[1]
    ax1.axis('off')

    supp_lines = []
    for s in (supp_coords or []):
        supp_lines.append(f"  • Node {s['node_id']}: (x={s['x']}, y={s['y']}) -> {s['type']}")
    if not supp_lines:
        supp_lines = ["  • Standard boundary constraints"]
    supp_text = "\n".join(supp_lines[:4])

    load_lines = []
    for ld in (load_details or []):
        load_lines.append(f"  • Node {ld['node_id']}: (x={ld['x']}, y={ld['y']}) | |F|={ld['magnitude']:.2f}, θ={ld['angle_deg']:.0f}°\n    Vector: (Fx={ld['Fx']:+.2f}, Fy={ld['Fy']:+.2f})")
    if not load_lines:
        load_lines = ["  • Standard nodal forces"]
    load_text = "\n".join(load_lines[:3])

    scorecard = (
        f"═══════════════════════════════════════════\n"
        f"      CARRIED VARIABLES SCORECARD\n"
        f"═══════════════════════════════════════════\n"
        f"• Benchmark Case:   {case_name}\n"
        f"• Domain Grid:      {nelx} × {nely} elements\n"
        f"• Total DOFs:       {ndof:,} DOFs\n"
        f"• Optimization Mode:\n"
        f"    {mode_str}\n"
        f"───────────────────────────────────────────\n"
        f"MATERIAL & VOLUME REDUCTION TARGETS\n"
        f"• Initial Density:  ρ⁽⁰⁾ = {volfrac:.2f} (Uniform solid block)\n"
        f"• Target Volume Vf: {volfrac:.2f}\n"
        f"• Material Removal: {vol_red_pct:.1f}% volume removed\n"
        f"• Converged Vf:     {actual_vf:.3f} (Error: {abs(actual_vf - volfrac):.4f})\n"
        f"───────────────────────────────────────────\n"
        f"BOUNDARY SUPPORTS ({len(supp_coords or [])} constrained nodes)\n"
        f"{supp_text}\n"
        f"───────────────────────────────────────────\n"
        f"APPLIED LOADS ({len(load_details or [])} point forces)\n"
        f"{load_text}\n"
        f"───────────────────────────────────────────\n"
        f"STRUCTURAL PERFORMANCE METRICS\n"
        f"• Compliance:       C₀ = {initial_c:.2f} ➔ C* = {final_c:.2f}\n"
        f"• Stiffness Gain:   {c_red_pct:.1f}% stiffer than initial block\n"
        f"• Critical Buckling: λ₁ = {lambda_str}\n"
        f"• Convergence:      {meta.get('iterations_run', 50)} iters in {meta.get('elapsed_time', 0.0):.1f}s\n"
        f"═══════════════════════════════════════════"
    )

    ax1.text(0.5, 0.5, scorecard, transform=ax1.transAxes,
             fontsize=8.2, family='monospace', va='center', ha='center',
             bbox=dict(boxstyle='round,pad=0.7', facecolor='#F4F6F9', edgecolor='#455A64', lw=1.5))

    # ============================================================
    # Panel 3: AFTER Optimization (Converged SIMP Topology)
    # ============================================================
    ax2 = axes[2]
    draw_domain_and_bcs(ax2, ch_bc, nelx, nely, is_optimized=True, target=final_density,
                        support_coords=supp_coords, ch_bc_x=ch_bc_x, ch_bc_y=ch_bc_y)
    draw_loads(ax2, loads, arrow_len=18)

    ax2.set_xlim(-margin_x, nelx + margin_x)
    ax2.set_ylim(nely + margin_y, -margin_y)
    ax2.set_aspect('equal')
    ax2.set_title(
        f"2. AFTER Optimization: Converged SIMP Topology\n"
        f"Actual Vf = {actual_vf:.3f} | C* = {final_c:.2f} (-{c_red_pct:.1f}%) | λ₁ = {lambda_str}",
        fontsize=11, fontweight='bold', pad=10
    )
    ax2.set_xlabel("X (elements)", fontsize=9.5)
    ax2.set_ylabel("Y (elements)", fontsize=9.5)

    leg_items_2 = [
        mpatches.Patch(facecolor='#111111', edgecolor='#333333', label=f'Optimized Topology (Vf* = {actual_vf:.3f})'),
        mpatches.Patch(facecolor='#D32F2F', edgecolor='#B71C1C', label='Boundary Support'),
        mpatches.Patch(facecolor='#1565C0', edgecolor='#0D47A1', label='Applied Load Vector')
    ]
    ax2.legend(handles=leg_items_2, loc='upper right', fontsize=8.2, framealpha=0.92)

    fig.suptitle(
        f"Phase 1 SIMP Topology Optimization — Before & After Evaluation [{case_name} | {mode_str}]",
        fontsize=13.5, fontweight='bold', y=0.98
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved Before & After figure: {os.path.abspath(output_path)}")


def render_buckling_head_to_head(input_tensor, meta_no_buckle, meta_with_buckle, output_path):
    """
    Creates a 4-panel Head-to-Head figure comparing Pure Compliance SIMP vs Buckling-Aware SIMP:
      Panel 1: BEFORE Optimization (Initial Block & Problem Specification)
      Panel 2: AFTER Pure Compliance SIMP (Without Buckling: C*, lambda_1)
      Panel 3: AFTER Buckling-Aware SIMP (With Buckling: C*, lambda_1, % gain)
      Panel 4: Material Redistribution Delta rho = rho_buckle - rho_pure (coolwarm heatmap)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    nelx = meta_no_buckle.get('nelx', 120)
    nely = meta_no_buckle.get('nely', 60)
    case_name = meta_no_buckle.get('case_name', 'Benchmark').upper()

    if input_tensor.shape[0] == 6:
        ch_domain = input_tensor[0]
        ch_bc_x = input_tensor[1]
        ch_bc_y = input_tensor[2]
        ch_fx = input_tensor[3]
        ch_fy = input_tensor[4]
        ch_vf = input_tensor[5]
        ch_bc = np.maximum(ch_bc_x, ch_bc_y)
        volfrac = meta_no_buckle.get('volfrac', float(ch_vf[0, 0]))
    else:
        ch_domain = input_tensor[0]
        ch_bc = input_tensor[1]
        ch_fx = input_tensor[2]
        ch_fy = input_tensor[3]
        ch_vf = input_tensor[4]
        ch_bc_x = ch_bc
        ch_bc_y = ch_bc
        volfrac = meta_no_buckle.get('volfrac', float(ch_vf[0, 0]))

    vol_red_pct = meta_no_buckle.get('volume_reduction_pct', (1.0 - volfrac) * 100.0)
    initial_density = meta_no_buckle.get('initial_density', volfrac)

    supp_coords = meta_no_buckle.get('support_coords', None)
    load_details = meta_no_buckle.get('load_details', None)
    if load_details is not None and len(load_details) > 0:
        loads = load_details
    else:
        loads = extract_discrete_loads(ch_fx, ch_fy)

    # Metrics Pure Compliance
    rho_pure = meta_no_buckle['final_density']
    c_pure = meta_no_buckle['final_compliance']
    c_red_pure = meta_no_buckle['compliance_reduction_pct']
    lam_pure = meta_no_buckle['final_lambda_1']
    lam_pure_str = f"{lam_pure:.2f}" if (0.0 < lam_pure < 900) else "N/A"

    # Metrics Buckling Aware
    rho_buckle = meta_with_buckle['final_density']
    c_buckle = meta_with_buckle['final_compliance']
    c_red_buckle = meta_with_buckle['compliance_reduction_pct']
    lam_buckle = meta_with_buckle['final_lambda_1']
    wb = meta_with_buckle.get('buckle_weight', 0.45)
    lam_buckle_str = f"{lam_buckle:.2f}" if (0.0 < lam_buckle < 900) else "N/A"

    # Gain in buckling factor & compliance tradeoff
    if (0.0 < lam_pure < 900) and (0.0 < lam_buckle < 900):
        lam_gain = ((lam_buckle - lam_pure) / max(1e-6, lam_pure)) * 100.0
        lam_gain_str = f"+{lam_gain:.1f}% λ₁ gain"
    else:
        lam_gain_str = "λ₁ active"
    comp_diff = ((c_buckle - c_pure) / max(1e-6, c_pure)) * 100.0

    # Material redistribution
    delta_rho = rho_buckle - rho_pure
    pos_mass = float(np.sum(delta_rho[delta_rho > 0.05]))
    neg_mass = float(np.sum(np.abs(delta_rho[delta_rho < -0.05])))

    fig, axes = plt.subplots(2, 2, figsize=(21, 12))
    margin_x = 24
    margin_y = 22

    # ============================================================
    # Panel 1: BEFORE Optimization (Initial Uniform Block)
    # ============================================================
    ax0 = axes[0, 0]
    draw_initial_structure_and_bcs(ax0, ch_bc, initial_density, nelx, nely,
                                   support_coords=supp_coords, ch_bc_x=ch_bc_x, ch_bc_y=ch_bc_y)
    draw_loads(ax0, loads, arrow_len=18)
    ax0.set_xlim(-margin_x, nelx + margin_x)
    ax0.set_ylim(nely + margin_y, -margin_y)
    ax0.set_aspect('equal')
    ax0.set_title(
        f"1. BEFORE Optimization: Problem Specification & Initial Block\n"
        f"Target Vf = {volfrac:.2f} ({vol_red_pct:.1f}% volume reduction) | Initial C₀ = {meta_no_buckle.get('initial_compliance', 0.0):.2f}",
        fontsize=11, fontweight='bold', pad=10
    )
    ax0.set_xlabel("X (elements)", fontsize=9.5)
    ax0.set_ylabel("Y (elements)", fontsize=9.5)
    leg_0 = [
        mpatches.Patch(facecolor='#888888', edgecolor='#333333', label=f'Initial Uniform Block (ρ⁽⁰⁾ = {volfrac:.2f})'),
        mpatches.Patch(facecolor='#D32F2F', edgecolor='#B71C1C', label='Boundary Supports'),
        mpatches.Patch(facecolor='#1565C0', edgecolor='#0D47A1', label='Applied Forces')
    ]
    ax0.legend(handles=leg_0, loc='upper right', fontsize=8.2, framealpha=0.92)

    # ============================================================
    # Panel 2: AFTER Pure Compliance SIMP (Without Buckling)
    # ============================================================
    ax1 = axes[0, 1]
    draw_domain_and_bcs(ax1, ch_bc, nelx, nely, is_optimized=True, target=rho_pure,
                        support_coords=supp_coords, ch_bc_x=ch_bc_x, ch_bc_y=ch_bc_y)
    draw_loads(ax1, loads, arrow_len=18)
    ax1.set_xlim(-margin_x, nelx + margin_x)
    ax1.set_ylim(nely + margin_y, -margin_y)
    ax1.set_aspect('equal')
    ax1.set_title(
        f"2. AFTER: SIMP Without Buckling (Pure Compliance Minimization)\n"
        f"C* = {c_pure:.2f} (-{c_red_pure:.1f}%) | Critical Buckling Factor λ₁ = {lam_pure_str}",
        fontsize=11, fontweight='bold', pad=10
    )
    ax1.set_xlabel("X (elements)", fontsize=9.5)
    ax1.set_ylabel("Y (elements)", fontsize=9.5)
    leg_1 = [
        mpatches.Patch(facecolor='#111111', edgecolor='#333333', label=f'Pure Compliance (Vf* = {meta_no_buckle.get("actual_vf", volfrac):.3f})')
    ]
    ax1.legend(handles=leg_1, loc='upper right', fontsize=8.2, framealpha=0.92)

    # ============================================================
    # Panel 3: AFTER Buckling-Aware SIMP (With Buckling)
    # ============================================================
    ax2 = axes[1, 0]
    draw_domain_and_bcs(ax2, ch_bc, nelx, nely, is_optimized=True, target=rho_buckle,
                        support_coords=supp_coords, ch_bc_x=ch_bc_x, ch_bc_y=ch_bc_y)
    draw_loads(ax2, loads, arrow_len=18)
    ax2.set_xlim(-margin_x, nelx + margin_x)
    ax2.set_ylim(nely + margin_y, -margin_y)
    ax2.set_aspect('equal')
    ax2.set_title(
        f"3. AFTER: SIMP With Buckling (Buckling-Aware, wb = {wb:.2f})\n"
        f"C* = {c_buckle:.2f} ({comp_diff:+.1f}%) | λ₁ = {lam_buckle_str} ({lam_gain_str}!)",
        fontsize=11, fontweight='bold', pad=10
    )
    ax2.set_xlabel("X (elements)", fontsize=9.5)
    ax2.set_ylabel("Y (elements)", fontsize=9.5)
    leg_2 = [
        mpatches.Patch(facecolor='#111111', edgecolor='#333333', label=f'Buckling-Aware (Vf* = {meta_with_buckle.get("actual_vf", volfrac):.3f})')
    ]
    ax2.legend(handles=leg_2, loc='upper right', fontsize=8.2, framealpha=0.92)

    # ============================================================
    # Panel 4: Material Redistribution Delta rho
    # ============================================================
    ax3 = axes[1, 1]
    vmax = max(0.4, float(np.max(np.abs(delta_rho))))
    im = ax3.imshow(delta_rho, cmap='coolwarm', vmin=-vmax, vmax=vmax,
                    extent=[0, nelx, nely, 0], zorder=1)

    domain_border = mpatches.Rectangle((0, 0), nelx, nely, fill=False,
                                       edgecolor='#333333', lw=2.0, zorder=5)
    ax3.add_patch(domain_border)
    draw_supports(ax3, ch_bc, nelx, nely, support_coords=supp_coords, ch_bc_x=ch_bc_x, ch_bc_y=ch_bc_y)
    draw_loads(ax3, loads, arrow_len=18)

    ax3.set_xlim(-margin_x, nelx + margin_x)
    ax3.set_ylim(nely + margin_y, -margin_y)
    ax3.set_aspect('equal')
    ax3.set_title(
        f"4. Material Redistribution Δρ (Buckling-Aware − Pure Compliance)\n"
        f"Red = Reinforced Struts (+{pos_mass:.1f} mass) | Blue = Shed Void (-{neg_mass:.1f} mass)",
        fontsize=11, fontweight='bold', pad=10
    )
    ax3.set_xlabel("X (elements)", fontsize=9.5)
    ax3.set_ylabel("Y (elements)", fontsize=9.5)

    cbar = plt.colorbar(im, ax=ax3, orientation='horizontal', pad=0.12, shrink=0.75)
    cbar.set_label(r"Density Difference $\Delta\rho = \rho_{\mathrm{buckle}} - \rho_{\mathrm{pure}}$ (Cool: Shed Material, Warm: Thickened Member)", fontsize=9.0)

    fig.suptitle(
        f"Phase 1 Head-to-Head Comparison: SIMP Without vs. With In-Plane Buckling [{case_name}]",
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved Head-to-Head comparison figure: {os.path.abspath(output_path)}")


def visualize_dataset(npz_path='topo_dataset_1000.npz', output_dir='images', max_samples=10):
    """Backward compatible visualizer for npz dataset files."""
    if not os.path.exists(npz_path):
        print(f"Error: Dataset file '{npz_path}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_path)
    inputs = data['inputs']
    targets = data['targets']
    compliance = data['compliance'] if 'compliance' in data else None
    strain_energy = data['strain_energy'] if 'strain_energy' in data else None

    total_samples = inputs.shape[0]
    num_to_plot = min(total_samples, max_samples)
    print(f"Visualizing {num_to_plot} of {total_samples} samples from '{npz_path}'...")

    nely, nelx = 60, 120
    margin_x = 22
    margin_y = 20

    for i in range(num_to_plot):
        inp = inputs[i]
        target = targets[i]
        if inp.shape[0] == 6:
            ch_domain = inp[0]
            ch_bc_x = inp[1]
            ch_bc_y = inp[2]
            ch_fx = inp[3]
            ch_fy = inp[4]
            ch_vf = inp[5]
            ch_bc = np.maximum(ch_bc_x, ch_bc_y)
        else:
            ch_domain = inp[0]
            ch_bc = inp[1]
            ch_fx = inp[2]
            ch_fy = inp[3]
            ch_vf = inp[4]
            ch_bc_x = ch_bc
            ch_bc_y = ch_bc

        target_vf = float(ch_vf[0, 0])
        actual_vf = float(np.mean(target))
        loads = extract_discrete_loads(ch_fx, ch_fy)
        supp_coords = extract_supports_from_bcs(ch_bc_x, ch_bc_y, nelx, nely)

        fig, axes = plt.subplots(1, 3, figsize=(22, 6.5),
                                 gridspec_kw={'width_ratios': [1.1, 0.4, 1.1]})

        ax0 = axes[0]
        draw_domain_and_bcs(ax0, ch_bc, nelx, nely, is_optimized=False,
                            support_coords=supp_coords, ch_bc_x=ch_bc_x, ch_bc_y=ch_bc_y)
        draw_loads(ax0, loads, arrow_len=18)
        ax0.set_xlim(-margin_x, nelx + margin_x)
        ax0.set_ylim(nely + margin_y, -margin_y)
        ax0.set_aspect('equal')
        ax0.set_title(f"Input Conditions (Problem Specification)\nTarget Volume Fraction Vf = {target_vf:.3f}",
                      fontsize=12, fontweight='bold', pad=10)
        ax0.set_xlabel("X (elements)", fontsize=10)
        ax0.set_ylabel("Y (elements)", fontsize=10)

        ax1 = axes[1]
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.annotate('', xy=(8.5, 5), xytext=(1.5, 5),
                     arrowprops=dict(arrowstyle='->', color='#1A237E', lw=4, mutation_scale=28))
        ax1.text(5, 7.2, "SIMP\nOptimization", ha='center', va='center',
                 fontsize=14, fontweight='bold', color='#1A237E')
        ax1.text(5, 2.8,
                 f"• Objective: Min Compliance\n"
                 f"• Penalization: p = 3.0\n"
                 f"• Sensitivity Filter: R = 1.5\n"
                 f"• Max Iterations: 60-80\n"
                 f"• Self-Weight: Excluded",
                 ha='center', va='center', fontsize=9.5, color='#37474F',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECEFF1', edgecolor='#B0BEC5', alpha=0.9))
        ax1.axis('off')

        ax2 = axes[2]
        draw_domain_and_bcs(ax2, ch_bc, nelx, nely, is_optimized=True, target=target,
                            support_coords=supp_coords, ch_bc_x=ch_bc_x, ch_bc_y=ch_bc_y)
        draw_loads(ax2, loads, arrow_len=18)
        ax2.set_xlim(-margin_x, nelx + margin_x)
        ax2.set_ylim(nely + margin_y, -margin_y)
        ax2.set_aspect('equal')
        c_info = f" | Compliance C* = {compliance[i]:.2f}" if (compliance is not None and i < len(compliance)) else f" (Error: {abs(actual_vf - target_vf):.4f})"
        ax2.set_title(f"Optimized Structure (Ground Truth Target)\nActual Vf = {actual_vf:.3f}{c_info}",
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
    parser.add_argument("--dataset", type=str, default="topo_dataset_1000.npz", help="Path to .npz dataset file")
    parser.add_argument("--output_dir", type=str, default="images", help="Output directory for PNG visualizations")
    parser.add_argument("--max_samples", type=int, default=10, help="Maximum number of samples to visualize")
    args = parser.parse_args()

    visualize_dataset(npz_path=args.dataset, output_dir=args.output_dir, max_samples=args.max_samples)


if __name__ == '__main__':
    main()
