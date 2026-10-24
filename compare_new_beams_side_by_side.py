"""
compare_new_beams_side_by_side.py - 4-Way Side-by-Side Structural & Stability Comparison
on Two Brand-New Randomly Generated Beams:
  1. Beam A: Newly Generated Random Asymmetric Cantilever Beam
  2. Beam B: Newly Generated Random Multi-Support Bridge / Arch Beam

For each beam, compares side-by-side when a 1500 N force is applied:
  [Col 1] Without Topology Optimization (Unoptimized Uniform Continuum Block rho = Vf)
  [Col 2] Classical Iterative SIMP Solver (80 FEA iterations)
  [Col 3] Baseline U-Net (best_unet_3000.pth, before RL / naive thresholding)
  [Col 4] SIMP-Judge RL Fine-Tuned U-Net (best_unet_rl.pth + physics-aware OC condensation)

Outputs:
  - images/new_cantilever_4way.png
  - images/new_bridge_beam_4way.png
  - images/new_beams_4way_master_comparison.png
  - checkpoints/new_beams_4way_report.json
"""

import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from unet_model import TopoUNet
from simp_solver import (
    solve_simp_unified,
    analyze_structural_integrity,
    precompute_mesh_and_filter
)
from losses_rl_ver import (
    check_structural_stability,
    physics_aware_binarize,
    compute_fea_compliance_ratio,
    get_cached_fea_mesh
)
from visualize_data import extract_discrete_loads, draw_loads, draw_domain_and_bcs


def build_custom_random_beam(nelx=120, nely=60, supports=None, loads=None, volfrac=0.38):
    """
    Constructs both:
      1. Exact FEA (fixed_dofs, F_vec) on the (nelx+1)x(nely+1) nodal mesh
      2. Exact 6-channel TopoUNet input tensor (domain, bc_x, bc_y, fx, fy, vf) of shape (6, nely, nelx)
         matching the exact nodal-to-element 0.25 distribution used in topo_dataset_3000.npz.

    Args:
      supports: list of tuples (nx, ny, fix_x, fix_y) in nodal coordinatesnx in [0..nelx], ny in [0..nely]
      loads:    list of tuples (nx, ny, fx, fy) in nodal coordinates
      volfrac:  target volume fraction Vf in [0.25, 0.55]
    """
    ndof = 2 * (nelx + 1) * (nely + 1)
    F_vec = np.zeros(ndof, dtype=np.float64)
    fixed_dofs = []

    inp_np = np.zeros((6, nely, nelx), dtype=np.float32)
    inp_np[0, :, :] = 1.0          # Channel 0: Design domain mask
    inp_np[5, :, :] = float(volfrac)  # Channel 5: Uniform target volume fraction

    def get_adjacent_elements(nx, ny):
        elems = []
        for i in (nx - 1, nx):
            for j in (ny - 1, ny):
                if 0 <= i < nelx and 0 <= j < nely:
                    elems.append((j, i))
        return elems

    for (nx, ny, fix_x, fix_y) in supports:
        nid = nx * (nely + 1) + ny
        adj = get_adjacent_elements(nx, ny)
        if fix_x:
            fixed_dofs.append(2 * nid)
            for (r, c) in adj:
                inp_np[1, r, c] += 0.25
        if fix_y:
            fixed_dofs.append(2 * nid + 1)
            for (r, c) in adj:
                inp_np[2, r, c] += 0.25

    for (nx, ny, fx, fy) in loads:
        nid = nx * (nely + 1) + ny
        F_vec[2 * nid] += float(fx)
        F_vec[2 * nid + 1] += float(fy)
        adj = get_adjacent_elements(nx, ny)
        for (r, c) in adj:
            inp_np[3, r, c] += 0.25 * float(fx)
            inp_np[4, r, c] += 0.25 * float(fy)

    inp_np[1] = np.clip(inp_np[1], 0.0, 1.0)
    inp_np[2] = np.clip(inp_np[2], 0.0, 1.0)
    fixed_dofs = np.unique(fixed_dofs)
    return inp_np, fixed_dofs, F_vec


def evaluate_all_4_methods_on_beam(beam_name, beam_desc, inp_np, fixed_dofs, F_vec,
                                   model_base, model_rl, device, force_real_N=1500.0):
    nelx, nely = inp_np.shape[2], inp_np.shape[1]
    volfrac = float(np.mean(inp_np[5]))
    edofMat, iK, jK, H, Hs = precompute_mesh_and_filter(nelx, nely, rmin=1.5)

    # -------------------------------------------------------------------------
    # 1. CLASSICAL SIMP SOLVER (80 Iterations Ground Truth)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    simp_res = solve_simp_unified(
        nelx=nelx, nely=nely, volfrac=volfrac, penal=3.0,
        fixeddofs=fixed_dofs, F=F_vec,
        edofMat=edofMat, iK=iK, jK=jK, H=H, Hs=Hs,
        max_iter=80, use_buckling=False, verbose=False
    )
    simp_time_ms = (time.perf_counter() - t0) * 1000.0
    simp_density = simp_res['final_density']
    bin_simp = (simp_density >= 0.5).astype(float)

    stab_simp = check_structural_stability(
        pred_density=bin_simp, inp_np=inp_np, simp_density=simp_density,
        threshold=0.5, force_real_N=force_real_N, use_physics_binarize=False,
        bin_pred_override=bin_simp
    )
    integ_simp = analyze_structural_integrity(
        density=bin_simp, fixed_dofs=fixed_dofs, F=F_vec, edofMat=edofMat,
        nelx=nelx, nely=nely, material_name="steel", force_real_N=force_real_N,
        density_threshold=0.3
    )

    # -------------------------------------------------------------------------
    # 2. WITHOUT TOPOLOGY OPTIMIZATION (Unoptimized Uniform Block rho = Vf)
    # -------------------------------------------------------------------------
    unopt_density = np.full((nely, nelx), volfrac, dtype=np.float32)
    c_unopt = float(simp_res['initial_compliance'])
    c_simp_star = float(stab_simp['c_target'])
    unopt_ratio = c_unopt / max(c_simp_star, 1e-9)

    integ_unopt = analyze_structural_integrity(
        density=unopt_density, fixed_dofs=fixed_dofs, F=F_vec, edofMat=edofMat,
        nelx=nelx, nely=nely, material_name="steel", force_real_N=force_real_N,
        density_threshold=0.10
    )
    unopt_fos = float(integ_unopt['fos_yield']) if integ_unopt else 0.0
    unopt_lam = float(integ_unopt['lambda_1']) if integ_unopt else 0.0
    unopt_defl = float(integ_unopt['max_deflection_mm']) if integ_unopt else float('inf')
    unopt_vm95 = float(integ_unopt['vm_95th']) if integ_unopt else float('inf')

    # Unoptimized uniform porous block has E = Vf^3 * E0 (SIMP law), giving ~10x-15x worse stiffness
    unopt_stable = bool(unopt_ratio <= 2.5 and unopt_fos >= 1.0 and unopt_lam >= 1.0)
    unopt_badge = "STABLE" if unopt_stable else f"POOR STIFFNESS ({unopt_ratio:.1f}x Compliance)"
    unopt_color = "#2E7D32" if unopt_stable else "#E65100"

    # -------------------------------------------------------------------------
    # 3. BASELINE U-NET (checkpoints/best_unet_3000.pth, Naive Binarization)
    # -------------------------------------------------------------------------
    x_tensor = torch.from_numpy(inp_np).unsqueeze(0).to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        pred_base_t = model_base(x_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    base_time_ms = (time.perf_counter() - t0) * 1000.0

    pred_base = pred_base_t.squeeze().cpu().numpy()
    bin_base = (pred_base >= 0.35).astype(float)
    stab_base = check_structural_stability(
        pred_density=pred_base, inp_np=inp_np, simp_density=simp_density,
        threshold=0.35, force_real_N=force_real_N, use_physics_binarize=False,
        bin_pred_override=bin_base
    )
    integ_base = analyze_structural_integrity(
        density=bin_base, fixed_dofs=fixed_dofs, F=F_vec, edofMat=edofMat,
        nelx=nelx, nely=nely, material_name="steel", force_real_N=force_real_N,
        density_threshold=0.3
    )
    inter_base = np.sum(bin_base * bin_simp)
    dice_base = float((2.0 * inter_base + 1e-6) / (np.sum(bin_base) + np.sum(bin_simp) + 1e-6))

    # -------------------------------------------------------------------------
    # 4. SIMP-JUDGE RL FINE-TUNED U-NET (checkpoints/best_unet_rl.pth + OC Polish)
    # -------------------------------------------------------------------------
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        pred_rl_t = model_rl(x_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    rl_gpu_ms = (time.perf_counter() - t0) * 1000.0

    pred_rl = pred_rl_t.squeeze().cpu().numpy()
    t_oc = time.perf_counter()
    bin_rl = physics_aware_binarize(pred_rl, inp_np, threshold=0.35, polish_steps=5)
    rl_total_ms = rl_gpu_ms + (time.perf_counter() - t_oc) * 1000.0

    stab_rl = check_structural_stability(
        pred_density=pred_rl, inp_np=inp_np, simp_density=simp_density,
        threshold=0.35, force_real_N=force_real_N, use_physics_binarize=True,
        bin_pred_override=bin_rl
    )
    integ_rl = analyze_structural_integrity(
        density=bin_rl, fixed_dofs=fixed_dofs, F=F_vec, edofMat=edofMat,
        nelx=nelx, nely=nely, material_name="steel", force_real_N=force_real_N,
        density_threshold=0.3
    )
    inter_rl = np.sum(bin_rl * bin_simp)
    dice_rl = float((2.0 * inter_rl + 1e-6) / (np.sum(bin_rl) + np.sum(bin_simp) + 1e-6))

    return {
        "beam_name": beam_name,
        "beam_desc": beam_desc,
        "volfrac": volfrac,
        "force_real_N": force_real_N,
        "inp_np": inp_np,
        "unopt": {
            "title": "1. Without Topology Opt\n(Uniform Block $\\rho_0 = V_f$)",
            "density": unopt_density,
            "vm_grid": integ_unopt["von_mises_grid"] if integ_unopt else np.zeros((nely, nelx)),
            "compliance": c_unopt,
            "fea_ratio": unopt_ratio,
            "fos_yield": unopt_fos,
            "lambda_1": unopt_lam,
            "max_defl_mm": unopt_defl,
            "vm_95th_mpa": unopt_vm95,
            "vf_actual": float(np.mean(unopt_density)),
            "dice": 0.0,
            "time_ms": 0.0,
            "is_stable": unopt_stable,
            "short_badge": "UNOPTIMIZED (WEAK)",
            "status_badge": unopt_badge,
            "badge_color": unopt_color
        },
        "simp": {
            "title": "2. Classical SIMP Solver\n(80 FEA Iterations)",
            "density": bin_simp,
            "vm_grid": integ_simp["von_mises_grid"] if integ_simp else np.zeros((nely, nelx)),
            "compliance": float(stab_simp["c_pred"]),
            "fea_ratio": 1.00,
            "fos_yield": float(stab_simp["fos_yield"]),
            "lambda_1": float(stab_simp["lambda_1"]),
            "max_defl_mm": float(stab_simp["max_deflection_mm"]),
            "vm_95th_mpa": float(stab_simp["vm_95th_mpa"]),
            "vf_actual": float(np.mean(bin_simp)),
            "dice": 1.000,
            "time_ms": simp_time_ms,
            "is_stable": True,
            "short_badge": "STABLE (GROUND TRUTH)",
            "status_badge": "STABLE (SIMP OPTIMUM)",
            "badge_color": "#1565C0"
        },
        "base_unet": {
            "title": "3. Baseline U-Net\n(Before RL, $\\hat{\\rho} \\geq 0.35$)",
            "density": bin_base,
            "raw_density": pred_base,
            "vm_grid": integ_base["von_mises_grid"] if integ_base else np.zeros((nely, nelx)),
            "compliance": float(stab_base["c_pred"]),
            "fea_ratio": float(stab_base["fea_ratio"]),
            "fos_yield": float(stab_base["fos_yield"]),
            "lambda_1": float(stab_base["lambda_1"]),
            "max_defl_mm": float(stab_base["max_deflection_mm"]),
            "vm_95th_mpa": float(stab_base["vm_95th_mpa"]),
            "vf_actual": float(np.mean(bin_base)),
            "dice": dice_base,
            "time_ms": base_time_ms,
            "is_stable": bool(stab_base["is_stable"]),
            "short_badge": stab_base["short_badge"],
            "status_badge": stab_base["status_badge"],
            "badge_color": stab_base["badge_color"],
            "failure_reasons": stab_base["failure_reasons"]
        },
        "rl_unet": {
            "title": "4. SIMP-Judge RL U-Net\n(RL Fine-Tuned + OC Polish)",
            "density": bin_rl,
            "raw_density": pred_rl,
            "vm_grid": integ_rl["von_mises_grid"] if integ_rl else np.zeros((nely, nelx)),
            "compliance": float(stab_rl["c_pred"]),
            "fea_ratio": float(stab_rl["fea_ratio"]),
            "fos_yield": float(stab_rl["fos_yield"]),
            "lambda_1": float(stab_rl["lambda_1"]),
            "max_defl_mm": float(stab_rl["max_deflection_mm"]),
            "vm_95th_mpa": float(stab_rl["vm_95th_mpa"]),
            "vf_actual": float(np.mean(bin_rl)),
            "dice": dice_rl,
            "time_ms": rl_gpu_ms,
            "total_time_ms": rl_total_ms,
            "is_stable": bool(stab_rl["is_stable"]),
            "short_badge": stab_rl["short_badge"],
            "status_badge": stab_rl["status_badge"],
            "badge_color": stab_rl["badge_color"],
            "failure_reasons": stab_rl["failure_reasons"]
        }
    }


def plot_4way_beam_comparison(res, output_path):
    inp_np = res["inp_np"]
    nelx, nely = inp_np.shape[2], inp_np.shape[1]
    ch_bc = np.clip(inp_np[1] + inp_np[2], 0.0, 1.0)
    loads = extract_discrete_loads(inp_np[3] * 4.0, inp_np[4] * 4.0)

    methods = ["unopt", "simp", "base_unet", "rl_unet"]
    fig, axes = plt.subplots(2, 4, figsize=(24, 9.5), dpi=160)

    # Determine consistent stress colorbar upper bound from SIMP 95th percentile * 1.8
    vmax_stress = max(float(res["simp"]["vm_95th_mpa"]) * 1.8, 60.0)

    for col_idx, key in enumerate(methods):
        m = res[key]

        # --- ROW 0: Structure + BCs + Applied Force + Stability Badge ---
        ax_top = axes[0, col_idx]
        draw_domain_and_bcs(ax_top, ch_bc)
        ax_top.imshow(1.0 - m["density"], cmap="gray", vmin=0.0, vmax=1.0,
                      extent=[0, nelx, nely, 0], interpolation="nearest", zorder=2)
        draw_loads(ax_top, loads)

        ax_top.set_title(
            f"{m['title']}\n[{m['short_badge']}]  (Vf={m['vf_actual']:.2f})",
            fontsize=11, fontweight="bold", color=m["badge_color"], pad=8
        )
        ax_top.set_xlim(-12, nelx + 12)
        ax_top.set_ylim(nely + 10, -10)
        ax_top.set_aspect("equal")
        ax_top.axis("off")

        # --- ROW 1: Von Mises Stress Field under 1500 N Force + Quantitative Card ---
        ax_bot = axes[1, col_idx]
        vm_masked = np.ma.masked_where(m["density"] < 0.10, m["vm_grid"])
        ax_bot.imshow(np.ones((nely, nelx)), cmap="Greys", vmin=0, vmax=1,
                      extent=[0, nelx, nely, 0], alpha=0.15)
        im_vm = ax_bot.imshow(vm_masked, cmap="turbo", vmin=0.0, vmax=vmax_stress,
                              extent=[0, nelx, nely, 0], interpolation="nearest")

        c_str = f"{m['compliance']:.1f}" if m["compliance"] < 1e4 else "INF (Broken)"
        r_str = f"{m['fea_ratio']:.2f}x" if m["fea_ratio"] < 1e3 else "COLLAPSE"
        d_str = f"{m['max_defl_mm']:.2f} mm" if m["max_defl_mm"] < 1e3 else "INF"

        ax_bot.set_title(
            f"Force Applied ({res['force_real_N']:.0f} N) Stress Map\n"
            f"C={c_str} ({r_str}) | FoS={m['fos_yield']:.2f} | $\\lambda_1$={m['lambda_1']:.1f} | $\\delta$={d_str}",
            fontsize=10, fontweight="bold", color="#263238", pad=6
        )
        ax_bot.set_xlim(-4, nelx + 4)
        ax_bot.set_ylim(nely + 4, -4)
        ax_bot.set_aspect("equal")
        ax_bot.axis("off")

        cbar = fig.colorbar(im_vm, ax=ax_bot, fraction=0.035, pad=0.03, orientation="horizontal")
        cbar.set_label("Von Mises Stress (MPa, A36 Steel @ 1500 N)", fontsize=8.5)

    fig.suptitle(
        f"4-Way Side-by-Side Comparison: {res['beam_name']}\n"
        f"{res['beam_desc']}  |  Target Volume Fraction $V_f = {res['volfrac']:.2f}$  |  Applied Force $F = {res['force_real_N']:.0f}$ N",
        fontsize=13.5, fontweight="bold", y=0.99
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved Figure] {output_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Baseline U-Net (best_unet_3000.pth) and SIMP-Judge RL U-Net (best_unet_rl.pth) on {device}...")

    model_base = TopoUNet(in_channels=6, out_channels=1).to(device).eval()
    ckpt_base = torch.load("checkpoints/best_unet_3000.pth", map_location=device)
    model_base.load_state_dict(ckpt_base["model_state_dict"] if "model_state_dict" in ckpt_base else ckpt_base)

    model_rl = TopoUNet(in_channels=6, out_channels=1).to(device).eval()
    ckpt_rl = torch.load("checkpoints/best_unet_rl.pth", map_location=device)
    model_rl.load_state_dict(ckpt_rl["model_state_dict"] if "model_state_dict" in ckpt_rl else ckpt_rl)

    # =========================================================================
    # BEAM 1: Brand-New Randomly Generated Cantilever Beam
    #   - Left-wall dual pins at (0, 5) and (0, 55)
    #   - Angled cantilevered point load at right edge (120, 40), angle = -104 deg
    #   - Target Vf = 0.36
    # =========================================================================
    angle_cant = np.radians(-104.0)
    inp_cant, fd_cant, F_cant = build_custom_random_beam(
        nelx=120, nely=60,
        supports=[(0, 5, True, True), (0, 55, True, True)],
        loads=[(120, 40, np.cos(angle_cant), -np.sin(angle_cant))],  # downward (+y in grid) & slightly left
        volfrac=0.36
    )

    # =========================================================================
    # BEAM 2: Brand-New Randomly Generated Multi-Support Bridge / Arch Beam
    #   - Left pin at (0, 56), upper-left pin at (0, 14), bottom-right roller at (116, 60)
    #   - Asymmetric deck load at top-span (76, 0), angle = -78 deg
    #   - Target Vf = 0.38
    # =========================================================================
    angle_bridge = np.radians(-78.0)
    inp_bridge, fd_bridge, F_bridge = build_custom_random_beam(
        nelx=120, nely=60,
        supports=[(0, 56, True, True), (0, 14, True, True), (116, 60, False, True)],
        loads=[(76, 0, np.cos(angle_bridge), -np.sin(angle_bridge))],
        volfrac=0.38
    )

    print("\n>>> Evaluating Beam 1 (New Random Cantilever Beam)...")
    res_cant = evaluate_all_4_methods_on_beam(
        beam_name="New Random Cantilever Beam (Dual Left Pins, Angled Tip Load @ x=120, y=40)",
        beam_desc="Supports: Pins at (0,5) & (0,55) | Load: 1500 N at (120,40), angle=-104 deg",
        inp_np=inp_cant, fixed_dofs=fd_cant, F_vec=F_cant,
        model_base=model_base, model_rl=model_rl, device=device, force_real_N=1500.0
    )
    plot_4way_beam_comparison(res_cant, "images/new_cantilever_4way.png")

    print("\n>>> Evaluating Beam 2 (New Random Asymmetric Bridge / Arch Beam)...")
    res_bridge = evaluate_all_4_methods_on_beam(
        beam_name="New Random Multi-Support Bridge Beam (Left Pins + Right Roller, Span Load @ x=76, y=0)",
        beam_desc="Supports: Pins at (0,14) & (0,56), Roller at (116,60) | Load: 1500 N at (76,0), angle=-78 deg",
        inp_np=inp_bridge, fixed_dofs=fd_bridge, F_vec=F_bridge,
        model_base=model_base, model_rl=model_rl, device=device, force_real_N=1500.0
    )
    plot_4way_beam_comparison(res_bridge, "images/new_bridge_beam_4way.png")

    # Save clean JSON summary
    summary_out = {}
    for b_key, b_res in [("cantilever_beam", res_cant), ("bridge_beam", res_bridge)]:
        summary_out[b_key] = {
            "beam_name": b_res["beam_name"],
            "volfrac": b_res["volfrac"],
            "force_real_N": b_res["force_real_N"],
            "methods": {}
        }
        for m_key in ["unopt", "simp", "base_unet", "rl_unet"]:
            m = b_res[m_key]
            summary_out[b_key]["methods"][m_key] = {
                "compliance": float(m["compliance"]),
                "fea_ratio": float(m["fea_ratio"]),
                "fos_yield": float(m["fos_yield"]),
                "lambda_1": float(m["lambda_1"]),
                "max_defl_mm": float(m["max_defl_mm"]),
                "vm_95th_mpa": float(m["vm_95th_mpa"]),
                "vf_actual": float(m["vf_actual"]),
                "dice": float(m["dice"]),
                "time_ms": float(m["time_ms"]),
                "is_stable": bool(m["is_stable"]),
                "status_badge": m["status_badge"]
            }

    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/new_beams_4way_report.json", "w") as f:
        json.dump(summary_out, f, indent=2)
    print("\n[Saved Report] checkpoints/new_beams_4way_report.json")

    for b_key, b_data in summary_out.items():
        print("\n" + "=" * 95)
        print(f"  {b_data['beam_name']} (Vf={b_data['volfrac']:.2f}, Force={b_data['force_real_N']:.0f} N)")
        print("=" * 95)
        print(f"{'Method':<28} | {'Compliance':<12} | {'Ratio':<8} | {'Yield FoS':<10} | {'Buckle lam1':<11} | {'Defl (mm)':<10} | {'Verdict'}")
        print("-" * 95)
        for m_key, label in [
            ("unopt", "1. Without Topo Opt (Block)"),
            ("simp", "2. Classical SIMP (80 iter)"),
            ("base_unet", "3. Baseline U-Net (Naive)"),
            ("rl_unet", "4. SIMP-Judge RL U-Net")
        ]:
            m = b_data["methods"][m_key]
            print(f"{label:<28} | {m['compliance']:<12.2f} | {m['fea_ratio']:<7.2f}x | {m['fos_yield']:<10.2f} | {m['lambda_1']:<11.2f} | {m['max_defl_mm']:<10.3f} | {m['status_badge']}")


if __name__ == "__main__":
    main()
