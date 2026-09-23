"""
stress_buckling_analyzer.py - Real-World Structural Integrity & In-Plane Buckling Inspector
Performs end-to-end engineering verification:
1. Physical unit scaling (mm, N, MPa) for standard materials (Steel, Aluminum, Titanium, PETG).
2. Linear FEA solving for true physical displacement and maximum deflection.
3. 2D in-plane stress tensor recovery and Von Mises equivalent stress field (MPa).
4. Yielding Factor of Safety (FoS) computation against material yield strength.
5. Generalized sparse eigenvalue solver for 2D in-plane buckling load factor (lambda_1) and mode shape.
6. Generates a 4-panel publication-grade Structural Health & Buckling Certificate.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
import argparse

import dataset_generator as dg
import structural_mechanics as sm
from visualize_data import extract_discrete_loads, draw_domain_and_bcs, draw_loads


def run_structural_inspection(sample_id=0, material="steel", force_N=1500.0,
                              length_mm=120.0, height_mm=60.0, thickness_mm=10.0,
                              output_dir="images", max_iter=80):
    os.makedirs(output_dir, exist_ok=True)
    
    mat_key = material.lower()
    mat = sm.MATERIALS.get(mat_key, sm.MATERIALS["steel"])
    
    print(f"\n============================================================")
    print(f" Structural Integrity & Buckling Inspection: Sample {sample_id:03d}")
    print(f" Material: {mat['name']} (E = {mat['E']/1e3:.1f} GPa, Yield = {mat['yield_strength']:.1f} MPa)")
    print(f" Force: {force_N:.1f} N (~{force_N/9.81:.1f} kg) | Dimensions: {length_mm:.0f}x{height_mm:.0f}x{thickness_mm:.0f} mm")
    print(f"============================================================")
    
    nelx = 120
    nely = 60
    
    # 1. Generate problem and SIMP topology
    print("Running SIMP optimization to generate optimal topology...")
    edofMat, iK, jK, H, Hs = dg.precompute_mesh_and_filter(nelx, nely, 1.5)
    res = dg.sample_problem(
        sample_id=sample_id,
        nelx=nelx,
        nely=nely,
        penal=3.0,
        rmin=1.5,
        max_iter=max_iter,
        global_data=(edofMat, iK, jK, H, Hs),
        return_meta=True
    )
    
    if res is None:
        print("Error: SIMP sample generation failed.")
        return None
        
    input_tensor, density, meta = res
    ch_domain, ch_bc, ch_fx, ch_fy, ch_vf = input_tensor
    volfrac = float(ch_vf[0, 0])
    loads = extract_discrete_loads(ch_fx, ch_fy)
    
    # 2. Run comprehensive physical structural mechanics analysis
    print(f"Executing 2D stress analysis & in-plane eigenvalue buckling solve...")
    results = sm.analyze_structural_integrity(
        density=density,
        fixed_dofs=meta['fixed_dofs'],
        F=meta['F'],
        edofMat=edofMat,
        nelx=nelx,
        nely=nely,
        material_name=mat_key,
        force_real_N=force_N,
        length_mm=length_mm,
        height_mm=height_mm,
        thickness_mm=thickness_mm,
        density_threshold=0.3
    )
    
    # Calculate mass of the optimized structure in grams
    solid_volume_mm3 = float(np.sum(density)) * (length_mm / nelx) * (height_mm / nely) * thickness_mm
    mass_grams = solid_volume_mm3 * mat["density"] * 1e3
    solid_pct = float(np.mean(density)) * 100.0
    
    print("\n--- ANALYSIS SUMMARY ---")
    print(f"• Peak Von Mises Stress:       {results['peak_vm_stress']:.2f} MPa")
    print(f"• 95th Percentile Stress:      {results['vm_95th']:.2f} MPa")
    print(f"• Material Yield Strength:     {results['yield_strength']:.1f} MPa")
    print(f"• Factor of Safety (Yielding): {results['fos_yield']:.2f} -> {results['yield_verdict']}")
    print(f"• Critical Buckling Multiplier:{results['lambda_1']:.2f} -> {results['buckle_verdict']}")
    print(f"• Critical Buckling Load P_cr: {results['p_critical_N']:.1f} N (~{results['p_critical_N']/9.81:.1f} kg)")
    print(f"• Maximum Deflection:          {results['max_deflection_mm']:.4f} mm")
    print(f"• Total Structural Mass:       {mass_grams:.1f} grams ({solid_pct:.1f}% volume fraction)")
    print(f"• Overall Engineering Verdict: {'PASS (SAFE)' if results['overall_pass'] else 'FAIL (UNSAFE)'}")
    
    # --- 3. RENDER 4-PANEL STRUCTURAL REPORT ---
    fig = plt.figure(figsize=(20, 11), facecolor="#F8F9FA")
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.22)
    
    title_banner = (
        f"Structural Integrity & In-Plane Buckling Verification — Sample {sample_id:03d}\n"
        f"Material: {mat['name']} (E={mat['E']/1e3:.0f} GPa, σ_yield={mat['yield_strength']:.0f} MPa) | "
        f"Applied Real Load: {force_N:.0f} N ({force_N/9.81:.1f} kg) | Dimensions: {length_mm:.0f}x{height_mm:.0f}x{thickness_mm:.0f} mm"
    )
    fig.suptitle(title_banner, fontsize=14, fontweight="bold", y=0.98, color="#1A237E")
    
    # ----------------------------------------------------
    # Panel 1: Problem Physics & Optimized Truss Geometry
    # ----------------------------------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    draw_domain_and_bcs(ax0, ch_bc, nelx, nely, is_optimized=True, target=density)
    draw_loads(ax0, loads, arrow_len=18)
    ax0.set_xlim(-20, nelx + 20)
    ax0.set_ylim(nely + 22, -20)
    ax0.set_aspect('equal')
    ax0.set_title("1. Optimized Topology & Applied Boundary Conditions", fontsize=11, fontweight="bold", pad=10)
    ax0.set_xlabel("X (elements)", fontsize=9)
    ax0.set_ylabel("Y (elements)", fontsize=9)
    
    leg_items = [
        mpatches.Patch(facecolor='#D32F2F', edgecolor='#B71C1C', label='Boundary Support (Pin/Roller)'),
        mpatches.Patch(facecolor='#1565C0', edgecolor='#0D47A1', label=f'Applied Load: {force_N:.0f} N'),
        mpatches.Patch(facecolor='#333333', label=f'Solid Structure ({solid_pct:.1f}% Vf)')
    ]
    ax0.legend(handles=leg_items, loc='upper right', fontsize=8.5, framealpha=0.9)
    
    # ----------------------------------------------------
    # Panel 2: Von Mises Equivalent Stress Heatmap (MPa)
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 1])
    vm_grid = results["von_mises_grid"]
    solid_mask = (density >= 0.3)
    
    # Mask out voids so background is clean white
    vm_display = np.where(solid_mask, vm_grid, np.nan)
    vmax = max(results["vm_95th"] * 1.5, 1.0)
    
    im1 = ax1.imshow(vm_display, cmap="turbo", extent=[0, nelx, nely, 0], vmin=0, vmax=vmax, zorder=2)
    ax1.set_facecolor("#FFFFFF")
    domain_border = mpatches.Rectangle((0, 0), nelx, nely, fill=False, edgecolor='#444444', lw=1.5, zorder=5)
    ax1.add_patch(domain_border)
    
    # Annotate peak stress location
    peak_y, peak_x = np.unravel_index(np.argmax(np.nan_to_num(vm_display)), vm_display.shape)
    ax1.plot(peak_x + 0.5, peak_y + 0.5, marker="*", markersize=14, color="#FFEB3B", markeredgecolor="black", zorder=10)
    ax1.text(peak_x + 2, peak_y - 2, f"Peak: {results['peak_vm_stress']:.1f} MPa",
             fontsize=8.5, fontweight="bold", color="black",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF9C4", edgecolor="#FBC02D", lw=1.2), zorder=12)
    
    ax1.set_xlim(-5, nelx + 5)
    ax1.set_ylim(nely + 5, -5)
    ax1.set_aspect('equal')
    ax1.set_title(f"2. Von Mises Stress Field (MPa) [Yield Limit = {mat['yield_strength']:.0f} MPa]",
                  fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("X (elements)", fontsize=9)
    ax1.set_ylabel("Y (elements)", fontsize=9)
    
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.035, pad=0.03)
    cbar1.set_label("Von Mises Stress (MPa)", fontsize=9, fontweight="bold")
    
    # ----------------------------------------------------
    # Panel 3: 2D In-Plane Buckling Mode Shape
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    buckle_data = results["buckling_data"]
    
    if buckle_data and "phi_mag" in buckle_data:
        phi_mag = buckle_data["phi_mag"]
        # Mask out voids to show buckling exclusively in solid struts
        # Rescale phi_mag to element grid (60, 120)
        phi_elem = 0.25 * (
            phi_mag[:-1, :-1] + phi_mag[1:, :-1] +
            phi_mag[:-1, 1:] + phi_mag[1:, 1:]
        )
        phi_display = np.where(solid_mask, phi_elem, np.nan)
        im2 = ax2.imshow(phi_display, cmap="magma", extent=[0, nelx, nely, 0], vmin=0, vmax=1.0, zorder=2)
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.035, pad=0.03)
        cbar2.set_label("Normalized Buckling Amplitude", fontsize=9, fontweight="bold")
    else:
        # Fallback if structure is in pure tension
        ax2.imshow(1.0 - density, cmap="gray", extent=[0, nelx, nely, 0])
        ax2.text(nelx/2, nely/2, "Pure Tension / Stable\nNo Compressive Buckling Possible",
                 ha="center", va="center", fontsize=11, fontweight="bold", color="#2E7D32")
                 
    domain_border2 = mpatches.Rectangle((0, 0), nelx, nely, fill=False, edgecolor='#444444', lw=1.5, zorder=5)
    ax2.add_patch(domain_border2)
    ax2.set_xlim(-5, nelx + 5)
    ax2.set_ylim(nely + 5, -5)
    ax2.set_aspect('equal')
    ax2.set_title(f"3. In-Plane Buckling Mode 1 (Critical Factor λ₁ = {results['lambda_1']:.2f})",
                  fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("X (elements)", fontsize=9)
    ax2.set_ylabel("Y (elements)", fontsize=9)
    
    # ----------------------------------------------------
    # Panel 4: Engineering Health & Safety Dashboard
    # ----------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    
    overall_color = "#2E7D32" if results["overall_pass"] else "#C62828"
    overall_text = "[PASS] STRUCTURALLY SOUND (SAFE)" if results["overall_pass"] else "[FAIL] STRUCTURAL FAILURE RISK"
    
    yield_badge_color = "#2E7D32" if results["yield_status"] == "PASS" else ("#F57F17" if results["yield_status"] == "MARGINAL" else "#C62828")
    buckle_badge_color = "#2E7D32" if results["buckle_status"] == "PASS" else ("#F57F17" if results["buckle_status"] == "MARGINAL" else "#C62828")
    
    # Dashboard text block
    dashboard_card = (
        f"==============================================================\n"
        f"                 STRUCTURAL HEALTH VERDICT                    \n"
        f"==============================================================\n\n"
        f"1. YIELDING RESISTANCE (VON MISES):\n"
        f"   • Material Yield Strength (σ_yield):    {results['yield_strength']:.1f} MPa\n"
        f"   • 95th Percentile Working Stress:       {results['vm_95th']:.1f} MPa\n"
        f"   • Peak Local Hotspot Stress:            {results['peak_vm_stress']:.1f} MPa\n"
        f"   • Factor of Safety (FoS):               {results['fos_yield']:.2f}\n"
        f"   • Status:                               [{results['yield_verdict']}]\n\n"
        f"2. IN-PLANE ELASTIC BUCKLING RESISTANCE:\n"
        f"   • Critical Buckling Multiplier (λ₁):     {results['lambda_1']:.2f}\n"
        f"   • Applied Working Load:                 {force_N:.1f} N\n"
        f"   • Critical Buckling Capacity (P_cr):     {results['p_critical_N']:.1f} N (~{results['p_critical_N']/9.81:.1f} kg)\n"
        f"   • Status:                               [{results['buckle_verdict']}]\n\n"
        f"3. STIFFNESS & WEIGHT SPECIFICATIONS:\n"
        f"   • Maximum Displacement under Load:      {results['max_deflection_mm']:.4f} mm\n"
        f"   • Total Part Mass:                      {mass_grams:.1f} grams\n"
        f"   • Solid Material Fraction:              {solid_pct:.1f}% ({volfrac*100:.1f}% target)\n"
    )
    
    ax3.text(0.02, 0.98, overall_text, transform=ax3.transAxes,
             fontsize=13, fontweight="bold", color=overall_color, va="top")
             
    ax3.text(0.02, 0.88, dashboard_card, transform=ax3.transAxes,
             fontsize=9.2, fontfamily="monospace", va="top",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#ECEFF1", edgecolor="#90A4AE", lw=1.5))
             
    out_path = os.path.join(output_dir, f"structural_report_sample_{sample_id:03d}_{mat_key}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SAVED] Comprehensive Structural Report -> {out_path}")
    return out_path, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structural Integrity & Buckling Inspector")
    parser.add_argument("--sample_id", type=int, default=0, help="Sample ID to inspect")
    parser.add_argument("--material", type=str, default="steel",
                        choices=["steel", "aluminum", "titanium", "petg"],
                        help="Material preset")
    parser.add_argument("--force", type=float, default=1500.0, help="Applied real load in Newtons")
    parser.add_argument("--thickness", type=float, default=10.0, help="Part thickness in mm")
    parser.add_argument("--length", type=float, default=120.0, help="Domain length in mm")
    parser.add_argument("--height", type=float, default=60.0, help="Domain height in mm")
    parser.add_argument("--output_dir", type=str, default="images", help="Output directory")
    args = parser.parse_args()
    
    run_structural_inspection(
        sample_id=args.sample_id,
        material=args.material,
        force_N=args.force,
        length_mm=args.length,
        height_mm=args.height,
        thickness_mm=args.thickness,
        output_dir=args.output_dir
    )
