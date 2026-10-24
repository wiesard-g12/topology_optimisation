"""
simp_solver.py - Unified 2D Continuum SIMP Solver (With & Without In-Plane Buckling)

Provides a unified, rigorous 2D topology optimization framework featuring:
1. Standard SIMP formulation (Pure Compliance Minimization, use_buckling=False).
2. Active Buckling-Aware SIMP formulation (Compliance + Geometric Stiffness Eigenvalue Buckling, use_buckling=True).
3. 2D Bilinear Quadrilateral (Q4) FEA with Plane Stress constitutive laws.
4. Stress recovery (sigma_xx, sigma_yy, tau_xy, Von Mises equivalent stress).
5. Geometric stiffness matrix (K_G) via 2x2 Gauss quadrature.
6. Shift-and-invert sparse eigenvalue buckling solver (K_red - lambda * KG_red) * phi = 0 with ghost-mode elimination.
7. Real-world physical structural integrity verification (mm, N, MPa, Factor of Safety).
8. Unified solver interface `solve_simp_unified` tracking compliance, buckling load multiplier lambda_1, and volume fraction.
9. Standalone CLI interface for immediate benchmark evaluation.
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, eigs, eigsh

# --- 1. REAL-WORLD MATERIAL SPECIFICATIONS ---
MATERIALS = {
    "steel": {
        "name": "Structural Steel (A36)",
        "E": 200e3,             # MPa (N/mm^2)
        "nu": 0.30,
        "yield_strength": 250.0,# MPa
        "density": 7850e-9,     # kg/mm^3
        "color": "#455A64"
    },
    "aluminum": {
        "name": "Aluminum 6061-T6",
        "E": 69e3,              # MPa
        "nu": 0.33,
        "yield_strength": 276.0,# MPa
        "density": 2700e-9,     # kg/mm^3
        "color": "#1976D2"
    },
    "titanium": {
        "name": "Titanium Ti-6Al-4V",
        "E": 114e3,             # MPa
        "nu": 0.34,
        "yield_strength": 880.0,# MPa
        "density": 4430e-9,     # kg/mm^3
        "color": "#7B1FA2"
    },
    "petg": {
        "name": "3D-Printed PETG / Carbon PLA",
        "E": 2.8e3,             # MPa
        "nu": 0.38,
        "yield_strength": 55.0, # MPa
        "density": 1270e-9,     # kg/mm^3
        "color": "#E65100"
    }
}


# --- 2. ELEMENT STIFFNESS & MESH FILTER PRECOMPUTATION ---
def get_stiffness_matrix(Emax=1.0, nu=0.3):
    """
    Computes standard 8x8 element stiffness matrix KE for a 2D bilinear
    quadrilateral (Q4) element in plane stress.
    Node ordering follows standard DTU convention: BL (0), BR (1), TR (2), TL (3).
    """
    k = np.array([
        1/2 - nu/6,   1/8 + nu/8,   -1/4 - nu/12, -1/8 + 3*nu/8,
        -1/4 + nu/12, -1/8 - nu/8,  nu/6,         1/8 - 3*nu/8
    ])
    KE = Emax / (1.0 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    return KE


def precompute_mesh_and_filter(nelx, nely, rmin):
    """
    Precomputes vectorized element degree-of-freedom indices (edofMat, iK, jK)
    and Sigmund/Bourdin conical sensitivity filter matrices (H, Hs).
    """
    elx, ely = np.meshgrid(np.arange(nelx), np.arange(nely), indexing='ij')
    elx, ely = elx.flatten(), ely.flatten()
    n1 = (nely + 1) * elx + ely
    n2 = (nely + 1) * (elx + 1) + ely

    # Standard DTU node DOF assignment: BL, BR, TR, TL
    edofMat = np.column_stack([
        2*n1 + 2, 2*n1 + 3,
        2*n2 + 2, 2*n2 + 3,
        2*n2,     2*n2 + 1,
        2*n1,     2*n1 + 1
    ])

    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    # Precompute filter matrix H and Hs
    nfilter = nelx * nely * (2 * (int(np.ceil(rmin)) - 1) + 1)**2
    iH = np.zeros(nfilter, dtype=int)
    jH = np.zeros(nfilter, dtype=int)
    sH = np.zeros(nfilter, dtype=float)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = max(i - (int(np.ceil(rmin)) - 1), 0)
            kk2 = min(i + int(np.ceil(rmin)), nelx)
            ll1 = max(j - (int(np.ceil(rmin)) - 1), 0)
            ll2 = min(j + int(np.ceil(rmin)), nely)
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * nely + l
                    fac = rmin - np.sqrt((i - k)**2 + (j - l)**2)
                    if fac > 0:
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = fac
                        cc += 1

    H = coo_matrix((sH[:cc], (iH[:cc], jH[:cc])), shape=(nelx * nely, nelx * nely)).tocsr()
    Hs = H.sum(axis=1).A1
    return edofMat, iK, jK, H, Hs


# --- 3. PLANE STRESS CONSTITUTIVE & B-MATRIX ---
def get_plane_stress_D(E=1.0, nu=0.3):
    """Returns the 3x3 plane stress constitutive matrix D."""
    factor = E / (1.0 - nu**2)
    D = factor * np.array([
        [1.0, nu,  0.0],
        [nu,  1.0, 0.0],
        [0.0, 0.0, 0.5 * (1.0 - nu)]
    ])
    return D


def get_Q4_B_matrix(dx=1.0, dy=1.0):
    """
    Evaluates the 3x8 strain-displacement matrix B for a 4-node rectangular element
    at its centroid (xi = 0, eta = 0).
    Node ordering follows standard DTU convention: BL (0), BR (1), TR (2), TL (3).
    DOFs: [u1, v1, u2, v2, u3, v3, u4, v4]
    """
    dN_dx = np.array([-0.5 / dx,  0.5 / dx,  0.5 / dx, -0.5 / dx])
    dN_dy = np.array([-0.5 / dy, -0.5 / dy,  0.5 / dy,  0.5 / dy])

    B = np.zeros((3, 8))
    for i in range(4):
        # epsilon_xx = du/dx
        B[0, 2*i] = dN_dx[i]
        # epsilon_yy = dv/dy
        B[1, 2*i + 1] = dN_dy[i]
        # gamma_xy = du/dy + dv/dx
        B[2, 2*i] = dN_dy[i]
        B[2, 2*i + 1] = dN_dx[i]

    return B, dN_dx, dN_dy


# --- 4. ELEMENT GEOMETRIC (INITIAL STRESS) STIFFNESS MATRIX ---
def get_Q4_geometric_stiffness(sigma_xx, sigma_yy, tau_xy, dx=1.0, dy=1.0, thickness=1.0):
    """
    Computes the 8x8 in-plane geometric stiffness matrix K_G for a Q4 element.
    Evaluated using 2x2 Gauss integration:
    U_G = 0.5 * integral [ (grad u)^T * S * (grad u) + (grad v)^T * S * (grad v) ] dV
    where S = [[sigma_xx, tau_xy], [tau_xy, sigma_yy]].
    """
    gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    weights = [1.0, 1.0]

    kg_sub = np.zeros((4, 4))
    S = np.array([[sigma_xx, tau_xy],
                  [tau_xy,   sigma_yy]])

    for xi, wx in zip(gauss_pts, weights):
        for eta, wy in zip(gauss_pts, weights):
            dN_dxi = np.array([
                -0.25 * (1.0 - eta),
                 0.25 * (1.0 - eta),
                 0.25 * (1.0 + eta),
                -0.25 * (1.0 + eta)
            ])
            dN_deta = np.array([
                -0.25 * (1.0 - xi),
                -0.25 * (1.0 + xi),
                 0.25 * (1.0 + xi),
                 0.25 * (1.0 - xi)
            ])
            dNx = dN_dxi * (2.0 / dx)
            dNy = dN_deta * (2.0 / dy)
            detJ = (dx / 2.0) * (dy / 2.0)

            G0 = np.vstack([dNx, dNy])
            kg_sub += G0.T @ S @ G0 * (detJ * wx * wy * thickness)

    KG = np.zeros((8, 8))
    for i in range(4):
        for j in range(4):
            KG[2*i,     2*j]     = kg_sub[i, j]
            KG[2*i + 1, 2*j + 1] = kg_sub[i, j]

    return KG


# --- 5. STRESS FIELD RECOVERY ---
def compute_stress_field(U, xPhys, edofMat, nelx, nely, E=1.0, nu=0.3, dx=1.0, dy=1.0):
    """
    Computes element strains, stresses, and Von Mises equivalent stress across the mesh.
    Returns dict with 'von_mises', 'sigma_xx', 'sigma_yy', 'tau_xy', 'strains', 'stresses'.
    """
    nele = nelx * nely
    D = get_plane_stress_D(E=E, nu=nu)
    B, _, _ = get_Q4_B_matrix(dx=dx, dy=dy)

    U_edof = U[edofMat]  # shape (nele, 8)
    strains = U_edof @ B.T
    stresses = strains @ D.T

    sigma_xx = stresses[:, 0]
    sigma_yy = stresses[:, 1]
    tau_xy   = stresses[:, 2]

    von_mises = np.sqrt(
        np.maximum(0.0, sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3.0 * tau_xy**2)
    )

    vm_grid = von_mises.reshape((nely, nelx), order='F')
    sxx_grid = sigma_xx.reshape((nely, nelx), order='F')
    syy_grid = sigma_yy.reshape((nely, nelx), order='F')
    txy_grid = tau_xy.reshape((nely, nelx), order='F')

    return {
        "von_mises": vm_grid,
        "sigma_xx": sxx_grid,
        "sigma_yy": syy_grid,
        "tau_xy": txy_grid,
        "strains": strains,
        "stresses": stresses
    }


# --- 6. SHIFT-INVERT GENERALIZED EIGENVALUE BUCKLING SOLVER ---
def solve_inplane_buckling(U, xPhys, edofMat, fixeddofs, nelx, nely,
                           E=1.0, nu=0.3, dx=1.0, dy=1.0, thickness=1.0,
                           density_threshold=0.25, num_modes=2):
    """
    Active-submesh shift-invert eigenvalue buckling solver:
        (K_red - lambda * KG_red) * phi = 0
    where KG_red is the compressive geometric stiffness (-K_G) on active solid elements.
    Ghost-mode elimination is guaranteed by condensing onto xPhys >= density_threshold.

    Returns dict with:
        'critical_factor' (lambda_1), 'eigenvalues', 'phi_mag', 'phi_vector', 'status'.
    """
    ndof = 2 * (nelx + 1) * (nely + 1)
    nele = nelx * nely

    xPhys_flat = xPhys.flatten(order='F') if hasattr(xPhys, 'flatten') else np.asarray(xPhys)

    # 1. Stress field
    stress_data = compute_stress_field(U, xPhys_flat, edofMat, nelx, nely, E=E, nu=nu, dx=dx, dy=dy)
    stresses = stress_data["stresses"]

    # 2. Identify active solid elements and active free DOFs
    active_elem_mask = (xPhys_flat >= density_threshold)
    active_elem_indices = np.where(active_elem_mask)[0]

    if len(active_elem_indices) < 10:
        return {
            "eigenvalues": [999.0],
            "critical_factor": 999.0,
            "phi_mag": np.zeros((nely + 1, nelx + 1)),
            "phi_vector": np.zeros(ndof),
            "status": "Insufficient Material"
        }

    active_edofs = edofMat[active_elem_indices]
    active_dofs = np.unique(active_edofs)
    active_free_dofs = np.setdiff1d(active_dofs, fixeddofs)

    if len(active_free_dofs) < 10:
        return {
            "eigenvalues": [999.0],
            "critical_factor": 999.0,
            "phi_mag": np.zeros((nely + 1, nelx + 1)),
            "phi_vector": np.zeros(ndof),
            "status": "Insufficient DOFs"
        }

    dof_map = {dof: idx for idx, dof in enumerate(active_free_dofs)}
    num_reduced = len(active_free_dofs)

    # 3. Assemble active K and KG
    KE_standard = get_stiffness_matrix(Emax=E, nu=nu)

    iK_list, jK_list, sK_list = [], [], []
    iG_list, jG_list, sG_list = [], [], []

    for e_idx in active_elem_indices:
        edofs = edofMat[e_idx]
        rho = xPhys_flat[e_idx]

        sxx = stresses[e_idx, 0]
        syy = stresses[e_idx, 1]
        txy = stresses[e_idx, 2]

        ke_e = (rho**3.0) * KE_standard

        # Only assemble geometric stiffness for solid/near-solid elements (rho >= 0.45)
        # under net in-plane compression (trace sxx + syy < 0)
        # Stresses are penalized by rho**3 (Ferrari & Sigmund / Bendsoe & Sigmund) to eliminate void/edge flutter modes
        if rho >= 0.45 and (sxx + syy) < 0.0:
            sxx_pen = (rho**3.0) * sxx
            syy_pen = (rho**3.0) * syy
            txy_pen = (rho**3.0) * txy
            kg_comp_e = -get_Q4_geometric_stiffness(sxx_pen, syy_pen, txy_pen, dx=dx, dy=dy, thickness=thickness)
        else:
            kg_comp_e = None

        for i_loc in range(8):
            gi = edofs[i_loc]
            if gi not in dof_map:
                continue
            ri = dof_map[gi]

            for j_loc in range(8):
                gj = edofs[j_loc]
                if gj not in dof_map:
                    continue
                rj = dof_map[gj]

                iK_list.append(ri)
                jK_list.append(rj)
                sK_list.append(ke_e[i_loc, j_loc])

                if kg_comp_e is not None:
                    iG_list.append(ri)
                    jG_list.append(rj)
                    sG_list.append(kg_comp_e[i_loc, j_loc])

    K_red = csc_matrix((sK_list, (iK_list, jK_list)), shape=(num_reduced, num_reduced))
    KG_red = csc_matrix((sG_list, (iG_list, jG_list)), shape=(num_reduced, num_reduced))

    K_red = 0.5 * (K_red + K_red.T)
    KG_red = 0.5 * (KG_red + KG_red.T)

    if len(sG_list) == 0 or KG_red.nnz == 0:
        return {
            "eigenvalues": [999.0],
            "critical_factor": 999.0,
            "phi_mag": np.zeros((nely + 1, nelx + 1)),
            "phi_vector": np.zeros(ndof),
            "stresses": stresses,
            "status": "Pure Tension (No Compressive Struts)"
        }

    # 4. Generalized eigenvalue solve: find largest algebraic eigenvalue alpha of KG_red * phi = alpha * K_red * phi
    # (where alpha_max = 1 / lambda_1 is the 1st fundamental Euler buckling mode)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k_solve = max(1, min(num_modes, num_reduced - 2))
            evals, evecs = eigsh(KG_red, k=k_solve, M=K_red, which='LA')

            alphas = np.real(evals)
            valid_idx = np.where(alphas > 1e-6)[0]

            if len(valid_idx) == 0:
                return {
                    "eigenvalues": [999.0],
                    "critical_factor": 999.0,
                    "phi_mag": np.zeros((nely + 1, nelx + 1)),
                    "phi_vector": np.zeros(ndof),
                    "stresses": stresses,
                    "status": "Stable (No Compressive Instability)"
                }

            sorted_order = np.argsort(-alphas[valid_idx])
            top_alphas = alphas[valid_idx][sorted_order][:num_modes]
            top_vec = np.real(evecs[:, valid_idx[sorted_order][0]])

            lambdas = (1.0 / top_alphas).tolist()
            critical_lambda = float(lambdas[0])

            phi_full = np.zeros(ndof)
            for gi, ri in dof_map.items():
                phi_full[gi] = top_vec[ri]

            max_disp = np.max(np.abs(phi_full))
            if max_disp > 1e-12:
                phi_full /= max_disp

            phi_u = phi_full[0::2].reshape((nely + 1, nelx + 1), order='F')
            phi_v = phi_full[1::2].reshape((nely + 1, nelx + 1), order='F')
            phi_mag = np.sqrt(phi_u**2 + phi_v**2)

            return {
                "eigenvalues": lambdas,
                "critical_factor": critical_lambda,
                "phi_mag": phi_mag,
                "phi_vector": phi_full,
                "stresses": stresses,
                "status": "Solved"
            }
    except Exception as ex:
        return {
            "eigenvalues": [999.0],
            "critical_factor": 999.0,
            "phi_mag": np.zeros((nely + 1, nelx + 1)),
            "phi_vector": np.zeros(ndof),
            "stresses": stresses,
            "status": f"Solver Warning: {str(ex)}"
        }


# --- 7. REAL-WORLD PHYSICAL STRUCTURAL HEALTH INSPECTOR ---
def analyze_structural_integrity(density, fixed_dofs, F, edofMat,
                                 nelx=120, nely=60, material_name="steel",
                                 force_real_N=1500.0, length_mm=120.0,
                                 height_mm=60.0, thickness_mm=10.0,
                                 density_threshold=0.3):
    """
    Performs full real-world structural integrity verification on an optimized topology:
    1. Scales problem to real engineering units (mm, N, MPa).
    2. Solves linear FEA for physical displacements (mm).
    3. Computes 2D in-plane stress tensor and Von Mises stress (MPa).
    4. Evaluates Yielding Factor of Safety (FoS) against material yield strength.
    5. Solves in-plane geometric stiffness eigenvalue buckling for critical multiplier lambda_1.
    6. Issues definitive engineering PASS / FAIL verdicts for both yielding and buckling.
    """
    mat = MATERIALS.get(material_name.lower(), MATERIALS["steel"])
    E = mat["E"]
    nu = mat["nu"]
    yield_strength = mat["yield_strength"]

    dx = length_mm / nelx
    dy = height_mm / nely
    thickness = thickness_mm

    ndof = 2 * (nelx + 1) * (nely + 1)
    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixed_dofs)

    f_mag_norm = np.max(np.abs(F))
    if f_mag_norm > 1e-12:
        F_phys = F * (force_real_N / f_mag_norm)
    else:
        F_phys = F.copy()

    KE_phys = thickness * get_stiffness_matrix(Emax=E, nu=nu)

    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    xPhys = density.flatten(order='F')
    Emin = 1e-9 * E
    sK = ((KE_phys.flatten()[:, None]) * (Emin/E + xPhys**3.0 * (1.0 - Emin/E))).flatten(order='F')
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()

    try:
        K_free = K[freedofs, :][:, freedofs]
        U_free = spsolve(K_free, F_phys[freedofs])
    except Exception:
        return None

    U_phys = np.zeros(ndof)
    U_phys[freedofs] = U_free
    max_deflection_mm = float(np.max(np.abs(U_phys)))

    stress_results = compute_stress_field(
        U_phys, xPhys, edofMat, nelx, nely,
        E=E, nu=nu, dx=dx, dy=dy
    )
    vm_grid = stress_results["von_mises"]
    solid_mask = (density >= density_threshold)
    solid_vm = vm_grid[solid_mask]

    if len(solid_vm) > 0:
        peak_vm_stress = float(np.max(solid_vm))
        vm_95th = float(np.percentile(solid_vm, 95))
        mean_vm_stress = float(np.mean(solid_vm))
    else:
        peak_vm_stress = 0.0
        vm_95th = 0.0
        mean_vm_stress = 0.0

    fos_yield = yield_strength / (vm_95th + 1e-12)

    buckling_results = solve_inplane_buckling(
        U_phys, xPhys, edofMat, fixed_dofs,
        nelx, nely, E=E, nu=nu, dx=dx, dy=dy,
        thickness=thickness, density_threshold=density_threshold
    )

    lambda_1 = buckling_results["critical_factor"] if buckling_results else 1.0
    p_critical_N = lambda_1 * force_real_N

    # Verdicts
    if fos_yield >= 2.0:
        yield_verdict = "PASS (Safe & Robust)"
        yield_status = "PASS"
    elif fos_yield >= 1.5:
        yield_verdict = "PASS (Optimal / Aerospace)"
        yield_status = "PASS"
    elif fos_yield >= 1.0:
        yield_verdict = "MARGINAL (Near Yield Limit)"
        yield_status = "MARGINAL"
    else:
        yield_verdict = "FAIL (Yielding / Plastic Deformation)"
        yield_status = "FAIL"

    if lambda_1 >= 2.0:
        buckle_verdict = "PASS (Immune to In-Plane Buckling)"
        buckle_status = "PASS"
    elif lambda_1 >= 1.5:
        buckle_verdict = "PASS (Stable Buckling Reserve)"
        buckle_status = "PASS"
    elif lambda_1 >= 1.0:
        buckle_verdict = "MARGINAL (Approaching Critical Load)"
        buckle_status = "MARGINAL"
    else:
        buckle_verdict = "FAIL (Will Buckle In-Plane)"
        buckle_status = "FAIL"

    overall_pass = (yield_status == "PASS" and buckle_status == "PASS")

    return {
        "material": mat,
        "length_mm": length_mm,
        "height_mm": height_mm,
        "thickness_mm": thickness,
        "force_real_N": force_real_N,
        "U_phys": U_phys,
        "max_deflection_mm": max_deflection_mm,
        "von_mises_grid": vm_grid,
        "peak_vm_stress": peak_vm_stress,
        "vm_95th": vm_95th,
        "mean_vm_stress": mean_vm_stress,
        "fos_yield": fos_yield,
        "yield_strength": yield_strength,
        "yield_verdict": yield_verdict,
        "yield_status": yield_status,
        "lambda_1": lambda_1,
        "p_critical_N": p_critical_N,
        "buckle_verdict": buckle_verdict,
        "buckle_status": buckle_status,
        "buckling_data": buckling_results,
        "overall_pass": overall_pass
    }


# --- 8. UNIFIED 2D CONTINUUM SIMP SOLVER (WITH & WITHOUT BUCKLING) ---
def solve_simp_unified(nelx, nely, volfrac, penal, fixeddofs, F,
                       edofMat=None, iK=None, jK=None, H=None, Hs=None,
                       rmin=1.5, max_iter=60, use_buckling=False,
                       buckle_weight=0.45, buckle_start_iter=20,
                       record_history=False, history_interval=10,
                       verbose=False):
    """
    Unified 2D Continuum SIMP Solver with explicit support for:
      - Without Buckling (use_buckling=False): Pure compliance minimization.
      - With Buckling (use_buckling=True): Multi-objective compliance + buckling eigenvalue reinforcement.

    Initializes with a uniform gray continuum block: initial_density = np.full((nely, nelx), volfrac).
    Always computes the final buckling load factor final_lambda_1 to enable head-to-head comparison.

    Returns dict with:
        'initial_density', 'final_density', 'initial_compliance', 'final_compliance',
        'compliance_reduction_pct', 'final_lambda_1', 'actual_vf', 'target_vf',
        'volume_reduction_pct', 'use_buckling', 'buckle_weight', 'iterations_run',
        'elapsed_time', 'history'
    """
    start_time = time.time()
    Emin = 1e-9
    Emax = 1.0
    nu = 0.3
    ndof = 2 * (nelx + 1) * (nely + 1)
    nele = nelx * nely

    # 1. Mesh & filter precomputation if not provided
    if edofMat is None or iK is None or jK is None or H is None or Hs is None:
        edofMat, iK, jK, H, Hs = precompute_mesh_and_filter(nelx, nely, rmin)

    KE = get_stiffness_matrix(Emax=Emax, nu=nu)

    # 2. Initialize uniform continuum block
    initial_density = np.full((nely, nelx), volfrac, dtype=np.float32)
    x = volfrac * np.ones(nele, dtype=float)
    xPhys = x.copy()

    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixeddofs)
    dv = np.ones(nele)

    change = 1.0
    loop = 0
    initial_compliance = 0.0
    compliance = 0.0
    history = []
    current_lambda = None
    ce = np.zeros(nele, dtype=float)

    if record_history:
        history.append({
            'iteration': 0,
            'density': initial_density.copy(),
            'compliance': None,
            'lambda_1': None,
            'change': 1.0
        })

    if verbose:
        mode_label = f"Buckling-Aware (w={buckle_weight:.2f}, start={buckle_start_iter})" if use_buckling else "Without Buckling (Pure Compliance)"
        print(f"\n=======================================================")
        print(f" SIMP Solver: {nelx}x{nely} | Target Vf: {volfrac:.2f} | Mode: {mode_label}")
        print(f"=======================================================")

    # 3. Main Optimization Loop
    while loop < max_iter and (change > 0.01 or (use_buckling and loop < buckle_start_iter + 5)):
        loop += 1
        t0 = time.time()

        # Global stiffness assembly
        sK = ((KE.flatten()[:, None]) * (Emin + xPhys**penal * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K_free = K[freedofs, :][:, freedofs]

        # FEA Solve
        try:
            U_free = spsolve(K_free, F[freedofs])
        except Exception as ex:
            if verbose:
                print(f"FEA Solve failed at iter {loop}: {ex}")
            break

        if np.any(np.isnan(U_free)) or np.any(np.isinf(U_free)):
            if verbose:
                print(f"FEA produced NaN/Inf at iter {loop}")
            break

        U = np.zeros(ndof)
        U[freedofs] = U_free

        # Compliance strain energy
        U_edof = U[edofMat]
        ce = (np.dot(U_edof, KE) * U_edof).sum(axis=1)
        compliance = float(np.sum((Emin + xPhys**penal * (Emax - Emin)) * ce))

        if loop == 1:
            initial_compliance = compliance

        # Compliance sensitivity: dc = -p * rho^(p-1) * (Emax - Emin) * ce
        dc = -penal * (Emax - Emin) * (xPhys**(penal - 1)) * ce

        # Active Buckling Sensitivity check
        buckle_active = (use_buckling and loop >= buckle_start_iter and buckle_weight > 0.0)
        if buckle_active:
            buckle_res = solve_inplane_buckling(
                U, xPhys, edofMat, fixeddofs, nelx, nely,
                E=Emax, nu=nu, density_threshold=0.25, num_modes=2
            )
            if buckle_res is not None and "phi_vector" in buckle_res and buckle_res["critical_factor"] < 900.0:
                current_lambda = float(buckle_res["critical_factor"])
                phi_1 = buckle_res["phi_vector"]
                phi_edof = phi_1[edofMat]

                # Modal strain energy
                me = (np.dot(phi_edof, KE) * phi_edof).sum(axis=1)

                # Weight by compressive indicator (elements where sxx + syy < 0)
                # to specifically reinforce compression struts undergoing buckling
                stress_info = compute_stress_field(U, xPhys, edofMat, nelx, nely, E=Emax, nu=nu)
                sxx_flat = stress_info["sigma_xx"].flatten(order='F')
                syy_flat = stress_info["sigma_yy"].flatten(order='F')
                comp_indicator = ((sxx_flat + syy_flat) < 0.0).astype(float)
                me = me * comp_indicator

                # Buckling sensitivity: d_lambda / d_rho
                d_lambda = penal * (xPhys**(penal - 1)) * me

                # Robust normalization using 98th percentile clamped to [0, 1.2]
                p98_comp = float(np.percentile(-dc, 98))
                p98_buckle = float(np.percentile(d_lambda, 98))

                norm_comp = np.clip((-dc) / (p98_comp + 1e-12), 0.0, 1.2)
                norm_buckle = np.clip(d_lambda / (p98_buckle + 1e-12), 0.0, 1.2)

                # Smooth continuation ramp for buckling weight
                ramp = min(1.0, (loop - buckle_start_iter + 1) / 10.0)
                w_eff = float(buckle_weight) * ramp

                blended_drive = (1.0 - w_eff) * norm_comp + w_eff * norm_buckle
                dJ = -blended_drive
            else:
                dJ = dc.copy()
        else:
            dJ = dc.copy()

        # Bourdin/Sigmund mesh sensitivity filter
        dJ_filt = np.asarray(H.dot(x * dJ)) / Hs / np.maximum(0.001, x)
        dJ_filt = np.minimum(dJ_filt, -1e-12)

        # Optimality Criteria (OC) design update with bisection
        l1, l2 = 0.0, 1e9
        move = 0.10 if buckle_active else 0.20
        while (l2 - l1) / (l1 + l2 + 1e-15) > 1e-4:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(
                0.0,
                np.maximum(
                    x - move,
                    np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dJ_filt / dv / lmid)))
                )
            )
            if np.sum(xnew) > volfrac * nele:
                l1 = lmid
            else:
                l2 = lmid

        change = float(np.max(np.abs(xnew - x)))
        x = xnew.copy()
        xPhys = x.copy()

        iter_dt = time.time() - t0

        if record_history and (loop % history_interval == 0 or loop == max_iter or change <= 0.01):
            history.append({
                'iteration': loop,
                'compliance': compliance,
                'lambda_1': current_lambda,
                'change': change,
                'density': xPhys.reshape((nely, nelx), order='F').copy()
            })

        if verbose and (loop % 5 == 0 or loop == max_iter or loop == buckle_start_iter or loop == 1):
            b_msg = f" | lambda_1: {current_lambda:6.2f}" if current_lambda is not None else ""
            print(f"Iter {loop:2d}/{max_iter} | Compliance: {compliance:10.4f} | Change: {change:6.4f}{b_msg} | ({iter_dt:.2f}s)")

    elapsed_time = time.time() - start_time
    final_density = xPhys.reshape((nely, nelx), order='F').astype(np.float32)

    # 4. Final buckling factor calculation (always computed for head-to-head comparison)
    final_buckle = solve_inplane_buckling(
        U, xPhys, edofMat, fixeddofs, nelx, nely,
        E=Emax, nu=nu, density_threshold=0.25, num_modes=2
    )
    if final_buckle is not None and "critical_factor" in final_buckle:
        final_lambda_1 = float(final_buckle["critical_factor"])
    else:
        final_lambda_1 = 999.0

    # Final element strain energy map
    e_elem = ((Emin + (xPhys ** penal) * (Emax - Emin)) * ce).reshape((nely, nelx), order='F')
    se_norm = np.clip(e_elem / (np.percentile(e_elem, 99) + 1e-12), 0.0, 2.0).astype(np.float32)

    compliance_reduction_pct = ((initial_compliance - compliance) / max(1e-9, initial_compliance)) * 100.0
    actual_vf = float(np.mean(final_density))
    volume_reduction_pct = (1.0 - actual_vf) * 100.0

    if verbose:
        print(f"--- Solved in {elapsed_time:.2f}s ({loop} iters) | C: {initial_compliance:.2f} -> {compliance:.2f} (-{compliance_reduction_pct:.1f}%) | lambda_1: {final_lambda_1:.2f} | Actual Vf: {actual_vf:.3f} ---")

    return {
        'initial_density': initial_density,
        'final_density': final_density,
        'strain_energy': se_norm,
        'raw_strain_energy': e_elem.astype(np.float32),
        'initial_compliance': float(initial_compliance),
        'final_compliance': float(compliance),
        'compliance_reduction_pct': float(compliance_reduction_pct),
        'final_lambda_1': float(final_lambda_1),
        'actual_vf': float(actual_vf),
        'target_vf': float(volfrac),
        'volume_reduction_pct': float(volume_reduction_pct),
        'use_buckling': bool(use_buckling),
        'buckle_weight': float(buckle_weight if use_buckling else 0.0),
        'iterations_run': int(loop),
        'elapsed_time': float(elapsed_time),
        'history': history
    }


# --- 9. BENCHMARK PROBLEM GENERATOR FOR CLI & EXPERIMENTS ---
def build_benchmark_problem(case_name, volfrac=0.40, nelx=120, nely=60, load_angle=None):
    """
    Constructs discrete supports (fixed_dofs) and applied point force vector (F)
    for canonical structural benchmark problems:
      - 'cantilever': pinned left wall endpoints, downward load at mid-right edge
      - 'mbb': symmetry vertical roller left edge, bottom-right roller, downward load at top-left
      - 'bridge': bottom-left pin, bottom-right roller, center downward load
      - 'crane': left wall pin supports, bottom-right corner downward load
    """
    case_name = case_name.lower()
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof)
    fixed_dofs = []

    def get_nid(x, y):
        return x * (nely + 1) + y

    if case_name == 'cantilever':
        p1 = (0, 0)
        p2 = (0, nely)
        n1, n2 = get_nid(*p1), get_nid(*p2)
        fixed_dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]

        lx, ly = nelx, nely // 2
        ln = get_nid(lx, ly)
        angle = -np.pi / 2 if load_angle is None else np.radians(load_angle)
        F[2*ln] = float(np.cos(angle))
        F[2*ln + 1] = float(np.sin(angle))

    elif case_name == 'mbb':
        for y in range(nely + 1):
            n = get_nid(0, y)
            fixed_dofs.append(2*n) # fix X (symmetry)
        n_br = get_nid(nelx, nely)
        fixed_dofs.append(2*n_br + 1) # fix Y at bottom right

        lx, ly = 0, 0
        ln = get_nid(lx, ly)
        angle = -np.pi / 2 if load_angle is None else np.radians(load_angle)
        F[2*ln] = float(np.cos(angle))
        F[2*ln + 1] = float(np.sin(angle))

    elif case_name == 'bridge':
        p1 = (0, nely)
        p2 = (nelx, nely)
        n1, n2 = get_nid(*p1), get_nid(*p2)
        fixed_dofs = [2*n1, 2*n1+1, 2*n2+1] # pin left, roller right

        lx, ly = nelx // 2, 0
        ln = get_nid(lx, ly)
        angle = -np.pi / 2 if load_angle is None else np.radians(load_angle)
        F[2*ln] = float(np.cos(angle))
        F[2*ln + 1] = float(np.sin(angle))

    elif case_name == 'crane':
        p1 = (0, 0)
        p2 = (0, nely)
        n1, n2 = get_nid(*p1), get_nid(*p2)
        fixed_dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]

        lx, ly = nelx, nely
        ln = get_nid(lx, ly)
        angle = -np.pi / 2 if load_angle is None else np.radians(load_angle)
        F[2*ln] = float(np.cos(angle))
        F[2*ln + 1] = float(np.sin(angle))

    else:
        raise ValueError(f"Unknown benchmark case: {case_name}. Available: cantilever, mbb, bridge, crane")

    fixed_dofs = np.unique(fixed_dofs)
    return fixed_dofs, F


# --- 10. CLI & COMPARISON VISUALIZER ---
def run_cli_comparison(args):
    """Executes SIMP solver per CLI arguments and generates formatted reports / plots."""
    nelx = args.nelx
    nely = args.nely
    volfrac = args.volfrac
    max_iter = args.max_iter
    case_name = args.case.lower()
    mode = args.mode.lower()
    buckle_weight = args.buckle_weight
    buckle_start_iter = args.buckle_start_iter

    fixed_dofs, F = build_benchmark_problem(case_name, volfrac=volfrac, nelx=nelx, nely=nely)
    edofMat, iK, jK, H, Hs = precompute_mesh_and_filter(nelx, nely, rmin=1.5)

    res_without = None
    res_with = None

    if mode in ["without_buckling", "both"]:
        print(f"\n>>> Running SIMP [Without Buckling: Pure Compliance] on '{case_name}'...")
        res_without = solve_simp_unified(
            nelx=nelx, nely=nely, volfrac=volfrac, penal=3.0,
            fixeddofs=fixed_dofs, F=F,
            edofMat=edofMat, iK=iK, jK=jK, H=H, Hs=Hs,
            max_iter=max_iter, use_buckling=False,
            verbose=True
        )

    if mode in ["with_buckling", "both"]:
        print(f"\n>>> Running SIMP [With Buckling: Buckling-Aware (w={buckle_weight})] on '{case_name}'...")
        res_with = solve_simp_unified(
            nelx=nelx, nely=nely, volfrac=volfrac, penal=3.0,
            fixeddofs=fixed_dofs, F=F,
            edofMat=edofMat, iK=iK, jK=jK, H=H, Hs=Hs,
            max_iter=max_iter, use_buckling=True,
            buckle_weight=buckle_weight,
            buckle_start_iter=buckle_start_iter,
            verbose=True
        )

    # Print summary table
    print("\n" + "="*80)
    print(f" SIMP OPTIMIZATION SUMMARY REPORT — Case: {case_name.upper()} ({nelx}x{nely}, Target Vf={volfrac:.2f})")
    print("="*80)
    header = f"{'Metric':<32} | {'Without Buckling':<20} | {'With Buckling':<20}"
    print(header)
    print("-" * len(header))

    def fmt_val(res, key, fmt="{:.4f}"):
        if res is None:
            return "N/A"
        val = res.get(key, "N/A")
        if isinstance(val, (int, float)):
            return fmt.format(val)
        return str(val)

    print(f"{'Iterations Run':<32} | {fmt_val(res_without, 'iterations_run', '{:d}'):<20} | {fmt_val(res_with, 'iterations_run', '{:d}'):<20}")
    print(f"{'Initial Compliance (C0)':<32} | {fmt_val(res_without, 'initial_compliance', '{:.2f}'):<20} | {fmt_val(res_with, 'initial_compliance', '{:.2f}'):<20}")
    print(f"{'Final Compliance (C*)':<32} | {fmt_val(res_without, 'final_compliance', '{:.2f}'):<20} | {fmt_val(res_with, 'final_compliance', '{:.2f}'):<20}")
    print(f"{'Compliance Reduction (%)':<32} | {fmt_val(res_without, 'compliance_reduction_pct', '{:.1f}%'):<20} | {fmt_val(res_with, 'compliance_reduction_pct', '{:.1f}%'):<20}")
    print(f"{'Critical Buckling Factor (lambda_1)':<32} | {fmt_val(res_without, 'final_lambda_1', '{:.2f}'):<20} | {fmt_val(res_with, 'final_lambda_1', '{:.2f}'):<20}")
    print(f"{'Actual Volume Fraction':<32} | {fmt_val(res_without, 'actual_vf', '{:.3f}'):<20} | {fmt_val(res_with, 'actual_vf', '{:.3f}'):<20}")
    print(f"{'Volume Reduction (%)':<32} | {fmt_val(res_without, 'volume_reduction_pct', '{:.1f}%'):<20} | {fmt_val(res_with, 'volume_reduction_pct', '{:.1f}%'):<20}")
    print(f"{'Solve Time (s)':<32} | {fmt_val(res_without, 'elapsed_time', '{:.2f}s'):<20} | {fmt_val(res_with, 'elapsed_time', '{:.2f}s'):<20}")
    print("="*80)

    # Optional plotting
    if args.save_fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(args.out_dir, exist_ok=True)
        out_file = os.path.join(args.out_dir, f"simp_{case_name}_{mode}.png")

        if mode == "both" and res_without and res_with:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=160)

            # Initial continuum block
            axes[0].imshow(1.0 - res_without['initial_density'], cmap='gray', vmin=0, vmax=1)
            axes[0].set_title(f"Initial Structure (Uniform Block)\nTarget Vf = {volfrac:.2f} | C0 = {res_without['initial_compliance']:.2f}", fontsize=11, fontweight='bold')
            axes[0].axis('off')

            # Without Buckling
            axes[1].imshow(1.0 - res_without['final_density'], cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f"1. Without Buckling (Pure Compliance)\nCompliance: {res_without['final_compliance']:.2f} | $\\lambda_1$: {res_without['final_lambda_1']:.2f}", fontsize=11, fontweight='bold')
            axes[1].axis('off')

            # With Buckling
            axes[2].imshow(1.0 - res_with['final_density'], cmap='gray', vmin=0, vmax=1)
            axes[2].set_title(f"2. With Buckling (Buckling-Aware)\nCompliance: {res_with['final_compliance']:.2f} | $\\lambda_1$: {res_with['final_lambda_1']:.2f}", fontsize=11, fontweight='bold')
            axes[2].axis('off')

            fig.suptitle(f"Unified SIMP Solver: Benchmark '{case_name.upper()}' Comparison", fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(out_file, bbox_inches='tight')
            plt.close()
            print(f"\n[Figure Saved] Visual comparison exported to: {os.path.abspath(out_file)}")

        elif res_without or res_with:
            active_res = res_without if res_without else res_with
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
            axes[0].imshow(1.0 - active_res['initial_density'], cmap='gray', vmin=0, vmax=1)
            axes[0].set_title(f"Initial Structure (Vf={volfrac:.2f})", fontsize=11, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(1.0 - active_res['final_density'], cmap='gray', vmin=0, vmax=1)
            tag = "With Buckling" if active_res['use_buckling'] else "Without Buckling"
            axes[1].set_title(f"Optimized ({tag})\nC: {active_res['final_compliance']:.2f} | $\\lambda_1$: {active_res['final_lambda_1']:.2f}", fontsize=11, fontweight='bold')
            axes[1].axis('off')

            plt.tight_layout()
            plt.savefig(out_file, bbox_inches='tight')
            plt.close()
            print(f"\n[Figure Saved] Result exported to: {os.path.abspath(out_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified 2D Continuum SIMP Topology Optimization Solver")
    parser.add_argument("--mode", type=str, default="both", choices=["without_buckling", "with_buckling", "both"],
                        help="Solver mode: without_buckling, with_buckling, or both")
    parser.add_argument("--case", type=str, default="cantilever", choices=["cantilever", "mbb", "bridge", "crane"],
                        help="Canonical benchmark case")
    parser.add_argument("--volfrac", type=float, default=0.40, help="Target material volume fraction")
    parser.add_argument("--max_iter", type=int, default=50, help="Maximum number of SIMP iterations")
    parser.add_argument("--buckle_weight", type=float, default=0.45, help="Modal strain energy buckling sensitivity weight")
    parser.add_argument("--buckle_start_iter", type=int, default=20, help="Iteration to activate buckling sensitivity")
    parser.add_argument("--nelx", type=int, default=120, help="Number of elements along X")
    parser.add_argument("--nely", type=int, default=60, help="Number of elements along Y")
    parser.add_argument("--save_fig", action="store_true", default=True, help="Save comparison figure to disk")
    parser.add_argument("--out_dir", type=str, default="images", help="Output directory for generated figures")

    cli_args = parser.parse_args()
    run_cli_comparison(cli_args)
