"""
buckling_optimizer.py - 25-Iteration Active Buckling-Aware Topology Optimization
Implements a hybrid SIMP formulation that balances structural compliance (stiffness)
with geometric stiffness eigenvalue buckling resistance:
    Min Compliance & Max Buckling Load Factor (lambda_1) subject to Volume constraint.

Schedule:
- Iterations 1 to 14: Fast pure compliance minimization to establish primary truss topology.
- Iterations 15 to 25: Active modal strain energy sensitivity blending to reinforce
  slender compression struts against in-plane buckling.
"""

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import time

from dataset_generator import get_stiffness_matrix, precompute_mesh_and_filter
from structural_mechanics import solve_inplane_buckling


def optimize_with_buckling(F, fixeddofs, nelx=60, nely=30, volfrac=0.4,
                           penal=3.0, rmin=1.5, max_iter=25,
                           buckle_start_iter=15, buckle_weight=0.4,
                           Emax=1.0, Emin=1e-9, nu=0.3,
                           density_init=None, verbose=True):
    """
    Runs a 25-iteration buckling-aware SIMP topology optimization.
    
    Parameters:
        F: global force vector (ndof,)
        fixeddofs: array of constrained degree-of-freedom indices
        nelx, nely: domain resolution (number of elements along X and Y)
        volfrac: target solid volume fraction (e.g. 0.4)
        penal: SIMP stiffness penalization exponent (typically 3.0)
        rmin: sensitivity filter radius (in element units)
        max_iter: maximum number of iterations (default 25)
        buckle_start_iter: iteration where buckling sensitivity activates (default 15)
        buckle_weight: blending weight w in [0, 1] for buckling vs compliance
        density_init: optional initial density array (nely*nelx,)
        verbose: print progress per iteration
        
    Returns:
        dict containing:
            'density': 2D density array (nely, nelx) in Fortran order
            'history': list of per-iteration metrics (compliance, lambda_1, change)
            'final_compliance': float
            'final_lambda_1': float
            'elapsed_time': float (seconds)
    """
    start_time = time.time()
    ndof = 2 * (nelx + 1) * (nely + 1)
    nele = nelx * nely
    
    # 1. Element stiffness and mesh indexing
    KE = get_stiffness_matrix(Emax=Emax, nu=nu)
    edofMat, iK, jK, H, Hs = precompute_mesh_and_filter(nelx, nely, rmin)
    
    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixeddofs)
    
    # 2. Initialize densities
    if density_init is not None:
        x = density_init.flatten(order='F').copy()
    else:
        x = volfrac * np.ones(nele)
    xPhys = x.copy()
    
    history = []
    current_lambda_1 = 999.0
    
    if verbose:
        mode_str = f"Buckling-Aware (w={buckle_weight})" if buckle_weight > 0 else "Pure Compliance"
        print(f"--- Starting {max_iter}-Iteration SIMP [{mode_str}] ({nelx}x{nely}) ---")
    
    # 3. Main Optimization Loop
    for loop in range(1, max_iter + 1):
        t_iter_start = time.time()
        
        # Assemble global stiffness
        sK = ((KE.flatten()[np.newaxis, :]) * (Emin + xPhys[:, np.newaxis]**penal * (Emax - Emin))).flatten()
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K_free = K[freedofs, :][:, freedofs]
        
        # Solve FEA
        try:
            U_free = spsolve(K_free, F[freedofs])
        except Exception as ex:
            if verbose:
                print(f"FEA Solver error at iteration {loop}: {ex}")
            break
            
        U = np.zeros(ndof)
        U[freedofs] = U_free
        
        # Element compliance strain energy: ce = u_e^T * KE * u_e
        U_edof = U[edofMat]
        ce = (np.dot(U_edof, KE) * U_edof).sum(axis=1)
        compliance = float(np.sum((Emin + xPhys**penal * (Emax - Emin)) * ce))
        
        # Compliance sensitivity: dc = -p * rho^(p-1) * (Emax - Emin) * ce  (< 0)
        dc = -penal * (Emax - Emin) * (xPhys**(penal - 1)) * ce
        
        # Active Buckling Sensitivity (activated at buckle_start_iter)
        buckle_active = (loop >= buckle_start_iter and buckle_weight > 0.0)
        if buckle_active:
            # Solve eigenvalue buckling on condensed active structure
            buckle_res = solve_inplane_buckling(
                U, xPhys, edofMat, fixeddofs, nelx, nely,
                E=Emax, nu=nu, density_threshold=0.25, num_modes=2
            )
            
            if buckle_res is not None and "phi_vector" in buckle_res and buckle_res["critical_factor"] < 900:
                current_lambda_1 = buckle_res["critical_factor"]
                phi_1 = buckle_res["phi_vector"]
                
                # Modal strain energy of buckling mode: me = phi_e^T * KE * phi_e
                phi_edof = phi_1[edofMat]
                me = (np.dot(phi_edof, KE) * phi_edof).sum(axis=1)
                
                # Buckling sensitivity: d_lambda / d_rho_e
                # Stiffening buckling-prone elements increases buckling resistance
                d_lambda = penal * (xPhys**(penal - 1)) * me
                
                # Normalize both sensitivities to [0, 1] relative intensity
                max_dc = np.max(-dc)
                max_dlam = np.max(d_lambda)
                
                norm_comp = (-dc) / (max_dc + 1e-12)
                norm_buckle = d_lambda / (max_dlam + 1e-12)
                
                # Blended sensitivity: higher means more material desired
                # We want to minimize compliance (-dc) AND maximize buckling (d_lambda)
                blended_drive = (1.0 - buckle_weight) * norm_comp + buckle_weight * norm_buckle
                dJ = -blended_drive  # Keep negative for standard OC bisection
            else:
                dJ = dc.copy()
        else:
            dJ = dc.copy()
            
        # Volume sensitivity
        dv = np.ones(nele)
        
        # Mesh sensitivity filter
        dJ_filt = np.asarray(H.dot(x * dJ)) / Hs / np.maximum(0.001, x)
        dJ_filt = np.minimum(dJ_filt, -1e-12)
        
        # Optimality Criteria (OC) update with bisection on Lagrange multiplier
        l1, l2, move = 0.0, 1e9, 0.2
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
        
        iter_time = time.time() - t_iter_start
        
        history.append({
            'iteration': loop,
            'compliance': compliance,
            'lambda_1': current_lambda_1 if buckle_active else None,
            'change': change,
            'density': xPhys.reshape((nely, nelx), order='F').copy()
        })
        
        if verbose and (loop % 5 == 0 or loop == max_iter or loop == buckle_start_iter):
            b_tag = f" | Buckling lambda_1: {current_lambda_1:6.2f}" if buckle_active else " | Buckling: Inactive"
            print(f"Iter {loop:2d}/{max_iter} | Compliance: {compliance:10.4f} | Change: {change:6.4f}{b_tag} | ({iter_time:4.2f}s)")
            
    total_time = time.time() - start_time
    final_density = xPhys.reshape((nely, nelx), order='F')
    
    # Final buckling check if not yet evaluated
    if not buckle_active or current_lambda_1 > 900:
        final_check = solve_inplane_buckling(
            U, xPhys, edofMat, fixeddofs, nelx, nely,
            E=Emax, nu=nu, density_threshold=0.25, num_modes=1
        )
        if final_check is not None:
            current_lambda_1 = final_check["critical_factor"]
            
    if verbose:
        print(f"--- Completed in {total_time:.2f}s | Final C: {compliance:.4f} | Final lambda_1: {current_lambda_1:.2f} ---")
        
    return {
        "density": final_density,
        "history": history,
        "final_compliance": compliance,
        "final_lambda_1": current_lambda_1,
        "elapsed_time": total_time
    }
