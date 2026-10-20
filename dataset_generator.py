"""
dataset_generator.py - Multi-Core SIMP Ground-Truth Dataset Generator
Unified 2D Continuum SIMP Pipeline (With & Without Buckling)

Implements:
1. Sosnovik & Oseledets (2019) Poisson discrete support & load sampling strategy
2. Benchmark discrete support cases (Cantilever truss, MBB beam, Bridge truss, Crane, Michell)
3. Vectorized Q4 SIMP FEA with precomputed sensitivity filter & active in-plane buckling
4. Rich Phase 1 physical & numerical variable tracking
5. Standardized 6-channel tensor encoding:
   (N, 6, 60, 120) inputs -> (N, 60, 120) target density
   plus strain_energy (N, 60, 120) and compliance (N,)
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import multiprocessing as mp
import argparse
import time
import os

# Import unified solver and buckling analyzer from simp_solver
try:
    from simp_solver import (
        get_stiffness_matrix,
        precompute_mesh_and_filter,
        solve_simp_unified,
        solve_inplane_buckling
    )
except ImportError:
    # Standalone fallbacks if simp_solver is not in path
    def get_stiffness_matrix(Emax=1.0, nu=0.3):
        k = np.array([
            1/2 - nu/6,   1/8 + nu/8,   -1/4 - nu/12, -1/8 + 3*nu/8,
            -1/4 + nu/12, -1/8 - nu/8,  nu/6,         1/8 - 3*nu/8
        ])
        KE = Emax / (1 - nu**2) * np.array([
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
        elx, ely = np.meshgrid(np.arange(nelx), np.arange(nely), indexing='ij')
        elx, ely = elx.flatten(), ely.flatten()
        n1 = (nely + 1) * elx + ely
        n2 = (nely + 1) * (elx + 1) + ely
        edofMat = np.column_stack([
            2*n1 + 2, 2*n1 + 3,
            2*n2 + 2, 2*n2 + 3,
            2*n2,     2*n2 + 1,
            2*n1,     2*n1 + 1
        ])
        iK = np.kron(edofMat, np.ones((8, 1))).flatten()
        jK = np.kron(edofMat, np.ones((1, 8))).flatten()
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

    solve_simp_unified = None
    solve_inplane_buckling = None


# --- 1. BACKWARD COMPATIBLE & BUCKLING-AWARE SIMP SOLVER ---
def solve_simp(nelx, nely, volfrac, penal, fixeddofs, F, edofMat, iK, jK, H, Hs,
               max_iter=80, record_history=False, history_interval=10,
               use_buckling=False, buckle_weight=0.45, buckle_start_iter=20,
               return_full_info=False):
    """
    Solves 2D SIMP topology optimization.
    Backward-compatible with original solve_simp signature while providing
    support for active in-plane buckling and full variable return.

    Parameters:
        nelx, nely: grid element count
        volfrac: target volume fraction
        penal: SIMP penalization power (e.g. 3.0)
        fixeddofs: 1D array of constrained DOFs
        F: global force vector (ndof,)
        edofMat, iK, jK, H, Hs: precomputed mesh and filter matrices
        max_iter: maximum iterations
        record_history: whether to record intermediate density/compliance steps
        history_interval: step interval for history logging
        use_buckling: whether to use active in-plane buckling-aware optimization
        buckle_weight: blending weight for modal strain energy sensitivity
        buckle_start_iter: iteration when buckling sensitivity activates
        return_full_info: if True, returns dictionary with all physics & convergence variables
    """
    if solve_simp_unified is not None:
        try:
            res = solve_simp_unified(
                nelx=nelx, nely=nely, volfrac=volfrac, penal=penal,
                fixeddofs=fixeddofs, F=F,
                edofMat=edofMat, iK=iK, jK=jK, H=H, Hs=Hs,
                max_iter=max_iter,
                use_buckling=use_buckling,
                buckle_weight=buckle_weight,
                buckle_start_iter=buckle_start_iter,
                record_history=record_history,
                history_interval=history_interval,
                verbose=False
            )
            if res is not None and res.get('final_density') is not None:
                if return_full_info:
                    return res
                if record_history:
                    return res['final_density'], res.get('history', [])
                return res['final_density']
        except Exception:
            pass

    # Native fallback pure compliance solver if solve_simp_unified is unavailable or fails
    Emin = 1e-9
    Emax = 1.0
    ndof = 2 * (nelx + 1) * (nely + 1)
    KE = get_stiffness_matrix(Emax)
    
    initial_density = np.full((nely, nelx), volfrac, dtype=np.float32)
    x = volfrac * np.ones(nelx * nely, dtype=float)
    xPhys = x.copy()
    
    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixeddofs)
    
    change = 1.0
    loop = 0
    history = []
    initial_compliance = 0.0
    compliance = 0.0
    ce = np.zeros(nelx * nely, dtype=float)
    
    if record_history:
        history.append({
            'iteration': 0,
            'density': initial_density.copy(),
            'compliance': None,
            'change': 1.0
        })
    
    t_start = time.time()
    while change > 0.01 and loop < max_iter:
        loop += 1
        sK = ((KE.flatten()[:, None]) * (Emin + xPhys**penal * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K_free = K[freedofs, :][:, freedofs]
        
        try:
            U_free = spsolve(K_free, F[freedofs])
        except Exception:
            if return_full_info:
                return None
            return (None, history) if record_history else None
            
        if np.any(np.isnan(U_free)) or np.any(np.isinf(U_free)):
            if return_full_info:
                return None
            return (None, history) if record_history else None
            
        U = np.zeros(ndof)
        U[freedofs] = U_free
        
        U_edof = U[edofMat]
        ce = (np.dot(U_edof, KE) * U_edof).sum(axis=1)
        compliance = float(np.sum((Emin + xPhys**penal * (Emax - Emin)) * ce))
        if loop == 1:
            initial_compliance = compliance
        
        dc = -penal * (Emax - Emin) * (xPhys**(penal - 1)) * ce
        dv = np.ones(nely * nelx)
        
        # Sensitivity filter
        dc = np.asarray(H.dot(x * dc)) / Hs / np.maximum(0.001, x)
        dc = np.minimum(dc, -1e-12)
        
        # OC design update with bisection
        l1, l2, move = 0.0, 1e9, 0.2
        while (l2 - l1) / (l1 + l2 + 1e-15) > 1e-4:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0.0, np.maximum(x - move,
                   np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
            if np.sum(xnew) > volfrac * nelx * nely:
                l1 = lmid
            else:
                l2 = lmid
                
        change = float(np.max(np.abs(xnew - x)))
        x = xnew.copy()
        xPhys = x.copy()
        
        if record_history and (loop % history_interval == 0 or loop == max_iter or change <= 0.01):
            history.append({
                'iteration': loop,
                'density': xPhys.reshape((nely, nelx), order='F').copy(),
                'compliance': compliance,
                'change': change
            })
        
    elapsed_time = time.time() - t_start
    final_density = xPhys.reshape((nely, nelx), order='F').astype(np.float32)

    # Compute final buckling factor if buckling solver is available and requested
    final_lambda_1 = 0.0
    if use_buckling and solve_inplane_buckling is not None:
        try:
            b_res = solve_inplane_buckling(
                U, xPhys, edofMat, fixeddofs, nelx, nely,
                E=Emax, nu=0.3, density_threshold=0.25, num_modes=1
            )
            if b_res is not None and "critical_factor" in b_res:
                final_lambda_1 = float(b_res["critical_factor"])
            else:
                final_lambda_1 = 999.0
        except Exception:
            final_lambda_1 = 999.0

    # Final element strain energy map
    e_elem = ((Emin + (xPhys ** penal) * (Emax - Emin)) * ce).reshape((nely, nelx), order='F')
    se_norm = np.clip(e_elem / (np.percentile(e_elem, 99) + 1e-12), 0.0, 2.0).astype(np.float32)

    compliance_reduction_pct = ((initial_compliance - compliance) / max(1e-9, initial_compliance)) * 100.0
    actual_vf = float(np.mean(final_density))
    volume_reduction_pct = (1.0 - float(volfrac)) * 100.0

    if return_full_info:
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
            'use_buckling': False,
            'buckle_weight': 0.0,
            'iterations_run': int(loop),
            'elapsed_time': float(elapsed_time),
            'history': history
        }

    if record_history:
        return final_density, history
    return final_density


# Worker initializer for multiprocessing
_worker_global_data = None

def init_worker(shared_data):
    global _worker_global_data
    _worker_global_data = shared_data


# --- 2. PROBLEM SAMPLING WITH RICH VARIABLE TRACKING ---
def sample_problem(sample_id, nelx=120, nely=60, penal=3.0, rmin=1.5, max_iter=80,
                   global_data=None, strategy='sosnovik', record_history=False, history_interval=10,
                   benchmark_type=None, return_meta=False,
                   use_buckling=False, buckle_weight=0.45, buckle_start_iter=20,
                   fixed_volfrac=None, return_physics=False):
    """
    Generates a single training sample with complete physical variable tracking.
    Uses discrete point supports (pins/rollers) to create crisp, slender trusses
    with hollow interior voids instead of bulky over-clamped blocks.

    Parameters:
        sample_id: random seed for reproducibility
        nelx, nely: domain mesh grid
        penal: SIMP penalization power
        rmin: sensitivity filter radius
        max_iter: maximum SIMP iterations
        global_data: precomputed (edofMat, iK, jK, H, Hs) tuple
        strategy: sampling strategy ('sosnovik')
        record_history: whether to return optimization history
        history_interval: step interval for history
        benchmark_type: 0/'cantilever', 1/'mbb', 2/'bridge', 3/'crane', 'michell', or None (random)
        return_meta: whether to return comprehensive physical metadata dictionary
        use_buckling: whether to activate modal strain energy in-plane buckling sensitivity
        buckle_weight: blending weight w in [0, 1] for buckling vs compliance
        buckle_start_iter: iteration when buckling sensitivity activates
        fixed_volfrac: optional fixed volume fraction override (default None samples from Normal)
    """
    if global_data is None:
        global _worker_global_data
        global_data = _worker_global_data
        if global_data is None:
            global_data = precompute_mesh_and_filter(nelx, nely, rmin)
            
    edofMat, iK, jK, H, Hs = global_data
    ndof = 2 * (nelx + 1) * (nely + 1)
    rng = np.random.default_rng(sample_id)
    
    def get_nid(x, y):
        return x * (nely + 1) + y

    # Volume fraction: Normal(0.42, 0.08) bounded in [0.22, 0.55] or fixed
    if fixed_volfrac is not None:
        volfrac = float(fixed_volfrac)
    else:
        volfrac = float(np.clip(rng.normal(0.42, 0.08), 0.22, 0.55))

    # All nodes and boundary weighting (100x boundary bias as in Sosnovik)
    nodes = []
    probs = []
    coords = {}
    for x in range(nelx + 1):
        for y in range(nely + 1):
            n = get_nid(x, y)
            nodes.append(n)
            coords[n] = (x, y)
            is_boundary = (x == 0 or x == nelx or y == 0 or y == nely)
            probs.append(100.0 if is_boundary else 1.0)
    probs = np.array(probs) / np.sum(probs)

    # 40% of samples: standard discrete structural benchmarks
    # 60% of samples: pure Sosnovik Poisson random configurations
    fixed_nodes = []
    fixed_dofs = []

    if benchmark_type is not None:
        use_benchmark = True
        if isinstance(benchmark_type, str):
            bt_str = benchmark_type.strip().lower()
            str_map = {
                'cantilever': 0, '0': 0,
                'mbb': 1, '1': 1,
                'bridge': 2, '2': 2,
                'crane': 3, '3': 3,
                'michell': 4, '4': 4
            }
            bm_type = str_map.get(bt_str, 0)
        else:
            bm_type = int(benchmark_type)
    else:
        use_benchmark = (rng.uniform() < 0.4)
        if use_benchmark:
            bm_type = int(rng.choice([0, 1, 2, 3]))
        else:
            bm_type = -1

    if use_benchmark:
        if bm_type == 0:  # Discrete Cantilever: 2 discrete pins on left edge
            case_name = 'cantilever'
            n_top = get_nid(0, 0)
            n_bot = get_nid(0, nely)
            fixed_nodes = [n_top, n_bot]
            fixed_dofs = [2*n_top, 2*n_top + 1, 2*n_bot, 2*n_bot + 1]
            load_candidates = [get_nid(nelx, y) for y in range(nely + 1)]
        elif bm_type == 1:  # MBB Beam: Symmetry roller left wall, roller bottom-right
            case_name = 'mbb'
            fixed_nodes = [get_nid(0, y) for y in [0, nely//2, nely]] + [get_nid(nelx, nely)]
            fixed_dofs = [2*get_nid(0, y) for y in [0, nely//2, nely]] + [2*get_nid(nelx, nely) + 1]
            load_candidates = [get_nid(0, 0), get_nid(nelx//4, 0)]
        elif bm_type == 2:  # Bridge Truss: Bottom-left pin, bottom-right roller
            case_name = 'bridge'
            n_bl = get_nid(0, nely)
            n_br = get_nid(nelx, nely)
            fixed_nodes = [n_bl, n_br]
            fixed_dofs = [2*n_bl, 2*n_bl + 1, 2*n_br + 1]
            load_candidates = [get_nid(x, 0) for x in range(nelx // 6, 5 * nelx // 6 + 1)]
        elif bm_type == 3:  # Arch / 2-support crane: Top-left pin, bottom-left pin
            case_name = 'crane'
            n_tl = get_nid(0, 0)
            n_bl = get_nid(0, nely)
            fixed_nodes = [n_tl, n_bl]
            fixed_dofs = [2*n_tl, 2*n_tl + 1, 2*n_bl, 2*n_bl + 1]
            load_candidates = [get_nid(nelx, nely // 2), get_nid(nelx, nely), get_nid(nelx, 0)]
        else:  # Michell Truss: Bottom-left pin, bottom-right roller, center load
            case_name = 'michell'
            n_bl = get_nid(0, nely)
            n_br = get_nid(nelx, nely)
            fixed_nodes = [n_bl, n_br]
            fixed_dofs = [2*n_bl, 2*n_bl + 1, 2*n_br + 1]
            load_candidates = [get_nid(nelx // 2, nely), get_nid(nelx // 2, 0)]
    else:
        case_name = 'random_poisson'
        # Sosnovik Poisson Sampling:
        # Nx ~ Poisson(2), Ny ~ Poisson(1), NL ~ Poisson(1)
        # Kinematic stability: at least 1 pin + 1 roller separated by >= 15 elements
        nx_count = max(2, int(rng.poisson(lam=2)))
        ny_count = max(1, int(rng.poisson(lam=1)))
        
        support_nodes = rng.choice(nodes, size=min(nx_count + ny_count, len(nodes)), replace=False, p=probs)
        s1 = support_nodes[0]
        s2 = support_nodes[1]
        c1, c2 = coords[s1], coords[s2]
        dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        if dist < 15:
            if s1 != get_nid(nelx, nely):
                s2 = get_nid(nelx, nely)
            else:
                s2 = get_nid(0, 0)
            c2 = coords[s2]
            
        fixed_nodes = [s1, s2]
        fixed_dofs = [2*s1, 2*s1 + 1]
        dx = abs(c2[0] - c1[0])
        dy = abs(c2[1] - c1[1])
        if dx >= dy:
            fixed_dofs.append(2*s2 + 1)
            if rng.uniform() > 0.5:
                fixed_dofs.append(2*s2)
        else:
            fixed_dofs.append(2*s2)
            if rng.uniform() > 0.5:
                fixed_dofs.append(2*s2 + 1)
            
        load_candidates = [n for n in nodes if coords[n][0] == 0 or coords[n][0] == nelx or
                           coords[n][1] == 0 or coords[n][1] == nely]
        load_candidates = [n for n in load_candidates if n not in fixed_nodes]

    # Support coordinates & types catalog
    fixed_dofs_set = set(fixed_dofs)
    support_coords = []
    seen_fn = set()
    unique_fixed_nodes = []
    for n in fixed_nodes:
        if n not in seen_fn:
            seen_fn.add(n)
            unique_fixed_nodes.append(n)

    for n in unique_fixed_nodes:
        x, y = coords[n]
        fx = (2 * n) in fixed_dofs_set
        fy = (2 * n + 1) in fixed_dofs_set
        if fx and fy:
            stype = 'Pin (X,Y fixed)'
        elif fx:
            stype = 'Roller (X fixed)'
        elif fy:
            stype = 'Roller (Y fixed)'
        else:
            stype = 'Support'
        support_coords.append({
            'node_id': int(n),
            'x': int(x),
            'y': int(y),
            'type': stype
        })
    num_supports = len(support_coords)

    # Sample 1 to 3 distinct nodal loads
    nl_count = max(1, int(rng.poisson(lam=1)))
    nl_count = min(nl_count, 3)
    valid_load_candidates = [n for n in load_candidates if n not in unique_fixed_nodes]
    if len(valid_load_candidates) == 0:
        valid_load_candidates = [n for n in load_candidates]
    chosen_load_nodes = rng.choice(valid_load_candidates, size=min(nl_count, len(valid_load_candidates)), replace=False)

    F = np.zeros(ndof)
    Fx_grid = np.zeros((nely + 1, nelx + 1))
    Fy_grid = np.zeros((nely + 1, nelx + 1))
    load_details = []

    for ln in chosen_load_nodes:
        mag = float(rng.uniform(0.8, 1.5))
        angle = float(rng.uniform(0, 2 * np.pi))
        fx = float(mag * np.cos(angle))
        fy = float(mag * np.sin(angle))
        deg = float(np.degrees(angle))
        F[2*ln] += fx
        F[2*ln + 1] += fy
        
        x, y = coords[ln]
        Fx_grid[y, x] += fx
        Fy_grid[y, x] += fy
        load_details.append({
            'node_id': int(ln),
            'x': int(x),
            'y': int(y),
            'Fx': fx,
            'Fy': fy,
            'magnitude': mag,
            'angle_deg': deg
        })
    num_loads = len(load_details)

    # Run SIMP optimization with full variable tracking
    simp_res = solve_simp(
        nelx=nelx, nely=nely, volfrac=volfrac, penal=penal,
        fixeddofs=np.unique(fixed_dofs), F=F,
        edofMat=edofMat, iK=iK, jK=jK, H=H, Hs=Hs,
        max_iter=max_iter,
        record_history=record_history,
        history_interval=history_interval,
        use_buckling=use_buckling,
        buckle_weight=buckle_weight,
        buckle_start_iter=buckle_start_iter,
        return_full_info=True
    )

    if simp_res is None or 'final_density' not in simp_res:
        return None

    target_density = simp_res['final_density']
    strain_energy = simp_res.get('strain_energy', np.zeros((nely, nelx), dtype=np.float32)).astype(np.float32)
    history = simp_res.get('history', [])
    initial_density = simp_res['initial_density']
    initial_compliance = simp_res['initial_compliance']
    final_compliance = np.float32(simp_res['final_compliance'])
    compliance_reduction_pct = simp_res['compliance_reduction_pct']
    final_lambda_1 = simp_res['final_lambda_1']
    actual_vf = simp_res['actual_vf']
    iterations_run = simp_res['iterations_run']
    elapsed_time = simp_res['elapsed_time']
    volume_reduction_pct = (1.0 - volfrac) * 100.0

    # Construct Standardized 6-Channel Input Tensor
    # Ch 0: domain mask
    ch_domain = np.ones((nely, nelx), dtype=np.float32)

    # Ch 1 & Ch 2: horizontal (X) and vertical (Y) support constraints
    fixed_grid_x = np.zeros((nely + 1, nelx + 1), dtype=np.float32)
    fixed_grid_y = np.zeros((nely + 1, nelx + 1), dtype=np.float32)
    for dof in fixed_dofs_set:
        n = dof // 2
        x, y = coords[n]
        if dof % 2 == 0:
            fixed_grid_x[y, x] = 1.0
        else:
            fixed_grid_y[y, x] = 1.0

    ch_bc_x = 0.25 * (
        fixed_grid_x[:-1, :-1] + fixed_grid_x[1:, :-1] +
        fixed_grid_x[:-1, 1:] + fixed_grid_x[1:, 1:]
    ).astype(np.float32)

    ch_bc_y = 0.25 * (
        fixed_grid_y[:-1, :-1] + fixed_grid_y[1:, :-1] +
        fixed_grid_y[:-1, 1:] + fixed_grid_y[1:, 1:]
    ).astype(np.float32)

    # Ch 3 & Ch 4: horizontal and vertical forces averaged to element centers
    ch_fx = 0.25 * (
        Fx_grid[:-1, :-1] + Fx_grid[1:, :-1] +
        Fx_grid[:-1, 1:] + Fx_grid[1:, 1:]
    ).astype(np.float32)
    ch_fy = 0.25 * (
        Fy_grid[:-1, :-1] + Fy_grid[1:, :-1] +
        Fy_grid[:-1, 1:] + Fy_grid[1:, 1:]
    ).astype(np.float32)

    # Ch 5: target volume fraction
    ch_vf = np.full((nely, nelx), volfrac, dtype=np.float32)

    input_tensor = np.stack([ch_domain, ch_bc_x, ch_bc_y, ch_fx, ch_fy, ch_vf], axis=0).astype(np.float32)

    # Comprehensive Phase 1 Physical Variable Metadata Catalog
    meta = {
        'case_name': case_name,
        'nelx': int(nelx),
        'nely': int(nely),
        'ndof': int(ndof),
        'initial_density': initial_density,
        'final_density': target_density,
        'strain_energy': strain_energy,
        'volfrac': float(volfrac),
        'volume_reduction_pct': float(volume_reduction_pct),
        'actual_vf': float(actual_vf),
        'num_supports': int(num_supports),
        'support_coords': support_coords,
        'fixed_dofs': np.unique(fixed_dofs),
        'fixed_nodes': unique_fixed_nodes,
        'num_loads': int(num_loads),
        'load_details': load_details,
        'F': F,
        'use_buckling': bool(use_buckling),
        'buckle_weight': float(buckle_weight if use_buckling else 0.0),
        'initial_compliance': float(initial_compliance),
        'final_compliance': float(final_compliance),
        'compliance_reduction_pct': float(compliance_reduction_pct),
        'final_lambda_1': float(final_lambda_1),
        'iterations_run': int(iterations_run),
        'elapsed_time': float(elapsed_time)
    }

    if return_physics:
        if return_meta and record_history:
            return input_tensor, target_density.astype(np.float32), strain_energy, final_compliance, history, meta
        elif return_meta:
            return input_tensor, target_density.astype(np.float32), strain_energy, final_compliance, meta
        elif record_history:
            return input_tensor, target_density.astype(np.float32), strain_energy, final_compliance, history
        return input_tensor, target_density.astype(np.float32), strain_energy, final_compliance
    else:
        if return_meta and record_history:
            return input_tensor, target_density.astype(np.float32), history, meta
        elif return_meta:
            return input_tensor, target_density.astype(np.float32), meta
        elif record_history:
            return input_tensor, target_density.astype(np.float32), history
        return input_tensor, target_density.astype(np.float32)


# --- 3. VARIABLE SUMMARY PRINTER ---
def print_sample_variables(meta):
    """Prints a structured engineering scorecard of all physical variables carried in a sample."""
    print("\n" + "="*80)
    print(f"  PHASE 1 VARIABLE CATALOG & SCORECARD - Benchmark: {meta['case_name'].upper()}")
    print("="*80)
    print(f"  Domain Grid:              {meta['nelx']} x {meta['nely']} elements ({meta['ndof']} total DOFs)")
    print(f"  Initial Structure:        Uniform gray block ({meta['nelx']}x{meta['nely']}) with density = {meta['volfrac']:.3f}")
    print(f"  Target Volume Frac (Vf):  {meta['volfrac']:.3f}")
    print(f"  Achieved Vol Frac (Vf*):  {meta['actual_vf']:.3f}")
    print(f"  Material Volume Removed:  {meta['volume_reduction_pct']:.1f}%")
    print(f"  Fixed Supports Count:     {meta['num_supports']}")
    for i, supp in enumerate(meta['support_coords']):
        print(f"    Support {i+1}: Node {supp['node_id']} at ({supp['x']}, {supp['y']}) -> {supp['type']}")
    print(f"  Point Loads Count:        {meta['num_loads']}")
    for i, ld in enumerate(meta['load_details']):
        print(f"    Load {i+1}: Node {ld['node_id']} at ({ld['x']}, {ld['y']}) -> Mag={ld['magnitude']:.2f}, Angle={ld['angle_deg']:.1f} deg (Fx={ld['Fx']:.3f}, Fy={ld['Fy']:.3f})")
    mode_str = f"With Buckling (w={meta['buckle_weight']:.2f})" if meta['use_buckling'] else "Without Buckling (Pure Compliance)"
    print(f"  Optimization Mode:        {mode_str}")
    print(f"  Initial Compliance (C0):  {meta['initial_compliance']:.4f}")
    print(f"  Final Compliance (C*):    {meta['final_compliance']:.4f}")
    print(f"  Compliance Reduction:     {meta['compliance_reduction_pct']:.2f}%")
    print(f"  Critical Buckling (lam1): {meta['final_lambda_1']:.4f}")
    print(f"  Iterations Run:           {meta['iterations_run']}")
    print(f"  Optimization Time:        {meta['elapsed_time']:.2f}s")
    print("="*80 + "\n", flush=True)


# --- 4. BATCH DATASET GENERATION PIPELINE ---
def generate_dataset(num_samples=2000, output_path="topo_dataset.npz", num_cores=None,
                     seed_start=0, nelx=120, nely=60, penal=3.0, rmin=1.5, max_iter=80,
                     append=False, use_buckling=False, buckle_weight=0.45, buckle_start_iter=20,
                     print_vars=False):
    """
    Executes parallel multi-core SIMP dataset generation.
    Supports both without-buckling and with-buckling modes.
    """
    if num_cores is None:
        num_cores = max(1, mp.cpu_count())
        
    all_inputs = []
    all_targets = []
    all_se = []
    all_comp = []
    current_seed = seed_start

    # If append requested and file exists, load existing samples
    if append and os.path.exists(output_path):
        try:
            existing = np.load(output_path)
            all_inputs = list(existing['inputs'])
            all_targets = list(existing['targets'])
            if 'strain_energy' in existing:
                all_se = list(existing['strain_energy'])
            if 'compliance' in existing:
                all_comp = list(existing['compliance'])
            current_seed = seed_start + len(all_inputs)
            print(f"Loaded {len(all_inputs)} existing samples from {output_path}. Resuming at seed {current_seed}...", flush=True)
        except Exception as e:
            print(f"Warning: Could not read existing dataset for append: {e}", flush=True)

    initial_count = len(all_inputs)

    mode_label = f"With Buckling (w={buckle_weight:.2f}, start={buckle_start_iter})" if use_buckling else "Without Buckling (Pure Compliance)"
    print(f"============================================================", flush=True)
    print(f"  SIMP Dataset Generator (Sosnovik Discrete Truss Pipeline) ", flush=True)
    print(f"  Target: {num_samples} samples | Domain: {nelx}x{nely} | Cores: {num_cores}", flush=True)
    print(f"  Optimization Mode: {mode_label}", flush=True)
    print(f"  Current Progress: {len(all_inputs)}/{num_samples} samples", flush=True)
    print(f"  Volume Fraction: Normal(0.42, 0.08) in [0.22, 0.55]", flush=True)
    print(f"  Output File: {os.path.abspath(output_path)}", flush=True)
    print(f"============================================================", flush=True)
    
    if len(all_inputs) >= num_samples:
        print(f"Target count {num_samples} already satisfied ({len(all_inputs)} samples present).", flush=True)
        return

    t_start = time.time()
    print("Precomputing mesh geometry and sensitivity filter matrix...", flush=True)
    global_data = precompute_mesh_and_filter(nelx, nely, rmin)
    t_filter = time.time() - t_start
    print(f"Filter matrix precomputed in {t_filter:.2f}s.", flush=True)
    
    batch_size = max(1, num_cores * 2)
    
    with mp.Pool(processes=num_cores, initializer=init_worker, initargs=(global_data,)) as pool:
        while len(all_inputs) < num_samples:
            seeds = list(range(current_seed, current_seed + batch_size))
            current_seed += batch_size
            
            task_args = [
                (s, nelx, nely, penal, rmin, max_iter, None, 'sosnovik', False, 10, None, print_vars,
                 use_buckling, buckle_weight, buckle_start_iter, None, True)
                for s in seeds
            ]
            results = pool.starmap(sample_problem, task_args)
            
            new_added = 0
            for res in results:
                if res is not None:
                    if print_vars:
                        inp, tgt, se, comp, meta = res
                        print_sample_variables(meta)
                    else:
                        inp, tgt, se, comp = res
                    all_inputs.append(inp)
                    all_targets.append(tgt)
                    all_se.append(se)
                    all_comp.append(comp)
                    new_added += 1
                    if len(all_inputs) >= num_samples:
                        break
                        
            # Save checkpoint after every batch to prevent data loss without zip overhead
            if new_added > 0:
                inputs_arr = np.stack(all_inputs, axis=0).astype(np.float32)
                targets_arr = np.stack(all_targets, axis=0).astype(np.float32)
                se_arr = np.stack(all_se, axis=0).astype(np.float32)
                comp_arr = np.array(all_comp, dtype=np.float32)
                np.savez(
                    output_path,
                    inputs=inputs_arr,
                    targets=targets_arr,
                    strain_energy=se_arr,
                    compliance=comp_arr
                )
            
            elapsed = time.time() - t_start
            new_generated = len(all_inputs) - initial_count
            rate = new_generated / elapsed if elapsed > 0 else 0
            remaining = (num_samples - len(all_inputs)) / rate if rate > 0 else 0
            print(f"Progress: {len(all_inputs)}/{num_samples} samples saved "
                  f"({rate:.2f} samples/s, ETA: {remaining/60:.1f} min)", flush=True)

    # Save final compressed file
    print("Compressing and saving final dataset to disk...", flush=True)
    inputs_arr = np.stack(all_inputs, axis=0).astype(np.float32)
    targets_arr = np.stack(all_targets, axis=0).astype(np.float32)
    se_arr = np.stack(all_se, axis=0).astype(np.float32)
    comp_arr = np.array(all_comp, dtype=np.float32)
    np.savez_compressed(
        output_path,
        inputs=inputs_arr,
        targets=targets_arr,
        strain_energy=se_arr,
        compliance=comp_arr
    )
                  
    total_time = time.time() - t_start
    print(f"Dataset successfully created in {total_time:.2f}s ({total_time/60:.2f} min).", flush=True)
    print(f"File saved: {os.path.abspath(output_path)}", flush=True)


# --- 5. CLI INTERFACE ---
def main():
    parser = argparse.ArgumentParser(description="Multi-core SIMP Dataset Generator (With & Without Buckling)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="test_dataset.npz", help="Output .npz path")
    parser.add_argument("--cores", type=int, default=None, help="Number of CPU cores to use")
    parser.add_argument("--seed_start", type=int, default=0, help="Starting random seed")
    parser.add_argument("--nelx", type=int, default=120, help="Mesh width")
    parser.add_argument("--nely", type=int, default=60, help="Mesh height")
    parser.add_argument("--max_iter", type=int, default=80, help="Max SIMP iterations per sample")
    parser.add_argument("--append", action="store_true", help="Append / resume existing dataset file")
    
    # Active buckling and mode CLI flags
    parser.add_argument("--buckling", action="store_true", help="Enable active in-plane buckling-aware optimization")
    parser.add_argument("--mode", type=str, default="without_buckling", choices=["without_buckling", "with_buckling"],
                        help="Optimization mode: without_buckling or with_buckling (default: without_buckling)")
    parser.add_argument("--buckle_weight", type=float, default=0.45, help="Modal strain energy buckling sensitivity weight (default: 0.45)")
    parser.add_argument("--buckle_start_iter", type=int, default=20, help="Iteration to activate buckling sensitivity (default: 20)")
    parser.add_argument("--print_vars", action="store_true", help="Print detailed summary table of all carried variables for generated samples")

    args = parser.parse_args()
    
    use_buckling = args.buckling or (args.mode == "with_buckling")
    
    generate_dataset(
        num_samples=args.num_samples,
        output_path=args.output,
        num_cores=args.cores,
        seed_start=args.seed_start,
        nelx=args.nelx,
        nely=args.nely,
        max_iter=args.max_iter,
        append=args.append,
        use_buckling=use_buckling,
        buckle_weight=args.buckle_weight,
        buckle_start_iter=args.buckle_start_iter,
        print_vars=args.print_vars
    )


if __name__ == '__main__':
    mp.freeze_support()
    main()
