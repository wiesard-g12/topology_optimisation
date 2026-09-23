"""
dataset_generator.py - Multi-Core SIMP Ground-Truth Dataset Generator
Implements:
1. Sosnovik & Oseledets (2019) Poisson discrete support & load sampling strategy
2. Benchmark discrete support cases (Cantilever truss, MBB beam, Bridge truss)
3. Vectorized Q4 SIMP FEA with precomputed sensitivity filter
4. Standardized 5-channel tensor encoding:
   (N, 5, 60, 120) inputs -> (N, 60, 120) target density
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import multiprocessing as mp
import argparse
import time
import os

# --- 1. ELEMENT STIFFNESS MATRIX (Q4 Isoparametric) ---
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

# --- 2. VECTORIZED EDOF MATRIX & PRECOMPUTED FILTER ---
def precompute_mesh_and_filter(nelx, nely, rmin):
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

# --- 3. CORE SIMP OPTIMIZATION SOLVER ---
def solve_simp(nelx, nely, volfrac, penal, fixeddofs, F, edofMat, iK, jK, H, Hs, max_iter=80):
    Emin = 1e-9
    Emax = 1.0
    ndof = 2 * (nelx + 1) * (nely + 1)
    KE = get_stiffness_matrix(Emax)
    
    x = volfrac * np.ones(nelx * nely, dtype=float)
    xPhys = x.copy()
    
    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixeddofs)
    
    change = 1.0
    loop = 0
    
    while change > 0.01 and loop < max_iter:
        loop += 1
        sK = ((KE.flatten()[:, None]) * (Emin + xPhys**penal * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K_free = K[freedofs, :][:, freedofs]
        
        try:
            U_free = spsolve(K_free, F[freedofs])
        except Exception:
            return None
            
        if np.any(np.isnan(U_free)) or np.any(np.isinf(U_free)):
            return None
            
        U = np.zeros(ndof)
        U[freedofs] = U_free
        
        U_edof = U[edofMat]
        ce = (np.dot(U_edof, KE) * U_edof).sum(axis=1)
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
                
        change = np.max(np.abs(xnew - x))
        x = xnew.copy()
        xPhys = x.copy()
        
    return xPhys.reshape((nely, nelx), order='F')

# Worker initializer for multiprocessing
_worker_global_data = None

def init_worker(shared_data):
    global _worker_global_data
    _worker_global_data = shared_data

# --- 4. PROBLEM SAMPLING (SOSNOVIK POISSON DISCRETE STRATEGY) ---
def sample_problem(sample_id, nelx=120, nely=60, penal=3.0, rmin=1.5, max_iter=80,
                   global_data=None, strategy='sosnovik'):
    """
    Generates a single training sample.
    Uses discrete point supports (pins/rollers) to create crisp, slender trusses
    with hollow interior voids instead of bulky over-clamped blocks.
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

    # Volume fraction: Normal(0.42, 0.08) bounded in [0.22, 0.55]
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
    use_benchmark = (rng.uniform() < 0.4)

    fixed_nodes = []
    fixed_dofs = []

    if use_benchmark:
        bm_type = rng.choice([0, 1, 2, 3])
        if bm_type == 0:  # Discrete Cantilever: 2 discrete pins on left edge
            n_top = get_nid(0, 0)
            n_bot = get_nid(0, nely)
            fixed_nodes = [n_top, n_bot]
            fixed_dofs = [2*n_top, 2*n_top + 1, 2*n_bot, 2*n_bot + 1]
            load_candidates = [get_nid(nelx, y) for y in range(nely + 1)]
        elif bm_type == 1:  # MBB Beam: Symmetry roller left wall, roller bottom-right
            fixed_nodes = [get_nid(0, y) for y in [0, nely//2, nely]] + [get_nid(nelx, nely)]
            fixed_dofs = [2*get_nid(0, y) for y in [0, nely//2, nely]] + [2*get_nid(nelx, nely) + 1]
            load_candidates = [get_nid(0, 0), get_nid(nelx//4, 0)]
        elif bm_type == 2:  # Bridge Truss: Bottom-left pin, bottom-right roller
            n_bl = get_nid(0, nely)
            n_br = get_nid(nelx, nely)
            fixed_nodes = [n_bl, n_br]
            fixed_dofs = [2*n_bl, 2*n_bl + 1, 2*n_br + 1]
            load_candidates = [get_nid(x, 0) for x in range(nelx // 6, 5 * nelx // 6 + 1)]
        else:  # Arch / 2-support crane: Top-left pin, bottom-left pin
            n_tl = get_nid(0, 0)
            n_bl = get_nid(0, nely)
            fixed_nodes = [n_tl, n_bl]
            fixed_dofs = [2*n_tl, 2*n_tl + 1, 2*n_bl, 2*n_bl + 1]
            load_candidates = [get_nid(nelx, nely // 2), get_nid(nelx, nely), get_nid(nelx, 0)]
    else:
        # Sosnovik Poisson Sampling:
        # Nx ~ Poisson(2), Ny ~ Poisson(1), NL ~ Poisson(1)
        # To guarantee kinematic stability: place at least 1 pin + 1 roller at distinct boundary nodes
        nx_count = max(2, int(rng.poisson(lam=2)))
        ny_count = max(1, int(rng.poisson(lam=1)))
        
        # Pick 2-3 distinct boundary support nodes
        support_nodes = rng.choice(nodes, size=min(nx_count + ny_count, len(nodes)), replace=False, p=probs)
        # Ensure at least 2 distinct support nodes separated by at least 15 elements
        s1 = support_nodes[0]
        s2 = support_nodes[1]
        c1, c2 = coords[s1], coords[s2]
        dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        if dist < 15:
            # Pick corner if too close
            s2 = get_nid(nelx, nely)
            
        fixed_nodes = [s1, s2]
        # s1 is a full pin (x, y fixed), s2 is a roller/pin
        fixed_dofs = [2*s1, 2*s1 + 1, 2*s2 + 1]
        if rng.uniform() > 0.5:
            fixed_dofs.append(2*s2) # make s2 a pin as well
            
        # Candidates for load: boundary nodes away from supports
        load_candidates = [n for n in nodes if coords[n][0] == 0 or coords[n][0] == nelx or
                           coords[n][1] == 0 or coords[n][1] == nely]
        load_candidates = [n for n in load_candidates if n not in fixed_nodes]

    # Sample 1 to 3 distinct nodal loads
    nl_count = max(1, int(rng.poisson(lam=1)))
    nl_count = min(nl_count, 3)
    chosen_load_nodes = rng.choice(load_candidates, size=min(nl_count, len(load_candidates)), replace=False)

    F = np.zeros(ndof)
    Fx_grid = np.zeros((nely + 1, nelx + 1))
    Fy_grid = np.zeros((nely + 1, nelx + 1))

    for ln in chosen_load_nodes:
        # Load value: magnitude in [0.8, 1.5], angle directed inward/boundary
        mag = rng.uniform(0.8, 1.5)
        angle = rng.uniform(0, 2 * np.pi)
        fx = mag * np.cos(angle)
        fy = mag * np.sin(angle)
        F[2*ln] += fx
        F[2*ln + 1] += fy
        
        x = coords[ln][0]
        y = coords[ln][1]
        Fx_grid[y, x] += fx
        Fy_grid[y, x] += fy

    # Run SIMP optimization
    target_density = solve_simp(
        nelx, nely, volfrac, penal,
        np.unique(fixed_dofs), F,
        edofMat, iK, jK, H, Hs,
        max_iter=max_iter
    )
    if target_density is None:
        return None

    # Construct Standardized 5-Channel Input Tensor
    ch_domain = np.ones((nely, nelx), dtype=np.float32)

    fixed_grid = np.zeros((nely + 1, nelx + 1), dtype=np.float32)
    for n in fixed_nodes:
        x, y = coords[n]
        fixed_grid[y, x] = 1.0
    ch_bc = 0.25 * (
        fixed_grid[:-1, :-1] + fixed_grid[1:, :-1] +
        fixed_grid[:-1, 1:] + fixed_grid[1:, 1:]
    )

    ch_fx = 0.25 * (
        Fx_grid[:-1, :-1] + Fx_grid[1:, :-1] +
        Fx_grid[:-1, 1:] + Fx_grid[1:, 1:]
    )
    ch_fy = 0.25 * (
        Fy_grid[:-1, :-1] + Fy_grid[1:, :-1] +
        Fy_grid[:-1, 1:] + Fy_grid[1:, 1:]
    )

    ch_vf = np.full((nely, nelx), volfrac, dtype=np.float32)

    input_tensor = np.stack([ch_domain, ch_bc, ch_fx, ch_fy, ch_vf], axis=0).astype(np.float32)
    return input_tensor, target_density.astype(np.float32)

# --- 5. BATCH DATASET GENERATION PIPELINE ---
def generate_dataset(num_samples=2000, output_path="topo_dataset.npz", num_cores=None,
                     seed_start=0, nelx=120, nely=60, penal=3.0, rmin=1.5, max_iter=80):
    if num_cores is None:
        num_cores = max(1, mp.cpu_count() - 1)
        
    print(f"============================================================")
    print(f"  SIMP Dataset Generator (Sosnovik Discrete Truss Pipeline) ")
    print(f"  Target: {num_samples} samples | Domain: {nelx}x{nely} | Cores: {num_cores}")
    print(f"  Volume Fraction: Normal(0.42, 0.08) in [0.22, 0.55]")
    print(f"  Output File: {os.path.abspath(output_path)}")
    print(f"============================================================")
    
    t_start = time.time()
    print("Precomputing mesh geometry and sensitivity filter matrix...")
    global_data = precompute_mesh_and_filter(nelx, nely, rmin)
    t_filter = time.time() - t_start
    print(f"Filter matrix precomputed in {t_filter:.2f}s.")
    
    all_inputs = []
    all_targets = []
    current_seed = seed_start
    batch_size = num_cores * 2
    
    with mp.Pool(processes=num_cores, initializer=init_worker, initargs=(global_data,)) as pool:
        while len(all_inputs) < num_samples:
            seeds = list(range(current_seed, current_seed + batch_size))
            current_seed += batch_size
            
            task_args = [
                (s, nelx, nely, penal, rmin, max_iter, None)
                for s in seeds
            ]
            results = pool.starmap(sample_problem, task_args)
            
            for res in results:
                if res is not None:
                    inp, tgt = res
                    all_inputs.append(inp)
                    all_targets.append(tgt)
                    if len(all_inputs) >= num_samples:
                        break
                        
            elapsed = time.time() - t_start
            rate = len(all_inputs) / elapsed if elapsed > 0 else 0
            remaining = (num_samples - len(all_inputs)) / rate if rate > 0 else 0
            print(f"Progress: {len(all_inputs)}/{num_samples} samples generated "
                  f"({rate:.2f} samples/s, ETA: {remaining/60:.1f} min)")
                  
    inputs_arr = np.stack(all_inputs, axis=0)
    targets_arr = np.stack(all_targets, axis=0)
    
    print(f"Saving dataset arrays: Inputs {inputs_arr.shape}, Targets {targets_arr.shape}...")
    np.savez_compressed(output_path, inputs=inputs_arr, targets=targets_arr)
    total_time = time.time() - t_start
    print(f"Dataset successfully created in {total_time:.2f}s ({total_time/60:.2f} min).")
    print(f"File saved: {os.path.abspath(output_path)}")

def main():
    parser = argparse.ArgumentParser(description="Multi-core SIMP Dataset Generator")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="test_dataset.npz", help="Output .npz path")
    parser.add_argument("--cores", type=int, default=None, help="Number of CPU cores to use")
    parser.add_argument("--seed_start", type=int, default=0, help="Starting random seed")
    parser.add_argument("--nelx", type=int, default=120, help="Mesh width")
    parser.add_argument("--nely", type=int, default=60, help="Mesh height")
    parser.add_argument("--max_iter", type=int, default=80, help="Max SIMP iterations per sample")
    args = parser.parse_args()
    
    generate_dataset(
        num_samples=args.num_samples,
        output_path=args.output,
        num_cores=args.cores,
        seed_start=args.seed_start,
        nelx=args.nelx,
        nely=args.nely,
        max_iter=args.max_iter
    )

if __name__ == '__main__':
    mp.freeze_support()
    main()
