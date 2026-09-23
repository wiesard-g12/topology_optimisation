# Real-Time Structural Topology Optimization via Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance pipeline for generating structural topology optimization datasets using the **Solid Isotropic Material with Penalization (SIMP)** method, tailored for training deep learning models (e.g., U-Net) to predict optimal topologies in real-time (< 10 ms).

![Sample Visualization](images/sample_000.png)

---

## Overview

Traditional Topology Optimization (SIMP) iteratively solves large finite element systems ($\mathbf{K} \mathbf{U} = \mathbf{F}$) 50–100 times per design, taking minutes to converge. This project builds a deep learning foundation to predict structural density distributions directly from:
- Boundary conditions (fixed $X$ and $Y$ displacements)
- Point loads (magnitude and angle)
- Target volume fractions ($V_f$)

### Key Highlights
- **Domain Size:** $60 \times 120$ bilinear quadrilateral element grid ($Q_4$), 14,762 degrees of freedom.
- **Physics-Informed Poisson Fixture Sampling:** Discrete Poisson point supports ($N_x \sim \text{Poisson}(2)$, $N_y \sim \text{Poisson}(1)$, $N_L \sim \text{Poisson}(1)$ with boundary bias) physically driving the optimizer toward slender, triangulated Michell truss bars with internal voids rather than trivial solid blocks.
- **Optimized FEA Solver:** Correct $Q_4$ degree-of-freedom ordering, sparse matrix pre-assembly, and Sigmund/Bourdin mesh-independency sensitivity filtering precomputed in shared memory.
- **Multi-Core High-Throughput Generator:** Parallelized SIMP generation utilizing Python `multiprocessing`.

---

## Repository Structure

```
topo_opto/
├── dataset_generator.py          # Production SIMP dataset generation engine with multiprocessing
├── structural_mechanics.py       # 2D stress recovery, geometric stiffness KG, and eigenvalue buckling
├── stress_buckling_analyzer.py   # 4-panel structural integrity & buckling verification inspector
├── visualize_data.py             # 3-panel engineering visualizer (BCs, loads, and final density)
├── visualize_evolution.py        # Multi-panel & animated GIF visualizer for optimization iterations
├── generate_iteration_visualizations.py # 3-panel snapshots for iterations 10, 20... 80
├── handover.md                   # In-depth technical reference and mathematical derivations
├── topo_opti_project.pdf         # Project and course specification document
├── images/
│   ├── sample_000.png            # Sample engineering visualization output
│   ├── structural_report_sample_000_steel.png    # 4-panel structural health report (Steel)
│   ├── structural_report_sample_000_aluminum.png # 4-panel structural health report (Aluminum)
│   ├── evolution_cantilever.png  # Iteration snapshots (0, 10, 20... 80)
│   └── evolution_cantilever.gif  # Animated optimization progression
└── .gitignore                    # Ignored cache files, checkpoints, and data archives
```

---

## Quickstart

### Prerequisites

Install dependencies:
```bash
pip install numpy scipy matplotlib pillow torch
```

### 1. Generate Dataset

Generate SIMP samples using multi-core parallelization:
```bash
python dataset_generator.py --num_samples 2000 --output topo_dataset.npz --cores 8
```

### 2. Visualize Samples

Inspect generated samples with boundary conditions, load vectors, and density fields:
```bash
python visualize_data.py --dataset topo_dataset.npz --num_visualizations 5 --output_dir images
```

### 3. Visualize Optimization Evolution (Every 10 Iterations)

Watch how SIMP iteratively removes low-strain material and forms triangulated Michell truss bars:
```bash
# Cantilever truss evolution
python visualize_evolution.py --case cantilever --interval 10

# MBB beam evolution
python visualize_evolution.py --case mbb --interval 10
```
This generates high-resolution snapshot grids (at Iterations 0, 10, 20, ..., 80) with convergence curves and animated frame-by-frame GIFs.

### 4. Real-World Structural Integrity & In-Plane Buckling Inspection

Verify that the optimized truss architecture sustains real forces without yielding or in-plane buckling:
```bash
# Structural Steel inspection (1,500 N working load)
python stress_buckling_analyzer.py --sample_id 0 --material steel --force 1500

# Aerospace Aluminum 6061-T6 inspection
python stress_buckling_analyzer.py --sample_id 0 --material aluminum --force 1500

# 3D-Printed PETG / Carbon PLA inspection
python stress_buckling_analyzer.py --sample_id 0 --material petg --force 300
```
This solves the 2D stress tensor and the generalized eigenvalue problem $(\mathbf{K} - \lambda_1 \mathbf{K}_G)\mathbf{\phi}_1 = \mathbf{0}$, generating a 4-panel engineering health certificate with:
- Von Mises Stress Heatmap (MPa) with peak stress callouts
- 2D In-Plane Buckling Mode Shape & Critical Load Multiplier $\lambda_1$
- Yielding Factor of Safety (FoS) and definitive PASS / FAIL structural verdicts.

### 5. Active Buckling-Aware Topology Optimization (25-Iteration Budget)

Actively reinforce compression struts against in-plane buckling within a fast 25-iteration computational budget:
```bash
# Compare Pure Compliance SIMP vs Buckling-Aware SIMP on Structural Steel
python compare_buckling_optimization.py --sample-index 0 --material steel --force 1500 --buckle-weight 0.45

# Compare on Aluminum 6061-T6 with custom volume fraction
python compare_buckling_optimization.py --sample-index 1 --material aluminum --force 1000 --volfrac 0.3 --buckle-weight 0.50
```
- **Warm-start schedule:** Iterations 1–14 run standard compliance SIMP to discover macro-truss topology; iterations 15–25 activate modal strain energy sensitivities to thicken compression members against geometric stiffness $\mathbf{K}_G$.
- **Speed:** Solves in $< 2.5\text{ seconds}$ per design.
- **Output:** Automatically exports side-by-side topology comparison, material redistribution maps ($\Delta \rho$), and structural performance metrics.

---

## Detailed Documentation

For full mathematical formulations, FEA derivations, benchmarking results, and Phase 2 deep learning architecture plans, refer to [`handover.md`](handover.md).

---

## Author

- **Vedant Gupta** ([Vedantgupta070@gmail.com](mailto:Vedantgupta070@gmail.com))
- GitHub: [@wiesard-g12](https://github.com/wiesard-g12)
