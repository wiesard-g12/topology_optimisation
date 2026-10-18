# Real-Time Structural Topology Optimization via Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance pipeline for generating structural topology optimization datasets using the **Solid Isotropic Material with Penalization (SIMP)** method, tailored for training deep learning models (e.g., U-Net) to predict optimal topologies in real-time (< 30 ms).

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
├── dataset_generator.py            # Phase 1: SIMP dataset generator with multiprocessing & auto-save
├── visualize_data.py               # Phase 1: 3-panel engineering visualizer (BCs, loads, density)
├── visualize_evolution.py          # Phase 1: Multi-panel & animated GIF visualizer for SIMP iterations
├── unet_model.py                   # Phase 2: 4-level U-Net architecture (TopoUNet, 6.01M params)
├── losses.py                       # Phase 2: Physics-informed composite loss (BCE + Dice + Vf + Bin)
├── dataset.py                      # Phase 2: PyTorch Dataset with vector-preserving reflections
├── train_unet.py                   # Phase 2: End-to-end training pipeline with validation & scheduler
├── predict_unet.py                 # Phase 2: Real-time inference benchmark & 5-panel visualizer
├── visualize_unet.py               # Phase 2: 5-mode advanced U-Net diagnostic & stress visualizer
├── interactive_demo.py             # Phase 2: CLI real-time benchmark & dataset sample runner
├── demo_server.py                  # Phase 2: Browser-based interactive real-time web application
├── structural_mechanics.py         # 2D stress recovery, geometric stiffness KG, and eigenvalue buckling
├── buckling_optimizer.py           # Active buckling-aware SIMP topology optimizer
├── stress_buckling_analyzer.py     # 4-panel structural integrity & buckling verification inspector
├── compare_buckling_optimization.py# Head-to-head comparison: pure compliance vs buckling-aware SIMP
├── checkpoints/best_unet.pth       # Trained TopoUNet production weights
├── topo_dataset_1000.npz           # 1,000-sample verified SIMP ground-truth dataset
├── handover.md                     # In-depth technical reference and mathematical derivations
├── topo_opti_project.pdf           # Project and course specification document
└── .gitignore                      # Ignored cache files, temporary logs, and transient outputs
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

### 6. Phase 2: Deep Learning U-Net Training & Real-Time Inference (< 15 ms)

Train the 4-level physics-guided U-Net to predict optimal material layout directly in a single forward pass:
```bash
# 1. Train TopoUNet on generated SIMP dataset
python train_unet.py --dataset topo_dataset_1000.npz --epochs 50 --batch_size 16 --lr 1e-3 --save_dir checkpoints

# 2. Benchmark inference latency & render 5-panel engineering comparisons
python predict_unet.py --checkpoint checkpoints/best_unet.pth --dataset topo_dataset_1000.npz --output_dir images --num_viz 5
```
- **Inputs:** 5-channel physics tensor `(B, 5, 60, 120)`: Solid Domain, Discrete BCs, $F_x$, $F_y$, Target $V_f$.
- **Outputs:** Continuous density field $\hat{\mathbf{y}} \in [0, 1]^{60 \times 120}$.
- **Loss formulation:** Multi-objective loss $\mathcal{L} = \mathcal{L}_{\text{BCE}} + 0.5 \mathcal{L}_{\text{Dice}} + 10.0 (\text{mean}(\hat{\mathbf{y}}) - V_f)^2 + \beta \sum \hat{y}_e (1 - \hat{y}_e)$.
- **Speed:** Replaces iterative FEM loop (~60 seconds) with an instantaneous forward pass taking **$< 15\text{ ms}$** on GPU and **$< 36\text{ ms}$** on CPU.

### 7. Interactive Real-Time Demo (Web & CLI)

Experience real-time U-Net topology prediction with live parameter controls:

```bash
# Web-based interactive GUI (zero external web dependencies, pure Python standard library)
python demo_server.py --port 5000 --checkpoint checkpoints/best_unet.pth --dataset topo_dataset_1000.npz
# Open http://localhost:5000 in your browser!

# CLI interactive runner for standard engineering cases:
python interactive_demo.py --case cantilever --vf 0.40 --checkpoint checkpoints/best_unet.pth
python interactive_demo.py --case mbb --vf 0.50 --checkpoint checkpoints/best_unet.pth
python interactive_demo.py --case bridge --vf 0.35 --checkpoint checkpoints/best_unet.pth
python interactive_demo.py --case crane --vf 0.45 --checkpoint checkpoints/best_unet.pth
```

### 8. Comprehensive U-Net Visualization Tool (`visualize_unet.py`)

A multi-mode visualization suite for in-depth engineering analysis:

```bash
# Mode 1: 6-Panel Engineering Comparison (Loads, SIMP Ground Truth, U-Net, Binarized, Error, Histogram)
python visualize_unet.py --mode comparison --sample_idx 1 --checkpoint checkpoints/best_unet.pth --dataset topo_dataset_1000.npz

# Mode 2: Real-time Volume Fraction Parameter Sweep (Vf = 0.25, 0.35, 0.45, 0.55)
python visualize_unet.py --mode sweep_vf --case cantilever --checkpoint checkpoints/best_unet.pth

# Mode 3: Threshold Sensitivity Curve (τ in [0.1, 0.9])
python visualize_unet.py --mode sweep_threshold --sample_idx 1 --checkpoint checkpoints/best_unet.pth --dataset topo_dataset_1000.npz

# Mode 4: Plane Stress Recovery & FoS Inspection (Von Mises stress, max deflection, factor of safety)
python visualize_unet.py --mode stress --sample_idx 1 --material steel --checkpoint checkpoints/best_unet.pth --dataset topo_dataset_1000.npz

# Mode 5: Batch Statistical Scorecard (Dice distribution, Vf parity, MSE, latency histogram)
python visualize_unet.py --mode batch --checkpoint checkpoints/best_unet.pth --dataset topo_dataset_1000.npz --num_eval 200
```

---

## Detailed Documentation

For full mathematical formulations, FEA derivations, benchmarking results, and Phase 2 deep learning architecture plans, refer to [`handover.md`](handover.md).

---


- Team Members:  
Aditya Amodkar (CE23BTECH11004). 
Ayush Dhakad (CE23BTECH11010). 
Pushpraj Singh Khichi (CE23BTECH11043). 
Vedant Gupta (CE23BTECH11059).  

- GitHub: [@wiesard-g12](https://github.com/wiesard-g12)
