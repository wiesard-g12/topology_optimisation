# Topology Optimization (SIMP) & Deep Learning Pipeline Handover Report

**Project Title:** Real-Time Structural Topology Optimization via Deep Learning  
**Domain Size:** $60 \times 120$ element grid (Height = 60, Width = 120, Total DOFs = 14,762)  
**Phase 1 Target:** Verified, high-throughput SIMP dataset generation (5-channel input tensor $\to$ 1-channel optimal density target)  
**Phase 2 Target:** Deep Learning U-Net Architecture, Physics-Informed Multi-Objective Loss, Training Pipeline, and Real-Time Inference Evaluator  
**Current Status:** Phase 1 & Phase 2 Complete — High-throughput SIMP dataset generator + 4-level U-Net architecture (`TopoUNet`, 6.01M params), composite physics loss, production training pipeline (`train_unet.py`), and real-time evaluation engine (`predict_unet.py`) fully verified and operational.

---

## Quick Reference / Project FAQ (At a Glance)

### Q1: What are we solving?
Traditional Topology Optimization (SIMP) takes minutes per design because it iteratively solves large finite element systems ($\mathbf{K} \mathbf{U} = \mathbf{F}$) 50–100 times per sample. We are building a deep learning pipeline (U-Net) that predicts the optimal structural layout in a **single forward pass (< 10 ms)** directly from boundary conditions, load vectors, and target volume fractions. Phase 1 provides the mathematical foundation and high-throughput dataset generation engine for this system.

### Q2: What did we do?
1. Fixed the inverted $Q_4$ degree-of-freedom ordering (`edofMat`) which previously caused a **35.7% error** in the stiffness matrix and distorted topologies.
2. Fixed the displacement array dimension crash in the FEA compliance calculation.
3. Precomputed sparse sensitivity filter matrices once in shared memory, eliminating over $1.4 \times 10^7$ redundant loops.
4. Upgraded boundary sampling from continuous clamped walls (which created bulky solid blocks) to **discrete Poisson point supports** ($N_x \sim \text{Poisson}(2)$, $N_y \sim \text{Poisson}(1)$, $N_L \sim \text{Poisson}(1)$ with 100× boundary bias), physically forcing the optimizer to generate **slender, triangulated Michell truss bars with hollow voids**.
5. Built an engineering visualizer ([`visualize_data.py`](file:///c:/Users/vedan/OneDrive/Desktop/Study/topo_opto/visualize_data.py)) with unclipped force arrows, degree angles, and support symbols.
6. Conducted a deep codebase cleanup, purging all obsolete prototypes, temporary test files, and old image dumps.

### Q3: What is the structure of the codebase?
The repository has been streamlined to a clean, modular production core:
```
topo_opto/
├── dataset_generator.py             # Phase 1: SIMP dataset generation engine with multiprocessing
├── visualize_data.py                # Phase 1: 3-panel engineering visualizer
├── visualize_evolution.py           # Phase 1: Iteration snapshot & convergence visualizer
├── unet_model.py                    # Phase 2: 4-level TopoUNet architecture (6.01M params)
├── losses.py                        # Phase 2: Physics-informed composite loss
├── dataset.py                       # Phase 2: PyTorch Dataset with vector-preserving reflections
├── train_unet.py                    # Phase 2: Training pipeline
├── predict_unet.py                  # Phase 2: Real-time inference benchmark & evaluator
├── visualize_unet.py                # Phase 2: 5-mode advanced U-Net engineering visualizer
├── interactive_demo.py              # Phase 2: CLI real-time benchmark & sample demo runner
├── demo_server.py                   # Phase 2: Interactive browser web dashboard
├── structural_mechanics.py          # 2D stress recovery, geometric stiffness KG, eigenvalue buckling
├── buckling_optimizer.py            # Active buckling-aware SIMP optimizer
├── stress_buckling_analyzer.py      # 4-panel structural integrity & buckling inspector
├── compare_buckling_optimization.py # Pure compliance vs buckling-aware SIMP comparison
├── checkpoints/best_unet.pth        # Trained TopoUNet production checkpoint
├── topo_dataset_1000.npz            # Verified 1,000-sample SIMP ground-truth dataset
├── handover.md                      # Definitive technical reference & handover report (this file)
├── README.md                        # Quickstart & usage documentation
├── topo_opti_project.pdf            # Original course & project requirements specification
└── .gitignore                       # Standard gitignore (caches, temporary checkpoints, images)
```

### Q4: What all did we use?
- **Theory & Algorithms:** SIMP material penalization ($p=3.0$), 2D 4-node bilinear quadrilateral ($Q_4$) FEA, Sigmund/Bourdin mesh-independency sensitivity filtering ($r_{\min}=1.5$), Optimality Criteria (OC) with 1D bisection ($m=0.2$), Poisson discrete fixture sampling (Sosnovik & Oseledets 2019), 4-level Encoder-Decoder U-Net (`TopoUNet`), 2D Geometric Stiffness ($\mathbf{K}_G$) & generalized eigenvalue buckling.
- **Libraries & Environment:** Python 3.13, NumPy 2.2, SciPy 1.15 (`scipy.sparse`, `scipy.sparse.linalg.spsolve`, `scipy.sparse.linalg.eigs`, `scipy.ndimage`), Matplotlib 3.10, Python `multiprocessing`, PyTorch 2.10.
- **Key Literature:** Andreassen et al. (2011) DTU 88-line benchmark, Aage & Johansen (2013) Python SIMP, Sosnovik & Oseledets (2019) Deep Learning TO, Bendsøe & Sigmund (2003).

### Q5: What was removed in the cleanup and why?
- `images/`: All generated sample images, iteration snapshots, and GIFs removed so the workspace starts clean for the next phase (regenerated on-demand by any visualization script).
- One-off figure scripts (`generate_50iter_buckling_comparison.py`, `generate_side_by_side_50iter.py`, `generate_iteration_visualizations.py`, `generate_unet_diagram.py`, `simulate_optimization_evolution.py`, `train_and_evaluate_pipeline.py`): Removed redundant single-purpose scripts whose capabilities are cleanly covered by the core modules.
- Temporary data & checkpoint files (`topo_dataset.npz`, `results_50iter_sample_000.npz`, `checkpoints/latest_unet.pth`, `checkpoints/training_curves.png`, `checkpoints/training_history.json`): Removed temporary/duplicate artifacts while retaining `topo_dataset_1000.npz` and `checkpoints/best_unet.pth`.
- `__pycache__/`: Cleared compiled `.pyc` bytecode caches.

### Q6: How to run?
- **To generate samples:**
  ```powershell
  python dataset_generator.py --num_samples 1000 --output topo_dataset_1000.npz --cores 11
  ```
- **To visualize samples from the dataset:**
  ```powershell
  python visualize_data.py --dataset topo_dataset_1000.npz --output_dir images --max_samples 5
  ```
- **To launch the real-time interactive web demo:**
  ```powershell
  python demo_server.py --port 5000 --checkpoint checkpoints/best_unet.pth --dataset topo_dataset_1000.npz
  ```

---

## Table of Contents
1. [Executive Summary (What did we do?)](#1-executive-summary-what-did-we-do)
2. [Problem Statement (What are we solving?)](#2-problem-statement-what-are-we-solving)
3. [Codebase Architecture & Structure](#3-codebase-architecture--structure)
4. [Codebase Audit & Cleanup Log (What was removed and why)](#4-codebase-audit--cleanup-log-what-was-removed-and-why)
5. [Methodologies & Technologies Used (What all did we use?)](#5-methodologies--technologies-used-what-all-did-we-use)
6. [Bug Fixes & Technical Breakthroughs](#6-bug-fixes--technical-breakthroughs)
7. [Data Pipeline & 5-Channel U-Net Representation](#7-data-pipeline--5-channel-u-net-representation)
8. [Step-by-Step Execution Guide (How to run?)](#8-step-by-step-execution-guide-how-to-run)
9. [Phase 2 Implementation & Phase 3 Roadmap](#9-phase-2-implementation--phase-3-roadmap)

---

## 1. Executive Summary (What did we do?)

In Phases 1 & 2 of this project, we designed, debugged, and validated an end-to-end synthetic data generation and deep learning topology optimization pipeline:

1. **Diagnosed and Resolved Critical FEM & SIMP Bugs:**
   - Fixed an inverted degree-of-freedom ordering bug in `edofMat` ($Q_4$ bilinear quad elements) that previously caused a 35.7% error in the global stiffness matrix and produced non-physical, distorted topologies.
   - Fixed array dimension crashes in the displacement solver (`U[freedofs]`).
   - Eliminated massive performance bottlenecks by precomputing sparse sensitivity filter matrices once in shared memory rather than rebuilding them inside every iteration.
   - Clamped compliance sensitivities to eliminate `RuntimeWarning: invalid value in sqrt` during the Optimality Criteria (OC) bisection search.

2. **Upgraded from Bulky Continuous Walls to Crisp Discrete Truss Sampling:**
   - Identified why early topologies looked like thick solid blocks: clamping full boundary walls distributes reactions across the entire edge, allowing material to pool monolithically without triangulation.
   - Adopted the discrete boundary sampling strategy from **Sosnovik & Oseledets (2019)**:
     - Poisson-distributed discrete fixtures ($N_x \sim \text{Poisson}(2)$, $N_y \sim \text{Poisson}(1)$, $N_L \sim \text{Poisson}(1)$).
     - 100× boundary probability weighting over interior nodes.
     - Kinematic stability enforcement (guaranteeing non-collinear pin/roller support pairs separated by $\ge 15$ elements to eliminate singular stiffness matrices).
   - This physically forces the SIMP optimizer to build **slender, triangulated Michell truss bars with hollow interior voids**.

3. **Built an Engineering-Grade Visualization & Structural Analysis Suite:**
   - `visualize_data.py` & `visualize_evolution.py`: Reconstructs discrete load application points, draws unclipped engineering load vectors, and renders standard pin/roller/wall support symbols.
   - `structural_mechanics.py`, `buckling_optimizer.py`, `stress_buckling_analyzer.py`, `compare_buckling_optimization.py`: Full 2D plane stress recovery, Von Mises yield verification, geometric stiffness $\mathbf{K}_G$, and active buckling-aware optimization.

4. **Trained & Verified Phase 2 `TopoUNet` Deep Learning Pipeline:**
   - Implemented `unet_model.py`, `losses.py`, `dataset.py`, `train_unet.py`, `predict_unet.py`, `visualize_unet.py`, `interactive_demo.py`, and `demo_server.py`.

5. **Deep Codebase Cleanup & Streamlining:**
   - Purged all one-off scripts, temporary test datasets, cached `.npz` files, generated sample images, and bytecode caches.

---

## 2. Problem Statement (What are we solving?)

### 2.1 The Engineering Challenge
Topology Optimization (TO) is a mathematical method that optimizes material layout within a given design space, for a given set of loads, boundary conditions, and volume constraints, such that the resulting layout minimizes compliance (maximizes structural stiffness).

The standard numerical approach — the **Solid Isotropic Material with Penalization (SIMP)** method — requires solving the linear finite element equation:
$$\mathbf{K}(\mathbf{\rho}) \mathbf{U} = \mathbf{F}$$
iteratively for 50 to 100 steps per design. For fine grids (e.g. $120 \times 60 = 7,200$ elements, 14,762 degrees of freedom), each iteration involves assembling large sparse stiffness matrices and solving linear systems via Cholesky or LU factorization. This takes tens of seconds to minutes for a single geometry, making real-time design exploration, interactive CAD feedback, or edge computing impossible.

### 2.2 The Deep Learning Solution
To eliminate this computational bottleneck, we frame structural topology optimization as an image-to-image regression problem:
$$\hat{\mathbf{y}} = f_\theta(\mathbf{X})$$
where:
- $\mathbf{X} \in \mathbb{R}^{5 \times 60 \times 120}$ encodes the problem physics: solid domain mask, boundary support locations, horizontal force components $F_x$, vertical force components $F_y$, and target volume fraction $V_f$.
- $\hat{\mathbf{y}} \in [0, 1]^{60 \times 120}$ is the predicted optimal material density field.
- $f_\theta$ is a deep Convolutional Neural Network (U-Net) trained on thousands of verified SIMP solutions.

Once trained, the U-Net replaces the 100-step iterative FEM loop with a **single forward pass taking $< 15$ milliseconds** on modern GPU/CPU hardware.

---

## 3. Codebase Architecture & Structure

### Module Responsibilities:

1. **Data Generation & SIMP Core (`dataset_generator.py`, `visualize_data.py`, `visualize_evolution.py`):**
   - `dataset_generator.py`: Vectorized $Q_4$ SIMP FEA solver, precomputed sensitivity filter $\mathbf{H}$, Sosnovik Poisson discrete support/load sampler, and multi-core dataset generator.
   - `visualize_data.py`: 3-panel engineering visualizer for `.npz` datasets (creates `images/` on-demand).
   - `visualize_evolution.py`: Tracks and visualizes SIMP optimization progression across iterations.

2. **Phase 2 Deep Learning U-Net (`unet_model.py`, `losses.py`, `dataset.py`, `train_unet.py`, `predict_unet.py`, `visualize_unet.py`, `interactive_demo.py`, `demo_server.py`):**
   - `unet_model.py`: 4-level encoder-decoder `TopoUNet` with automatic reflection padding $(60, 120) \to (64, 128)$ and cropping.
   - `losses.py`: `TopoCompositeLoss` combining BCE, Soft Dice, Volume Fraction MSE, and scheduled Binarization penalty.
   - `dataset.py`: `TopoDataset` with physics-consistent horizontal/vertical reflections ($F_x \to -F_x$, $F_y \to -F_y$).
   - `train_unet.py` & `predict_unet.py`: End-to-end training and evaluation pipelines.
   - `visualize_unet.py`: 5-mode diagnostic suite (`comparison`, `sweep_vf`, `sweep_threshold`, `stress`, `batch`).
   - `interactive_demo.py` & `demo_server.py`: CLI and zero-dependency browser web application for real-time topology prediction.

3. **Structural Mechanics & Active Buckling (`structural_mechanics.py`, `buckling_optimizer.py`, `stress_buckling_analyzer.py`, `compare_buckling_optimization.py`):**
   - `structural_mechanics.py`: $Q_4$ plane stress recovery, Von Mises stress, geometric stiffness matrix $\mathbf{K}_G$, and shift-invert eigenvalue buckling solver.
   - `buckling_optimizer.py`: Hybrid compliance + modal strain energy buckling-aware SIMP optimizer.
   - `stress_buckling_analyzer.py` & `compare_buckling_optimization.py`: Engineering health certificates and head-to-head compliance vs. buckling comparisons.

---

## 4. Codebase Audit & Cleanup Log (What was removed and why)

All redundant, prototype, one-off, and transient artifact files were purged to maintain a clean workspace:

| Item Removed | Type | Rationale |
| :--- | :--- | :--- |
| **`images/`** | Directory | All generated sample plots, comparisons, and GIFs were cleared for a clean slate moving into the next phase. |
| **`generate_*.py`, `simulate_*.py`, `train_and_evaluate_pipeline.py`** | Scripts | One-off figure generators and polling wrappers removed; all core functionality resides in the modular scripts. |
| **`topo_dataset.npz`, `results_50iter_sample_000.npz`** | Temp Data | Temporary small/cached `.npz` archives removed; production `topo_dataset_1000.npz` retained. |
| **`checkpoints/latest_unet.pth`, `training_curves.png`, `training_history.json`** | Temp Logs | Redundant checkpoint and training log artifacts removed; production `checkpoints/best_unet.pth` retained. |
| **`__pycache__/`** | Cache | Python compiled `.pyc` bytecode directories cleared. |

---

## 5. Methodologies & Technologies Used (What all did we use?)

### 5.1 Theoretical & Algorithmic Foundations
1. **SIMP Framework (Solid Isotropic Material with Penalization):**
   - Interpolates Young's modulus between void ($E_{\min} = 10^{-9}$) and solid material ($E_0 = 1.0$) using penalization power $p = 3.0$:
     $$E(\rho_e) = E_{\min} + \rho_e^p (E_0 - E_{\min})$$
2. **2D Finite Element Analysis (FEA):**
   - 4-node bilinear isoparametric quadrilateral elements ($Q_4$) with 2 degrees of freedom per node ($u, v$).
   - Total nodes: $(120 + 1) \times (60 + 1) = 7,381$.
   - Total DOFs: $2 \times 7,381 = 14,762$.
3. **Mesh-Independency Sensitivity Filter (Sigmund / Bourdin):**
   - Convolves raw element compliance sensitivities with a conical distance-decay kernel of radius $r_{\min} = 1.5$ elements to eliminate checkerboard patterns and mesh dependency:
     $$\widehat{\frac{\partial c}{\partial \rho_e}} = \frac{1}{\max(10^{-3}, \rho_e) \sum_{i \in N_e} H_{ei}} \sum_{i \in N_e} H_{ei} \rho_i \frac{\partial c}{\partial \rho_i}, \quad H_{ei} = \max(0, r_{\min} - \text{dist}(e, i))$$
4. **Optimality Criteria (OC) Method with Bisection:**
   - Updates element densities subject to move limit $m = 0.2$ and volume fraction target $V_f$, solving for Lagrange multiplier $\lambda$ via 1D bisection.
5. **Poisson Discrete Fixture Sampling (Sosnovik & Oseledets, 2019):**
   - Samples discrete pin/roller support counts from Poisson distributions ($N_x \sim \text{Poisson}(2)$, $N_y \sim \text{Poisson}(1)$, $N_L \sim \text{Poisson}(1)$) with 100× boundary probability bias.

### 5.2 Software Libraries & Environments
- **Python 3.13** (64-bit on Windows).
- **NumPy 2.2**: Vectorized matrix transformations, Kronecker products, element DOF mappings.
- **SciPy 1.15**:
  - `scipy.sparse.coo_matrix`, `csr_matrix`, `csc_matrix` for sparse stiffness matrix assembly.
  - `scipy.sparse.linalg.spsolve` for direct sparse linear system solution.
  - `scipy.ndimage.label`, `center_of_mass` for geometric feature clustering in visualizer.
- **Matplotlib 3.10**: Publication-quality vector rendering, engineering annotations, and custom patch generation.
- **Python `multiprocessing`**: Parallel CPU worker pooling with global memory pre-initialization.
- **PyTorch 2.10**: Environment verified and ready for Phase 2 deep learning training.

### 5.3 Key Research Literature Referenced
- **Andreassen et al. (2011):** *"Efficient topology optimization in MATLAB using 88 lines of code"*, Struct Multidisc Optim.
- **Aage & Johansen (2013):** *"Topology Optimization using PETSc / Python"*, DTU Technical Report.
- **Sosnovik & Oseledets (2019):** *"Neural Networks for Topology Optimization"*, Skoltech / Russian Academy of Sciences.
- **Bendsøe & Sigmund (2003):** *"Topology Optimization: Theory, Methods, and Applications"*, Springer.

---

## 6. Bug Fixes & Technical Breakthroughs

| Issue / Bug | Root Cause | Impact | Resolution |
| :--- | :--- | :--- | :--- |
| **1. Permuted `edofMat` Ordering** | Element node DOFs were assigned in non-standard order (Top-Left assigned to slots 0 & 1 instead of Bottom-Left). | 35.7% error in global stiffness matrix $\mathbf{K}$; inverted shear and bending terms; caused non-physical asymmetric collapses. | Adopted standard DTU counter-clockwise ordering: `[2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1]`. |
| **2. Displacement Shape Crash** | `U` was sized as `(ndof, 1)` causing `U[edofMat]` to have shape `(1200, 8, 1)` which failed matrix multiplication with `KE (8, 8)`. | Fatal runtime `ValueError`. | Maintained `U` as a 1D vector `(ndof,)` enabling seamless vectorized element strain energy evaluation. |
| **3. Filter Neighborhood Loop Bottleneck** | Filter matrix $\mathbf{H}$ was rebuilt inside every iteration of every sample. | Generating 2,000 samples would require $> 1.4 \times 10^7$ Python loops, taking hours. | Precomputed $\mathbf{H}$ and $\mathbf{H}_s$ once into sparse CSR format in shared memory before launching worker processes. |
| **4. Bulky Monolithic Topologies** | Old generator clamped entire boundary walls (all 61 nodes). | Boundary absorbs moments across entire height; optimizer fills a solid block without needing diagonal bracing. | Switched to discrete point supports (2–3 boundary pins/rollers). Point supports force triangular truss bars and hollow voids. |
| **5. Rigid Body Instability in Random Sampling** | Unconstrained random selection of support DOFs produced mechanisms (free rotation or sliding). | Singular stiffness matrix; `spsolve` failed or produced infinite displacements. | Added kinematic stability verification ensuring non-collinear support points separated by at least 15 elements. |
| **6. OC Sensitivity Clamping** | Rounding noise occasionally made compliance sensitivities $dc > 0$. | Caused `RuntimeWarning: invalid value in sqrt` in $x \cdot \sqrt{-dc / dv / \lambda}$. | Clamped $dc = \min(dc, -10^{-12})$ ensuring strict non-positivity. |

---

## 7. Data Pipeline & 5-Channel U-Net Representation

Every training sample consists of a **5-channel input tensor** $\mathbf{X} \in \mathbb{R}^{5 \times 60 \times 120}$ and a **ground-truth target density** $\mathbf{Y} \in [0, 1]^{60 \times 120}$:

```
Input Tensor X (5, 60, 120)
├── Channel 0: Solid Domain Mask     [1.0 = solid design space, 0.0 = void / keepout]
├── Channel 1: Boundary Conditions   [1.0 = fixed support node mapped to element cells, 0.0 = free]
├── Channel 2: Horizontal Force Fx   [Nodal Fx force component distributed to elements]
├── Channel 3: Vertical Force Fy     [Nodal Fy force component distributed to elements]
└── Channel 4: Target Volume Frac    [Uniform plane with value = target Vf in [0.22, 0.55]]

                     │
                     ▼ Deep Convolutional U-Net (Phase 2)
                     │
Target Density Y (60, 120)
└── Optimal SIMP Material Density    [1.0 = solid structural member, 0.0 = void]
```

### Channel Mapping Mechanics:
Nodal boundary conditions and loads exist on grid vertices $(61 \times 121)$. To map them into the element-centered tensor $(60 \times 120)$, vertex quantities are averaged over the 4 surrounding nodes of each element:
$$\text{Cell}(i, j) = \frac{1}{4} \left(V_{i, j} + V_{i+1, j} + V_{i, j+1} + V_{i+1, j+1}\right)$$

---

## 8. Step-by-Step Execution Guide (How to run?)

### 8.1 Generating the Dataset (`dataset_generator.py`)

#### Generate a Small Test Batch (e.g. 5 samples):
```powershell
python dataset_generator.py --num_samples 5 --output test_dataset.npz --cores 4
```

#### Generate the Full 2,000-Sample Dataset:
```powershell
python dataset_generator.py --num_samples 2000 --output topo_dataset.npz --cores 11
```

#### CLI Options:
| Flag | Default | Description |
|---|---|---|
| `--num_samples` | `10` | Total number of valid training samples to generate. |
| `--output` | `topo_dataset.npz` | Path to save the compressed `.npz` archive. |
| `--cores` | `cpu_count()` | Number of parallel CPU processes to utilize. |
| `--max_iter` | `80` | Maximum SIMP iterations per sample (standard: 80). |
| `--rmin` | `1.5` | Filter radius in elements (controls minimum feature size). |
| `--penal` | `3.0` | SIMP penalization power $p$. |

---

### 8.2 Visualizing Datasets (`visualize_data.py`)

To inspect any generated dataset and produce 3-panel engineering graphics with boundary symbols and unclipped load arrows:

#### Visualize Generated Dataset:
```powershell
python visualize_data.py --dataset topo_dataset.npz --output_dir images --max_samples 10
```

*Note: The script automatically creates the `images/` directory if it does not already exist.*

#### CLI Options:
| Flag | Default | Description |
|---|---|---|
| `--dataset` | `topo_dataset.npz` | Dataset file to read. |
| `--output_dir` | `images` | Folder where `.png` figures are saved. |
| `--max_samples` | `10` | Maximum number of sample figures to render. |

---

## 9. Phase 2 Implementation & Phase 3 Roadmap

### 9.1 Phase 2 Architecture: `TopoUNet` (`unet_model.py`)
- **Inputs:** 5-channel physical state tensor $\mathbf{X} \in \mathbb{R}^{B \times 5 \times 60 \times 120}$
  - Channel 0: Domain Solid Mask
  - Channel 1: Boundary Support Fixtures (pins / rollers)
  - Channel 2: Horizontal Force Field $F_x$
  - Channel 3: Vertical Force Field $F_y$
  - Channel 4: Target Volume Fraction $V_f$
- **Spatial Dimension Management:** Forward pass auto-pads $(60, 120) \to (64, 128)$, divides cleanly by 2 across 4 downsampling stages ($32 \times 64 \to 16 \times 32 \to 8 \times 16 \to 4 \times 8$), upsamples back to $(64, 128)$ via skip connections, and crops back to exact $(60, 120)$.
- **Backbone Channels:** $5 \to 64 \to 128 \to 256 \to 512 \to 256 \to 128 \to 64 \to 1$.
- **Parameter Count:** 6,016,641 parameters.
- **Activation:** `Sigmoid` output head constraining density $\hat{\mathbf{y}} \in [0, 1]^{60 \times 120}$.

### 9.2 Phase 2 Physics-Informed Composite Loss (`losses.py`)
Multi-objective objective function balancing pixel-level fidelity, structural continuity, volume constraint, and boundary sharpness:
$$\mathcal{L} = w_{\text{BCE}} \mathcal{L}_{\text{BCE}}(\hat{\mathbf{y}}, \mathbf{y}) + w_{\text{Dice}} \mathcal{L}_{\text{Dice}}(\hat{\mathbf{y}}, \mathbf{y}) + \alpha \cdot \left(\text{mean}(\hat{\mathbf{y}}) - V_f\right)^2 + \beta(t) \cdot \frac{1}{N_e} \sum_{e} 4 \hat{y}_e (1 - \hat{y}_e)$$
- **$\mathcal{L}_{\text{BCE}}$:** Binary cross-entropy matching ground truth material distribution.
- **$\mathcal{L}_{\text{Dice}}$:** Soft Dice overlap loss penalizing topological member disconnections.
- **Volume Penalty ($\alpha = 10.0$):** Strictly penalizes deviation between predicted volume fraction and requested target $V_f$.
- **Binarization Regularizer ($\beta = 10^{-3}$):** Scheduled ramp-up after epoch 5 driving intermediate gray voxels to crisp 0 or 1 boundaries.

### 9.3 Phase 2 Production Training Pipeline (`train_unet.py`)
- AdamW optimizer with Cosine Annealing learning rate schedule.
- Vector-preserving physics reflections for data augmentation ($x \to -x \implies F_x \to -F_x$; $y \to -y \implies F_y \to -F_y$).
- Checkpoints best model based on validation loss / Dice score.
- Auto-exports loss curves (`training_curves.png`) and JSON history log (`training_history.json`).

### 9.4 Phase 2 Real-Time Inference & Evaluation Engine (`predict_unet.py`)
- Instantaneous single-pass inference ($< 15\text{ ms}$ on GPU, $\approx 35.8\text{ ms}$ on CPU).
- Renders publication-grade 5-panel engineering comparisons:
  1. Boundary Conditions & Applied Loads with magnitude and angle callouts
  2. Ground Truth SIMP (80 iterations)
  3. Continuous U-Net Predicted Density $\hat{\mathbf{y}}$
  4. Binarized Final Topology ($\hat{\mathbf{y}} \geq 0.5$)
  5. Absolute Error Heatmap $|\hat{\mathbf{y}} - \mathbf{y}^*|$
- Computes Dice similarity coefficient, MSE, MAE, volume fraction discrepancy, and binarization index.

### 9.5 Interactive Real-Time Demo Engine (Web & CLI)
- **Web Demo (`demo_server.py`):** Standalone web dashboard utilizing Python's built-in `http.server` (zero extra packages). Offers instant structural topology prediction, slider-based $V_f$ adjustment, threshold tuning, and interactive case presets (`cantilever`, `mbb`, `bridge`, `crane`).
- **CLI Demo Runner (`interactive_demo.py`):** Standalone script to test standard structural benchmarks or individual dataset samples and generate high-resolution PNG outputs with performance metrics.

### 9.6 Comprehensive U-Net Visualization Suite (`visualize_unet.py`)
A dedicated multi-mode diagnostic and visualization utility featuring 5 distinct operational modes:
1. `--mode comparison`: 6-panel diagnostic plot (BCs/Loads, Ground Truth SIMP, Continuous U-Net density, Binarized truss structure, Absolute error heatmap, and density histogram).
2. `--mode sweep_vf`: Evaluates model response across a parametric volume fraction sweep ($V_f \in [0.25, 0.35, 0.45, 0.55]$) for a chosen benchmark case.
3. `--mode sweep_threshold`: Sweeps binarization thresholds $\tau \in [0.1, 0.9]$, plotting achieved volume fraction and Dice overlap curve to determine the optimal cutoff.
4. `--mode stress`: Integrates 2D plane stress recovery (`structural_mechanics.py`) to evaluate Von Mises stress fields, peak deflection, and yielding Factor of Safety (FoS) on both ground truth and predicted topologies.
5. `--mode batch`: Generates a 4-quadrant statistical scorecard over a test batch (Dice score histogram, target vs predicted $V_f$ parity, MSE distribution, and inference latency histogram).

### 9.7 Phase 3 Roadmap (Active Structural Health & Buckling Integration)
- Couple trained U-Net with `structural_mechanics.py` for instantaneous closed-loop stress & buckling verification.
- Implement fine-tuning against geometric stiffness $\mathbf{K}_G$ modal strain energy.
- Real-time interactive CAD GUI / Web demo for interactive force dragging and live topology response.

---
*Report updated and verified for Phase 1 & Phase 2 completion.*

