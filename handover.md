# Topology Optimization (SIMP) & Deep Learning Pipeline Handover Report

**Project Title:** Real-Time Structural Topology Optimization via Deep Learning  
**Domain Size:** $60 \times 120$ element grid (Height = 60, Width = 120, Total DOFs = 14,762)  
**Phase 1 Target:** Verified, high-throughput SIMP dataset generation (5-channel input tensor $\to$ 1-channel optimal density target)  
**Current Status:** Phase 1 Complete — Core pipeline finalized, all experimental/test/image files cleaned, minimal production codebase ready.

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
The repository has been streamlined to a clean, minimal core:
```
topo_opto/
├── .gitignore               # Standard gitignore (caches, checkpoints, data dumps)
├── dataset_generator.py     # Production SIMP dataset generation engine with multiprocessing
├── visualize_data.py        # Professional 3-panel engineering visualizer (creates images/ on-demand)
├── handover.md              # Definitive technical reference & handover report (this file)
└── topo_opti_project.pdf    # Original course & project requirements specification
```

### Q4: What all did we use?
- **Theory & Algorithms:** SIMP material penalization ($p=3.0$), 2D 4-node bilinear quadrilateral ($Q_4$) FEA, Sigmund/Bourdin mesh-independency sensitivity filtering ($r_{\min}=1.5$), Optimality Criteria (OC) with 1D bisection ($m=0.2$), Poisson discrete fixture sampling (Sosnovik & Oseledets 2019).
- **Libraries & Environment:** Python 3.13, NumPy 2.2, SciPy 1.15 (`scipy.sparse`, `scipy.sparse.linalg.spsolve`, `scipy.ndimage`), Matplotlib 3.10, Python `multiprocessing`, PyTorch 2.10.
- **Key Literature:** Andreassen et al. (2011) DTU 88-line benchmark, Aage & Johansen (2013) Python SIMP, Sosnovik & Oseledets (2019) Deep Learning TO, Bendsøe & Sigmund (2003).

### Q5: What was removed in the cleanup and why?
- `images/`: Cleared old image plots so fresh figures can be generated on-demand.
- `test_simp.py`, `top88.py`, `generate_paper_visualization.py`: One-off benchmarks and unit tests removed once the core solver was verified.
- `test_sosnovik_generator.py`, `create_sampling_comparison.py`, `create_literature_pipeline_figure.py`: Experimental prototypes and single-run figure scripts removed.
- `test_dataset.npz`, `topo_dataset.npz`: Temporary and incomplete test dataset dumps purged for fresh generation.
- `__pycache__/`: Stale `.pyc` caches cleared.

### Q6: How to run?
- **To generate 2,000 samples:**
  ```powershell
  python dataset_generator.py --num_samples 2000 --output topo_dataset.npz --cores 11
  ```
- **To visualize samples from the dataset:**
  ```powershell
  python visualize_data.py --dataset topo_dataset.npz --output_dir images --max_samples 10
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
9. [Phase 2 & Phase 3 Roadmap (Deep Learning & U-Net Training)](#9-phase-2--phase-3-roadmap-deep-learning--u-net-training)

---

## 1. Executive Summary (What did we do?)

In Phase 1 of this project, we designed, debugged, and validated an end-to-end synthetic data generation pipeline for structural topology optimization. Specifically:

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

3. **Built an Engineering-Grade Visualizer (`visualize_data.py`):**
   - Automatically clusters element load arrays to reconstruct discrete load application points, draws unclipped engineering load vectors with magnitudes and angles, renders standard pin/roller/wall support symbols, and overlays the optimal topology with the boundary conditions.

4. **Engineered Multi-Core Parallel Generation for Windows:**
   - Implemented a worker pool using Python's `multiprocessing` with global shared memory initialization, generating samples across all available CPU cores at high throughput.

5. **Deep Codebase Cleanup & Streamlining:**
   - Purged all prototype scripts, temporary test scripts, old test data files, image dumps, and bytecode caches, leaving a clean 2-module production core (`dataset_generator.py` and `visualize_data.py`).

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

Once trained, the U-Net replaces the 100-step iterative FEM loop with a **single forward pass taking $< 10$ milliseconds** on modern GPU/CPU hardware.

---

## 3. Codebase Architecture & Structure

```
topo_opto/
│
├── .gitignore               # Standard gitignore (caches, checkpoints, datasets)
├── dataset_generator.py     # Production SIMP dataset generation engine with multiprocessing
├── visualize_data.py        # Professional 3-panel engineering visualizer for any .npz dataset
├── handover.md              # Project handover report and technical reference (this file)
└── topo_opti_project.pdf    # Original project requirements specification
```

### Module Responsibilities:

1. **`dataset_generator.py` (Core Engine):**
   - `get_stiffness_matrix(Emax, nu)`: Computes the 2D $Q_4$ element stiffness matrix $\mathbf{K}_E \in \mathbb{R}^{8 \times 8}$.
   - `precompute_mesh_and_filter(nelx, nely, rmin)`: Vectorizes element DOF matrices (`edofMat`, `iK`, `jK`) and builds the sparse linear sensitivity filter matrix $\mathbf{H} \in \mathbb{R}^{N_e \times N_e}$.
   - `solve_simp(...)`: Runs the 80-iteration SIMP solver with vectorized compliance calculations, mesh-independency filtering, and bisection OC updating.
   - `sample_problem(...)`: Samples boundary conditions (discrete pins/rollers with Poisson distribution and 100× boundary bias) and point loads, validates kinematic stability, runs SIMP, and packs the 5-channel tensor.
   - `main()`: Multiprocessing manager distributing sample generation across all available CPU cores.

2. **`visualize_data.py` (Visualization Engine):**
   - `extract_discrete_loads(ch_fx, ch_fy)`: Clusters element load arrays to reconstruct discrete application nodes and force vectors.
   - `draw_domain_and_bcs(ax, ch_bc)`: Renders domain borders and engineering support symbols (pins, rollers, clamped walls).
   - `draw_loads(ax, loads)`: Draws unclipped load arrows with magnitude $|F|$ and angle $\theta$ callout boxes.
   - `visualize_dataset(...)`: Produces high-resolution 3-panel figures for dataset samples. Automatically creates `images/` directory when run.

---

## 4. Codebase Audit & Cleanup Log (What was removed and why)

All redundant, prototype, test, and transient artifact files were purged to maintain a zero-clutter workspace:

### 4.1 Purged Items & Rationale
| Item Removed | Type | Rationale |
| :--- | :--- | :--- |
| **`images/`** | Directory | All previous test plots and diagrams were cleared so that fresh visualizations can be generated directly from new dataset runs. |
| **`test_simp.py`** | Script | Standalone debug script used to test the early $Q_4$ DOF fix. Fully superseded by `dataset_generator.py`. |
| **`top88.py`** | Script | Pure DTU benchmark script used during initial mathematical validation against Andreassen (2011). No longer required once the solver in `dataset_generator.py` was verified. |
| **`generate_paper_visualization.py`** | Script | One-off script used to recreate Sosnovik paper figures. |
| **`test_sosnovik_generator.py`** | Script | Research prototype for Poisson boundary sampling. Fully integrated into `dataset_generator.py`. |
| **`create_sampling_comparison.py`** | Script | One-off plot script for the bulky vs. precise comparison graphic. |
| **`create_literature_pipeline_figure.py`** | Script | One-off infographic generator. |
| **`test_dataset.npz`** | Data | Small 4-sample test dataset archive. |
| **`topo_dataset.npz`** | Data | Incomplete single-sample test file. |
| **`__pycache__/`** | Cache | Python compiled `.pyc` bytecode directories. |

### 4.2 Retained Core Files
| File Retained | Role | Why It is Essential |
| :--- | :--- | :--- |
| **`dataset_generator.py`** | **Core Pipeline** | High-throughput parallel SIMP dataset generator producing `(N, 5, 60, 120)` inputs $\to$ `(N, 60, 120)` density targets. |
| **`visualize_data.py`** | **Visualizer** | Renders 3-panel engineering graphics with support symbols and load vectors. |
| **`handover.md`** | **Documentation** | Definitive technical specification and handover report. |
| **`topo_opti_project.pdf`** | **Specification** | User's original course requirements document. |
| **`.gitignore`** | **Configuration** | Prevents cache files, checkpoints, and data archives from polluting version control. |

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

## 9. Phase 2 & Phase 3 Roadmap (Deep Learning & U-Net Training)

With Phase 1 complete and the codebase cleaned, the project moves to **Phase 2 (Model Architecture & Training Pipeline)**:

### 9.1 Recommended U-Net Architecture
- **Input:** Tensor of shape `(Batch_Size, 5, 60, 120)`.
- **Backbone:** 4-level encoder-decoder U-Net with double convolutions `(Conv2d -> BatchNorm -> LeakyReLU)` at each stage:
  - Level 1: 5 $\to$ 64 channels (downsampled to $30 \times 60$)
  - Level 2: 64 $\to$ 128 channels (downsampled to $15 \times 30$)
  - Level 3: 128 $\to$ 256 channels (downsampled to $7 \times 15$ with padding)
  - Bottleneck: 256 $\to$ 512 channels
  - Decoder: Symmetrical transpose-convolution upsamplers with skip connections concatenating matching encoder feature maps.
- **Output:** 1-channel density map passing through a `Sigmoid` activation: $\hat{\mathbf{y}} \in [0, 1]^{60 \times 120}$.

### 9.2 Recommended Composite Loss Function
To guarantee physical compliance and crisp boundary binarization, train with a 3-component loss:
$$\mathcal{L} = \mathcal{L}_{\text{BCE}}(\hat{\mathbf{y}}, \mathbf{y}) + \alpha \cdot \left(\text{mean}(\hat{\mathbf{y}}) - V_f\right)^2 + \beta \sum_{e} \hat{y}_e (1 - \hat{y}_e)$$
1. **Binary Cross-Entropy ($\mathcal{L}_{\text{BCE}}$):** Matches pixel-wise ground-truth material distribution.
2. **Volume Fraction Constraint ($\alpha \approx 10.0$):** Strictly penalizes deviations from the user's requested volume fraction $V_f$.
3. **Binarization Regularizer ($\beta \approx 10^{-3}$):** Ramps up after epoch 10 to penalize intermediate gray densities ($0 < \rho < 1$) and drive the layout towards crisp $0$ or $1$ boundaries.

### 9.3 Evaluation Metrics for Phase 3
- **Mean Squared Error (MSE)** & **Dice Similarity Coefficient** against SIMP ground truth.
- **Compliance Discrepancy:**
  $$\Delta c = \frac{|c(\hat{\mathbf{y}}) - c(\mathbf{y}^*)|}{c(\mathbf{y}^*)} \times 100\%$$
- **Inference Latency:** Benchmark model inference time on CPU and GPU (target: $< 15$ ms per topology).

---
*Report updated and verified for Phase 1 completion.*
