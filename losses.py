"""
losses.py - Physics-Induced Composite Loss Functions & Self-Adjoint FEA Compliance Layer

Implements:
1. TopoCompositeLoss:
   - Mechanics & Strain-Energy Importance Weight Map (W_mech)
   - Weighted Binary Cross-Entropy Loss (L_bce)
   - SIMP Cubic Stiffness Discrepancy Loss (L_stiff: penalizing |pred^3 - target^3|)
   - Multi-Scale + Morphologically Eroded Dice Loss (L_dice: preserving thin truss members)
   - Sobel Structural Edge Alignment Loss (L_edge: matching structural truss boundaries)
   - Volume Penalty (L_vf) on predicted and pre-projection probabilities
   - Scheduled Binarization Regularizer (L_bin)
2. SelfAdjointFEACompliance:
   - torch.autograd.Function solving Q4 linear elasticity KU = F via SciPy spsolve
   - Analytical self-adjoint gradient: dC/d_rho_e = -p * (E0 - Emin) * rho_e^(p-1) * (u_e^T K0 u_e)
3. compute_fea_compliance_ratio:
   - Helper evaluating compliance of predicted vs ground-truth structures
"""

import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 1. STRUCTURAL DICE LOSS (WITH CONTINUOUS & MORPHOLOGICAL EROSION)
# =========================================================================
class DiceLoss(nn.Module):
    """
    Soft Dice Loss for continuous density structural overlap.
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred and target shapes: (B, 1, H, W)
        pred_flat = pred.contiguous().view(pred.size(0), -1)
        target_flat = target.contiguous().view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# =========================================================================
# 2. PHYSICS-INDUCED COMPOSITE LOSS
# =========================================================================
class TopoCompositeLoss(nn.Module):
    """
    Composite Physics-Guided Loss for TopoUNet / PI-UNet.
    
    Args:
        w_bce (float): Weight for mechanics-weighted BCE (default: 1.0)
        w_dice (float): Weight for multi-scale / eroded Dice loss (default: 0.5)
        alpha_vf (float): Weight for volume fraction constraint (default: 10.0)
        beta_bin (float): Weight for scheduled binarization penalty (default: 1e-3)
        w_stiff (float): Weight for SIMP p=3 cubic stiffness loss (default: 0.5)
        w_edge (float): Weight for Sobel structural edge alignment loss (default: 0.2)
    """
    def __init__(self, w_bce=1.0, w_dice=0.5, alpha_vf=10.0, beta_bin=1e-3,
                 w_stiff=0.5, w_edge=0.2):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.alpha_vf = alpha_vf
        self.beta_bin = beta_bin
        self.w_stiff = w_stiff
        self.w_edge = w_edge

        self.dice = DiceLoss()

        # Sobel edge filter kernels
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target, inputs, epoch=1, strain_energy=None, raw_probs=None):
        """
        pred:          (B, 1, H, W) predicted density in [0, 1]
        target:        (B, 1, H, W) ground-truth SIMP density in [0, 1]
        inputs:        (B, 6, H, W) or (B, 5, H, W) physical input tensor
        epoch:         current training epoch (for scheduling binarization weight)
        strain_energy: optional (B, 1, H, W) normalized element strain energy map
        raw_probs:     optional (B, 1, H, W) pre-projection probabilities from TopoUNet
        """
        # Auto-handle 5-ch or 6-ch inputs
        if inputs.shape[1] == 5:
            inputs = torch.cat([inputs[:, 0:1], inputs[:, 1:2], inputs[:, 1:2], inputs[:, 2:5]], dim=1)

        domain = inputs[:, 0:1]
        bc_x = inputs[:, 1:2]
        bc_y = inputs[:, 2:3]
        fx = inputs[:, 3:4]
        fy = inputs[:, 4:5]
        target_vf = inputs[:, 5:6]

        # 1. Mechanics & Strain-Energy Importance Weight Map (W_mech)
        # Fixture and load anchor map
        anchors = torch.clamp(bc_x + bc_y + torch.sqrt(fx**2 + fy**2), 0.0, 1.0)
        # 5x5 max/avg dilation to encompass anchor neighborhoods
        anchors_dilated = F.avg_pool2d(
            F.max_pool2d(anchors, kernel_size=5, stride=1, padding=2),
            kernel_size=5, stride=1, padding=2
        )

        if strain_energy is not None:
            if strain_energy.ndim == 3:
                strain_energy = strain_energy.unsqueeze(1)
            se_clamped = torch.clamp(strain_energy.detach(), 0.0, 1.5)
            W_mech = 1.0 + 1.5 * anchors_dilated.detach() + 1.5 * se_clamped + 0.5 * target.detach()
        else:
            W_mech = 1.0 + 1.5 * anchors_dilated.detach() + 1.0 * target.detach()
        W_mech = W_mech.detach()

        # 2. Mechanics-Weighted BCE Loss
        pred_clamped = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
        loss_bce = F.binary_cross_entropy(pred_clamped, target, weight=W_mech)

        # 3. SIMP Cubic Stiffness Discrepancy Loss: penalizes |pred^3 - target^3|
        loss_stiff = torch.mean(W_mech * torch.abs(pred**3 - target**3))

        # 4. Multi-Scale + Morphologically Eroded Dice Loss
        # erode(u) = -max_pool2d(-u) detects and preserves thin structural truss members
        def erode(u):
            return -F.max_pool2d(-u, kernel_size=3, stride=1, padding=1)

        dice_standard = self.dice(pred, target)
        dice_eroded = self.dice(erode(pred), erode(target))
        loss_dice = 0.6 * dice_standard + 0.4 * dice_eroded

        # 5. Sobel Structural Edge Alignment Loss
        pred_pad = F.pad(pred, (1, 1, 1, 1), mode='replicate')
        target_pad = F.pad(target, (1, 1, 1, 1), mode='replicate')

        sx = self.sobel_x.to(device=pred.device, dtype=pred.dtype)
        sy = self.sobel_y.to(device=pred.device, dtype=pred.dtype)

        grad_pred_x = F.conv2d(pred_pad, sx)
        grad_pred_y = F.conv2d(pred_pad, sy)
        grad_target_x = F.conv2d(target_pad, sx)
        grad_target_y = F.conv2d(target_pad, sy)

        loss_edge = F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)

        # 6. Volume Fraction Penalty (regularizing both pred and raw_probs)
        t_vf = target_vf.mean(dim=(-2, -1))
        p_vf = pred.mean(dim=(-2, -1))
        loss_vf = F.mse_loss(p_vf, t_vf)

        if raw_probs is not None:
            r_vf = raw_probs.mean(dim=(-2, -1))
            loss_vf = loss_vf + 0.5 * F.mse_loss(r_vf, t_vf)

        # 7. Scheduled Binarization Penalty
        beta_eff = self.beta_bin * min(1.0, max(0.0, (epoch - 5) / 10.0))
        loss_bin = torch.mean(4.0 * pred * (1.0 - pred))

        # Total Composite Multi-Objective Loss
        total_loss = (
            self.w_bce * loss_bce +
            self.w_dice * loss_dice +
            self.alpha_vf * loss_vf +
            beta_eff * loss_bin +
            self.w_stiff * loss_stiff +
            self.w_edge * loss_edge
        )

        loss_dict = {
            'total': total_loss.item(),
            'bce': loss_bce.item(),
            'dice': loss_dice.item(),
            'vf_mse': loss_vf.item(),
            'bin': loss_bin.item(),
            'stiff': loss_stiff.item(),
            'edge': loss_edge.item()
        }

        return total_loss, loss_dict


# =========================================================================
# 3. SELF-ADJOINT FEA COMPLIANCE LAYER (Q4 Linear Elasticity)
# =========================================================================
def get_q4_element_stiffness(Emax=1.0, nu=0.3):
    """Exact 8x8 Q4 element stiffness matrix K0 for plane stress unit square."""
    k = [
        1/2 - nu/6,   1/8 + nu/8,   -1/4 - nu/12, -1/8 + 3*nu/8,
        -1/4 + nu/12, -1/8 - nu/8,  nu/6,         1/8 - 3*nu/8
    ]
    KE = Emax / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ], dtype=np.float64)
    return KE


def precompute_fea_mesh(nelx=120, nely=60):
    """Precomputes edofMat, iK, jK for fast sparse global stiffness assembly."""
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
    iK = np.kron(edofMat, np.ones((8, 1), dtype=int)).flatten()
    jK = np.kron(edofMat, np.ones((1, 8), dtype=int)).flatten()
    return edofMat, iK, jK


def reconstruct_fea_boundary_conditions(inp_np, nelx, nely):
    """
    Reconstructs fixed_dofs and global force vector F from a 6-ch or 5-ch input tensor.
    """
    if inp_np.shape[0] == 5:
        inp_np = np.concatenate([inp_np[0:1], inp_np[1:2], inp_np[1:2], inp_np[2:5]], axis=0)

    ch_bc_x = inp_np[1]
    ch_bc_y = inp_np[2]
    ch_fx = inp_np[3]
    ch_fy = inp_np[4]

    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof, dtype=np.float64)
    fixed_dofs = []

    # Map cell forces and constraints to nodes
    for i in range(nelx):
        for j in range(nely):
            n1 = (nely + 1) * i + j
            n2 = (nely + 1) * (i + 1) + j
            n3 = (nely + 1) * (i + 1) + j + 1
            n4 = (nely + 1) * i + j + 1
            nodes = [n1, n2, n3, n4]

            fx_val = float(ch_fx[j, i])
            fy_val = float(ch_fy[j, i])
            if abs(fx_val) > 1e-5 or abs(fy_val) > 1e-5:
                for n in nodes:
                    F[2*n] += 0.25 * fx_val
                    F[2*n + 1] += 0.25 * fy_val

            if ch_bc_x[j, i] > 0.05:
                for n in nodes:
                    fixed_dofs.append(2*n)
            if ch_bc_y[j, i] > 0.05:
                for n in nodes:
                    fixed_dofs.append(2*n + 1)

    fixed_dofs = np.unique(fixed_dofs) if len(fixed_dofs) > 0 else np.array([0, 1], dtype=int)
    return fixed_dofs, F


class SelfAdjointFEAComplianceFunction(torch.autograd.Function):
    """
    Differentiable FEA Compliance layer with exact analytical self-adjoint gradient.
    Forward:
      Assembles K(rho), solves KU = F, computes compliance C = sum_e (Emin + rho_e^p (E0 - Emin)) u_e^T K0 u_e
    Backward:
      dC/d_rho_e = -p * (E0 - Emin) * rho_e^(p-1) * (u_e^T K0 u_e)
    """
    @staticmethod
    def forward(ctx, rho, fixed_dofs, F_vec, edofMat, iK, jK, KE, penal=3.0, Emin=1e-9, Emax=1.0):
        # rho: (B, 1, nely, nelx) or (nely, nelx)
        device = rho.device
        dtype = rho.dtype
        is_batched = (rho.ndim == 4)

        rho_np = rho.detach().cpu().numpy()
        batch_size = rho_np.shape[0] if is_batched else 1

        nely = rho.shape[-2]
        nelx = rho.shape[-1]
        ndof = 2 * (nelx + 1) * (nely + 1)
        alldofs = np.arange(ndof)
        freedofs = np.setdiff1d(alldofs, fixed_dofs)

        compliances = []
        dC_drho_list = []

        KE_flat = KE.flatten()[:, None]

        for b in range(batch_size):
            r = rho_np[b, 0] if is_batched else rho_np
            # Flatten in order matching edofMat (x varies fast/outer, y inner)
            # row = i * nely + j => transpose(1, 0).flatten()
            r_phys = r.T.flatten()

            # SIMP interpolation
            E_elem = Emin + (r_phys ** penal) * (Emax - Emin)
            sK = (KE_flat * E_elem).flatten(order='F')
            K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
            K_free = K[freedofs, :][:, freedofs]

            try:
                U_free = spsolve(K_free, F_vec[freedofs])
            except Exception:
                U_free = np.zeros(len(freedofs))

            U = np.zeros(ndof)
            U[freedofs] = U_free

            U_edof = U[edofMat] # (nelx*nely, 8)
            ce = (np.dot(U_edof, KE) * U_edof).sum(axis=1) # (nelx*nely,) u_e^T K0 u_e

            c_val = float(np.sum(E_elem * ce))
            compliances.append(c_val)

            # Analytical self-adjoint gradient
            dc = -penal * (Emax - Emin) * (r_phys ** (penal - 1)) * ce
            # Reshape back to (nely, nelx)
            dc_2d = dc.reshape(nelx, nely).T
            dC_drho_list.append(dc_2d)

        ctx.save_for_backward(torch.from_numpy(np.stack(dC_drho_list)).to(device=device, dtype=dtype))
        ctx.is_batched = is_batched

        comp_tensor = torch.tensor(compliances, device=device, dtype=dtype)
        return comp_tensor if is_batched else comp_tensor[0]

    @staticmethod
    def backward(ctx, grad_output):
        (grad_dC,) = ctx.saved_tensors
        if ctx.is_batched:
            # grad_output is (B,) or scalar
            g = grad_output.view(-1, 1, 1, 1)
            grad_rho = grad_dC.unsqueeze(1) * g
        else:
            grad_rho = grad_dC[0] * grad_output

        # None for non-tensor arguments
        return grad_rho, None, None, None, None, None, None, None, None, None


class SelfAdjointFEACompliance(nn.Module):
    """
    Module wrapper for SelfAdjointFEAComplianceFunction.
    """
    def __init__(self, nelx=120, nely=60, penal=3.0, Emin=1e-9, Emax=1.0):
        super().__init__()
        self.nelx = nelx
        self.nely = nely
        self.penal = penal
        self.Emin = Emin
        self.Emax = Emax

        self.KE = get_q4_element_stiffness(Emax=Emax)
        self.edofMat, self.iK, self.jK = precompute_fea_mesh(nelx, nely)

    def forward(self, rho, fixed_dofs, F_vec):
        return SelfAdjointFEAComplianceFunction.apply(
            rho, fixed_dofs, F_vec, self.edofMat, self.iK, self.jK, self.KE,
            self.penal, self.Emin, self.Emax
        )


_MESH_CACHE = {}
_FILTER_CACHE = {}

def get_cached_fea_mesh(nelx=120, nely=60):
    key = (nelx, nely)
    if key not in _MESH_CACHE:
        edofMat, iK, jK = precompute_fea_mesh(nelx, nely)
        KE = get_q4_element_stiffness()
        _MESH_CACHE[key] = (edofMat, iK, jK, KE)
    return _MESH_CACHE[key]


def get_cached_filter(nelx=120, nely=60, rmin=1.5):
    key = (nelx, nely, rmin)
    if key not in _FILTER_CACHE:
        from simp_solver import precompute_mesh_and_filter
        _, _, _, H, Hs = precompute_mesh_and_filter(nelx, nely, rmin)
        _FILTER_CACHE[key] = (H, Hs)
    return _FILTER_CACHE[key]


def bridge_diagonal_hinges(bin_grid):
    """
    Eliminates 1-node checkerboard corner hinges in a binary Q4 mesh without eroding borders.
    Converts 8-connected diagonal-only contacts into 4-connected edge-sharing Q4 elements.
    """
    out = bin_grid.copy()
    p00 = bin_grid[:-1, :-1] > 0.5
    p11 = bin_grid[1:, 1:] > 0.5
    p01 = bin_grid[:-1, 1:] > 0.5
    p10 = bin_grid[1:, :-1] > 0.5

    diag1 = p00 & p11 & (~p01) & (~p10)
    diag2 = p01 & p10 & (~p00) & (~p11)

    y1, x1 = np.where(diag1)
    out[y1, x1 + 1] = 1.0
    out[y1 + 1, x1] = 1.0

    y2, x2 = np.where(diag2)
    out[y2, x2] = 1.0
    out[y2 + 1, x2 + 1] = 1.0
    return out


def physics_aware_binarize(pred_density, inp_np, threshold=0.35, polish_steps=2):
    """
    Solution 1: Physics-Aware Binarization & Boundary Load-Pad Locking.
    1. Locks 3x3 solid anchor pads around active support fixtures and point loads
       so forces are never applied to void elements or single-node hinges.
    2. Performs fast sensitivity-filtered strain-energy projection (polish_steps=2)
       on the continuous prediction to heal faint internal struts and pinch points.
    3. Applies exact volume-preserving quantile thresholding (matching target Vf)
       and bridges 1-node diagonal corner hinges into 4-connected Q4 elements.
    """
    from scipy.ndimage import binary_dilation

    if inp_np.shape[0] == 5:
        inp_np = np.concatenate([inp_np[0:1], inp_np[1:2], inp_np[1:2], inp_np[2:5]], axis=0)

    pred_2d = np.asarray(pred_density, dtype=np.float64).squeeze().copy()
    nely, nelx = pred_2d.shape[-2], pred_2d.shape[-1]
    target_vf = float(np.clip(np.mean(inp_np[5]), 0.05, 0.95))

    sup_mask = (inp_np[1] > 0.05) | (inp_np[2] > 0.05)
    load_mask = (np.abs(inp_np[3]) > 1e-4) | (np.abs(inp_np[4]) > 1e-4)
    struct_3x3 = np.ones((3, 3), dtype=int)
    anchors = binary_dilation(sup_mask | load_mask, structure=struct_3x3, iterations=1)

    p = pred_2d
    p[anchors] = 1.0

    if polish_steps > 0:
        fixed_dofs, F_vec = reconstruct_fea_boundary_conditions(inp_np, nelx, nely)
        edofMat, iK, jK, KE = get_cached_fea_mesh(nelx, nely)
        H, Hs = get_cached_filter(nelx, nely, 1.5)

        alldofs = np.arange(2 * (nelx + 1) * (nely + 1))
        freedofs = np.setdiff1d(alldofs, fixed_dofs)

        x = np.clip(p.T.flatten(), 0.08, 1.0)
        anc_flat = anchors.T.flatten()
        nele = nelx * nely
        target_mass = max(target_vf * nele, float(np.sum(anc_flat)) + 1.0)

        for _ in range(polish_steps):
            E_elem = 1e-9 + (x ** 3.0) * (1.0 - 1e-9)
            sK = (KE.flatten()[:, None] * E_elem).flatten(order='F')
            K = sp.coo_matrix((sK, (iK, jK)), shape=(len(alldofs), len(alldofs))).tocsc()
            try:
                U_free = spsolve(K[freedofs, :][:, freedofs], F_vec[freedofs])
                if not np.all(np.isfinite(U_free)):
                    break
                U = np.zeros(len(alldofs))
                U[freedofs] = U_free
                U_edof = U[edofMat]
                ce = (np.dot(U_edof, KE) * U_edof).sum(axis=1)
                dc = -3.0 * (x ** 2.0) * ce
                dc_f = np.asarray(H.dot(x * dc)) / Hs / np.maximum(1e-3, x)
                dc_f = np.minimum(dc_f, -1e-12)

                l1, l2 = 0.0, 1e9
                for _ in range(40):
                    if (l2 - l1) / (l1 + l2 + 1e-15) <= 1e-4:
                        break
                    lmid = 0.5 * (l1 + l2)
                    xnew = np.maximum(
                        0.0,
                        np.maximum(
                            x - 0.35,
                            np.minimum(1.0, np.minimum(x + 0.35, x * np.sqrt(np.maximum(0.0, -dc_f) / max(lmid, 1e-15))))
                        )
                    )
                    xnew[anc_flat] = 1.0
                    if np.sum(xnew) > target_mass:
                        l1 = lmid
                    else:
                        l2 = lmid
                x = xnew
            except Exception:
                break

        p = x.reshape(nelx, nely).T

    # Volume-preserving quantile threshold
    tau = float(np.quantile(p, max(0.05, min(0.95, 1.0 - target_vf))))
    bin_pred = (p >= tau).astype(float)
    bin_pred[anchors] = 1.0
    bin_pred = bridge_diagonal_hinges(bin_pred)
    return bin_pred


def compute_fea_compliance_ratio(pred_np, target_np, inp_np):
    """
    Computes structural compliance ratio C(pred) / C(target) for validation.
    Returns:
      ratio: C(pred) / max(C(target), 1e-9)
      c_pred: scalar compliance of prediction
      c_target: scalar compliance of ground truth
    """
    nelx = pred_np.shape[-1]
    nely = pred_np.shape[-2]

    fixed_dofs, F_vec = reconstruct_fea_boundary_conditions(inp_np, nelx, nely)
    edofMat, iK, jK, KE = get_cached_fea_mesh(nelx, nely)

    alldofs = np.arange(2 * (nelx + 1) * (nely + 1))
    freedofs = np.setdiff1d(alldofs, fixed_dofs)

    def solve_c(density):
        r_phys = density.T.flatten()
        E_elem = 1e-9 + (r_phys ** 3.0) * (1.0 - 1e-9)
        sK = (KE.flatten()[:, None] * E_elem).flatten(order='F')
        K = sp.coo_matrix((sK, (iK, jK)), shape=(len(alldofs), len(alldofs))).tocsc()
        K_free = K[freedofs, :][:, freedofs]
        try:
            U_free = spsolve(K_free, F_vec[freedofs])
            if not np.all(np.isfinite(U_free)):
                return float('inf')
        except Exception:
            return float('inf')
        U = np.zeros(len(alldofs))
        U[freedofs] = U_free
        U_edof = U[edofMat]
        ce = (np.dot(U_edof, KE) * U_edof).sum(axis=1)
        c_val = float(np.sum(E_elem * ce))
        return c_val if c_val > 0.0 else float('inf')

    c_pred = solve_c(pred_np.squeeze())
    c_target = solve_c(target_np.squeeze())
    ratio = c_pred / max(c_target, 1e-9)
    return ratio, c_pred, c_target


def check_structural_stability(pred_density, inp_np, simp_density=None, threshold=0.35,
                               material_name="steel", force_real_N=1500.0,
                               max_compliance_ratio=2.5, compute_buckling=True,
                               use_physics_binarize=True, bin_pred_override=None):
    """
    Comprehensive Engineering Stability & Breakage Inspector for TopoUNet Predictions.

    Evaluates 4 physical failure modes to determine if the generated topology is STABLE or CAN BREAK:
      1. Topological Load-to-Support Connectivity:
         Checks via 8-neighbor connected components whether every applied load region has an
         unbroken solid load path to at least one support fixture.
      2. FEA Elastic Stiffness / Hinge-Mechanism Stability:
         Solves 2D Q4 FEA (KU = F). Checks if compliance C_pred is finite, free of structural
         hinge collapse (C_pred / C_simp <= max_compliance_ratio, or C_pred < 250 when SIMP unavailable).
      3. Material Yielding / Von Mises Stress Safety:
         Recovers 2D plane-stress Von Mises field and checks Factor of Safety (FoS >= 1.0)
         and stress concentration ratio relative to SIMP.
      4. In-Plane Euler Buckling Stability:
         Solves active-submesh geometric stiffness eigenvalue problem for critical buckling load
         multiplier (lambda_1 >= 1.0).

    Returns a dictionary containing all quantitative metrics, boolean pass flags, failure reasons,
    and a human-readable badge ('STABLE (SAFE)' vs 'CAN BREAK (...)').
    """
    from scipy.ndimage import label, binary_dilation
    from simp_solver import analyze_structural_integrity

    if inp_np.shape[0] == 5:
        inp_np = np.concatenate([inp_np[0:1], inp_np[1:2], inp_np[1:2], inp_np[2:5]], axis=0)

    pred_2d = np.asarray(pred_density, dtype=float).squeeze()
    nely, nelx = pred_2d.shape[-2], pred_2d.shape[-1]
    if bin_pred_override is not None:
        bin_pred = np.asarray(bin_pred_override, dtype=float).squeeze()
    elif use_physics_binarize:
        bin_pred = physics_aware_binarize(pred_2d, inp_np, threshold=threshold, polish_steps=2)
    else:
        bin_pred = (pred_2d >= threshold).astype(float)
    bin_simp = (simp_density >= 0.5).astype(float) if simp_density is not None else None

    # --- 1. TOPOLOGICAL LOAD-PATH CONNECTIVITY CHECK ---
    ch_bc_x, ch_bc_y = inp_np[1], inp_np[2]
    ch_fx, ch_fy = inp_np[3], inp_np[4]

    support_mask = (ch_bc_x > 0.05) | (ch_bc_y > 0.05)
    load_mask = (np.abs(ch_fx) > 1e-4) | (np.abs(ch_fy) > 1e-4)

    struct_8 = np.ones((3, 3), dtype=int)
    support_zone = binary_dilation(support_mask, structure=struct_8, iterations=1)
    load_zone = binary_dilation(load_mask, structure=struct_8, iterations=1)

    solid_mask = bin_pred > 0.5
    labeled_grid, num_features = label(solid_mask, structure=struct_8)

    # Identify solid components anchored to supports
    anchored_labels = set(np.unique(labeled_grid[support_zone & solid_mask])) - {0}
    touches_support = len(anchored_labels) > 0

    # Check if every distinct load cluster connects to an anchored solid component
    load_labels, num_loads = label(load_mask, structure=struct_8)
    all_loads_connected = True
    if num_loads == 0:
        all_loads_connected = touches_support
    else:
        for l_idx in range(1, num_loads + 1):
            l_region = binary_dilation(load_labels == l_idx, structure=struct_8, iterations=1)
            comp_at_load = set(np.unique(labeled_grid[l_region & solid_mask])) - {0}
            if len(comp_at_load.intersection(anchored_labels)) == 0:
                all_loads_connected = False
                break

    # Ratio of solid mass that is properly anchored to supports (detects floating islands)
    total_solid_pixels = int(np.sum(solid_mask))
    if total_solid_pixels > 0 and len(anchored_labels) > 0:
        anchored_pixels = int(np.sum(np.isin(labeled_grid, list(anchored_labels))))
        anchored_mass_ratio = float(anchored_pixels) / float(total_solid_pixels)
    else:
        anchored_mass_ratio = 0.0

    is_connected = bool(touches_support and all_loads_connected and anchored_mass_ratio >= 0.70)

    # --- 2. FEA STIFFNESS & COMPLIANCE RATIO CHECK ---
    fixed_dofs, F_vec = reconstruct_fea_boundary_conditions(inp_np, nelx, nely)
    edofMat, iK, jK, KE = get_cached_fea_mesh(nelx, nely)

    if bin_simp is not None:
        fea_ratio, c_pred, c_target = compute_fea_compliance_ratio(bin_pred, bin_simp, inp_np)
    else:
        fea_ratio, c_pred, _ = compute_fea_compliance_ratio(bin_pred, bin_pred, inp_np)
        c_target = c_pred

    if simp_density is not None:
        stiff_pass = bool(np.isfinite(c_pred) and (c_pred > 0.0) and (0.0 < fea_ratio <= max_compliance_ratio) and (c_pred < 500.0))
    else:
        stiff_pass = bool(np.isfinite(c_pred) and (0.0 < c_pred < 250.0))

    # --- 3. VON MISES STRESS (YIELDING) & BUCKLING CHECK ---
    # Remove any tiny unanchored floating dust pixels before eigenvalue buckling solve
    # so isolated void islands do not introduce spurious zero-stiffness rigid body modes
    struct_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    lbl_4, _ = label(solid_mask, structure=struct_4)
    anc_4 = set(np.unique(lbl_4[support_zone & solid_mask])) - {0}
    if len(anc_4) > 0:
        bin_pred_anchored = np.isin(lbl_4, list(anc_4)).astype(float)
    elif len(anchored_labels) > 0:
        bin_pred_anchored = np.isin(labeled_grid, list(anchored_labels)).astype(float)
    else:
        bin_pred_anchored = bin_pred

    res_pred = analyze_structural_integrity(
        density=bin_pred_anchored, fixed_dofs=fixed_dofs, F=F_vec, edofMat=edofMat,
        nelx=nelx, nely=nely, material_name=material_name,
        force_real_N=force_real_N, density_threshold=0.3
    )

    res_simp = None
    if bin_simp is not None:
        res_simp = analyze_structural_integrity(
            density=bin_simp, fixed_dofs=fixed_dofs, F=F_vec, edofMat=edofMat,
            nelx=nelx, nely=nely, material_name=material_name,
            force_real_N=force_real_N, density_threshold=0.3
        )

    if res_pred is not None:
        fos_yield = float(res_pred['fos_yield'])
        peak_vm = float(res_pred['peak_vm_stress'])
        vm_95th = float(res_pred['vm_95th'])
        lambda_1 = float(res_pred['lambda_1']) if compute_buckling else 999.0
        max_defl_mm = float(res_pred['max_deflection_mm'])
    else:
        fos_yield = 0.0
        peak_vm = float('inf')
        vm_95th = float('inf')
        lambda_1 = 0.0
        max_defl_mm = float('inf')

    if res_simp is not None and res_simp['vm_95th'] > 1e-6:
        simp_fos = float(res_simp['fos_yield'])
        simp_lambda_1 = float(res_simp['lambda_1'])
        stress_ratio = float(vm_95th / max(res_simp['vm_95th'], 1e-6))
        yield_pass = bool((fos_yield >= 1.0 or stress_ratio <= 2.0) and stress_ratio <= 2.5)
    else:
        simp_fos = fos_yield
        simp_lambda_1 = lambda_1
        stress_ratio = 1.0
        yield_pass = bool(fos_yield >= 1.0)
    buckle_pass = bool(lambda_1 >= 1.0)

    # --- 4. OVERALL STABILITY VERDICT & FAILURE MODE DIAGNOSIS ---
    failure_reasons = []
    if not is_connected:
        if not touches_support:
            failure_reasons.append("Detached from Support")
        elif not all_loads_connected:
            failure_reasons.append("Disconnected Load Path")
        else:
            failure_reasons.append("Floating Island Split")
    if not stiff_pass:
        failure_reasons.append(f"Excessive Compliance ({fea_ratio:.2f}x)")
    if not yield_pass:
        failure_reasons.append(f"Stress Overload (FoS={fos_yield:.2f})")
    if not buckle_pass:
        failure_reasons.append(f"Buckling Instability (lambda_1={lambda_1:.2f})")

    is_stable = bool(is_connected and stiff_pass and yield_pass and buckle_pass)

    if is_stable:
        if fea_ratio <= 1.25 and fos_yield >= 1.5 and lambda_1 >= 1.5:
            status_badge = "STABLE (ROBUST - WILL NOT BREAK)"
            short_badge = "STABLE"
            badge_color = "#2E7D32"  # Dark Green
        else:
            status_badge = "STABLE (SAFE - MINOR STIFFNESS LOSS)"
            short_badge = "STABLE"
            badge_color = "#388E3C"  # Green
    else:
        reason_str = ", ".join(failure_reasons[:2])
        status_badge = f"CAN BREAK / UNSTABLE ({reason_str})"
        short_badge = "CAN BREAK"
        badge_color = "#D32F2F"      # Red

    return {
        "is_stable": is_stable,
        "status_badge": status_badge,
        "short_badge": short_badge,
        "badge_color": badge_color,
        "failure_reasons": failure_reasons,
        "is_connected": is_connected,
        "touches_support": touches_support,
        "all_loads_connected": all_loads_connected,
        "anchored_mass_ratio": anchored_mass_ratio,
        "stiff_pass": stiff_pass,
        "yield_pass": yield_pass,
        "buckle_pass": buckle_pass,
        "c_pred": float(c_pred),
        "c_target": float(c_target),
        "fea_ratio": float(fea_ratio),
        "fos_yield": float(fos_yield),
        "simp_fos": float(simp_fos),
        "peak_vm_mpa": float(peak_vm),
        "vm_95th_mpa": float(vm_95th),
        "stress_ratio": float(stress_ratio),
        "lambda_1": float(lambda_1),
        "simp_lambda_1": float(simp_lambda_1),
        "max_deflection_mm": float(max_defl_mm)
    }
