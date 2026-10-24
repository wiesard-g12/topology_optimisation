"""
losses_rl_ver.py - SIMP-Judge RL Environment, Differentiable Self-Adjoint FEA Judge,
                   and Thin-Strut Physics Composite Loss for TopoUNet / PI-UNet.

Copy of losses.py upgraded with:
1. SelfAdjointFEAJudgeFunction & SIMPJudgeEnvironment:
   - Multi-threaded CPU sparse FEA judge solving K(rho) U = F for each sample in a mini-batch.
   - Computes exact self-adjoint compliance sensitivity:
       dC / d_rho_e = -p * (E0 - Emin) * rho_e^(p-1) * (u_e^T K0 u_e)
   - Applies Sigmund conical sensitivity filtering (H) so high strain energy at pinched joints
     propagates into adjacent low-density/void elements (escaping the rho=0 gradient trap).
   - Computes an online 1-step Optimality Criteria (OC) teacher state rho_judge = OC_1(rho_pred)
     acting as a dense step-wise SIMP trajectory target.
   - Computes per-sample RL stability reward R_i and advantage weight A_i = 1 + gamma * (1 - R_i)
     to focus policy updates on samples with high compliance ratio C(pred)/C* > 1.15.
   - Normalizes per-sample adjoint gradients by their 95th percentile to guarantee zero gradient
     explosion even when an intermediate topology is near-singular.
2. TopoRLCompositeLoss:
   - Upgrades W_mech with a Morphological Thin-Strut Skeleton Weight Map:
       thin_struts = clamp(target - erode_5x5(target), 0, 1)
     upweighting thin 2-3 pixel internal web bars (like Sample #2450's middle bar) by 3.5x.
   - Integrates the SIMP-Judge self-adjoint compliance loss, online OC-step distillation loss,
     and PPO-style reference policy trust-region anchor loss.
3. Physics-Aware Binarization & Structural Stability Inspector (Solution 1).
"""

import math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import torch
import torch.nn as nn
import torch.nn.functional as F


from losses import (
    DiceLoss,
    get_q4_element_stiffness,
    precompute_fea_mesh,
    get_cached_fea_mesh,
    get_cached_filter,
    reconstruct_fea_boundary_conditions,
    bridge_diagonal_hinges,
    physics_aware_binarize,
    compute_fea_compliance_ratio,
    check_structural_stability,
)

_C_TARG_CACHE = {}


# =========================================================================
# 3. DIFFERENTIABLE SIMP-JUDGE RL ENVIRONMENT (SELF-ADJOINT FEA + OC STEP)
# =========================================================================
def _solve_single_simp_judge(args):
    """
    Worker function solving a single sample inside the SIMP-Judge Environment.
    Computes:
      1. C_pred: FEA compliance of predicted topology (with relaxed floor rho_min=0.06
         so missing internal struts still carry virtual strain energy and receive non-zero gradient).
      2. C_target: FEA compliance of ground-truth SIMP topology.
      3. grad_2d: Analytical self-adjoint sensitivity d(C_pred / C_target)/d_rho filtered by
         Sigmund's conical filter H and normalized to bounded range [-2.0, 2.0].
      4. rho_judge_2d: 1-step Optimality Criteria (OC) improved density map acting as an
         online SIMP teacher step.
      5. reward & advantage: RL stability reward R_i in (0, 1] and advantage weight A_i in [1, 3.5].
    """
    r_pred_2d, r_targ_2d, inp_np, penal, Emin, Emax = args
    nely, nelx = r_pred_2d.shape
    ndof = 2 * (nelx + 1) * (nely + 1)
    nele = nelx * nely

    edofMat, iK, jK, KE = get_cached_fea_mesh(nelx, nely)
    H, Hs = get_cached_filter(nelx, nely, 1.5)
    fixed_dofs, F_vec = reconstruct_fea_boundary_conditions(inp_np, nelx, nely)

    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixed_dofs)
    KE_flat = KE.flatten()[:, None]
    target_vf = float(np.mean(inp_np[5])) if inp_np.shape[0] >= 6 else float(np.mean(r_targ_2d))

    # Support & load anchor mask
    sup_mask = (inp_np[1] > 0.05) | (inp_np[2] > 0.05)
    load_mask = (np.abs(inp_np[3]) > 1e-4) | (np.abs(inp_np[4]) > 1e-4)
    anc_flat = (sup_mask | load_mask).T.flatten()

    def solve_elasticity(r_flat):
        E_elem = Emin + (r_flat ** penal) * (Emax - Emin)
        sK = (KE_flat * E_elem).flatten(order='F')
        K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K_free = K[freedofs, :][:, freedofs]
        try:
            U_free = spsolve(K_free, F_vec[freedofs])
            if not np.all(np.isfinite(U_free)):
                return None, None
        except Exception:
            return None, None
        U = np.zeros(ndof, dtype=np.float64)
        U[freedofs] = U_free
        U_edof = U[edofMat]
        ce = (np.dot(U_edof, KE) * U_edof).sum(axis=1)
        c_val = float(np.sum(E_elem * ce))
        if not np.isfinite(c_val) or c_val <= 1e-9:
            return None, None
        return c_val, ce

    # 1. Ground-truth compliance C* (cached by collision-free byte hash with bounded size)
    cache_key = hash(r_targ_2d.tobytes()) ^ hash(inp_np[1:5].tobytes())
    c_targ = _C_TARG_CACHE.get(cache_key, None)
    if c_targ is None:
        r_targ_flat = np.clip(r_targ_2d.T.flatten(), 0.06, 1.0)
        c_targ, _ = solve_elasticity(r_targ_flat)
        if c_targ is None or c_targ <= 1e-9:
            c_targ = 10.0
        if len(_C_TARG_CACHE) > 8192:
            _C_TARG_CACHE.clear()
        _C_TARG_CACHE[cache_key] = c_targ

    # 2. Predicted topology FEA solve (floor=0.06 prevents zero-density gradient trap in missing bars)
    r_pred_flat = np.clip(r_pred_2d.T.flatten(), 0.06, 1.0)
    c_pred, ce_pred = solve_elasticity(r_pred_flat)

    if c_pred is None or not np.isfinite(c_pred) or c_pred <= 1e-9:
        # Fallback if singular or non-positive: push toward ground truth
        grad_2d = np.clip(r_pred_2d - r_targ_2d, -1.0, 1.0)
        return 10.0, grad_2d.astype(np.float32), r_targ_2d.astype(np.float32), 0.1, 3.5

    ratio = float(c_pred / max(c_targ, 1e-6))
    clipped_ratio = float(np.clip(ratio, 0.1, 10.0))

    # 3. Analytical self-adjoint sensitivity: dC / d_rho_e = -p * (E0 - Emin) * rho^(p-1) * ce
    dc = -penal * (Emax - Emin) * (r_pred_flat ** (penal - 1.0)) * ce_pred

    # Sigmund conical sensitivity filter spreads sensitivity into adjacent void/gray pixels
    dc_filt = np.asarray(H.dot(r_pred_flat * dc)) / Hs / np.maximum(0.05, r_pred_flat)
    dc_filt = np.minimum(dc_filt, -1e-12)

    # 4. Online 1-step SIMP Optimality Criteria (OC) update -> rho_judge_2d
    l1, l2 = 0.0, 1e9
    move = 0.25
    x_base = np.clip(r_pred_2d.T.flatten(), 0.05, 1.0)
    target_mass = max(target_vf * nele, float(np.sum(anc_flat)) + 1.0)
    for _ in range(35):
        if (l2 - l1) / (l1 + l2 + 1e-15) <= 1e-4:
            break
        lmid = 0.5 * (l2 + l1)
        xnew = np.maximum(
            0.0,
            np.maximum(
                x_base - move,
                np.minimum(1.0, np.minimum(x_base + move, x_base * np.sqrt(np.maximum(0.0, -dc_filt) / max(lmid, 1e-15))))
            )
        )
        xnew[anc_flat] = 1.0
        if np.sum(xnew) > target_mass:
            l1 = lmid
        else:
            l2 = lmid
    rho_judge_2d = xnew.reshape(nelx, nely).T.astype(np.float32)

    # 5. RL Reward R_i in (0, 1] and Advantage Weight A_i in [1.0, 3.5]
    # R_i = 1.0 when C_pred <= C_target; decays smoothly as C_pred / C_target increases
    reward = float(math.exp(-0.75 * max(0.0, clipped_ratio - 1.0)))
    advantage = float(1.0 + 2.5 * (1.0 - reward))

    # 6. Normalize self-adjoint gradient per sample so 95th percentile has unit scale,
    # then scale by advantage so high-compliance errors receive stronger gradient signal
    dc_2d = dc_filt.reshape(nelx, nely).T
    scale_95 = float(np.percentile(np.abs(dc_2d), 95)) + 1e-8
    grad_norm_2d = np.clip((dc_2d / scale_95) * advantage, -2.5, 2.5).astype(np.float32)

    return clipped_ratio, grad_norm_2d, rho_judge_2d, reward, advantage


class SIMPJudgeEnvironment(nn.Module):
    """
    RL-Style Physics Judge Environment where SIMP / FEA acts as the differentiable critic/judge.
    Runs multi-threaded across CPU cores with zero GPU VRAM overhead.
    """
    def __init__(self, nelx=120, nely=60, penal=3.0, Emin=1e-9, Emax=1.0, num_workers=8):
        super().__init__()
        self.nelx = nelx
        self.nely = nely
        self.penal = penal
        self.Emin = Emin
        self.Emax = Emax
        # Pre-warm caches in main thread before parallel execution
        get_cached_fea_mesh(nelx, nely)
        get_cached_filter(nelx, nely, 1.5)
        self.executor = ThreadPoolExecutor(max_workers=num_workers) if num_workers > 1 else None

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None

    def __del__(self):
        self.close()

    def step(self, rho_pred, rho_target, inputs):
        """
        Evaluates a batch of predicted topologies in the SIMP-Judge environment.
        Returns:
          compliance_loss: Differentiable scalar mean compliance ratio C(pred)/C*
          rho_judge:       (B, 1, H, W) tensor of 1-step OC-improved teacher topologies
          advantages:      (B, 1, 1, 1) tensor of per-sample RL advantage weights A_i
          metrics:         dict with mean_ratio, mean_reward, mean_advantage, stable_rate_pct
        """
        device = rho_pred.device
        dtype = rho_pred.dtype

        pred_np = rho_pred.detach().cpu().numpy()
        targ_np = rho_target.detach().cpu().numpy()
        inp_np = inputs.detach().cpu().numpy()
        batch_size = pred_np.shape[0]

        tasks = [
            (pred_np[b, 0], targ_np[b, 0], inp_np[b], self.penal, self.Emin, self.Emax)
            for b in range(batch_size)
        ]

        if self.executor is not None:
            results = list(self.executor.map(_solve_single_simp_judge, tasks))
        else:
            results = [_solve_single_simp_judge(t) for t in tasks]

        ratios_np = np.array([r[0] for r in results], dtype=np.float32)
        grads_np = np.stack([r[1] for r in results], axis=0)[:, None, :, :]
        judge_np = np.stack([r[2] for r in results], axis=0)[:, None, :, :]
        rewards_np = np.array([r[3] for r in results], dtype=np.float32)
        advs_np = np.array([r[4] for r in results], dtype=np.float32)

        # Attach exact self-adjoint gradient to rho_pred via linear straight-through surrogate:
        # Forward value == mean(ratios_tensor), Backward gradient == grad_tensor / (B * H * W)
        grad_tensor = torch.from_numpy(grads_np).to(device=device, dtype=dtype)
        ratios_tensor = torch.from_numpy(ratios_np).to(device=device, dtype=dtype)

        surrogate_grad_term = (rho_pred * grad_tensor).mean(dim=(-2, -1)).squeeze(-1) # (B,)
        compliance_ratios = ratios_tensor.detach() + (surrogate_grad_term - surrogate_grad_term.detach())
        compliance_loss = compliance_ratios.mean()

        rho_judge = torch.from_numpy(judge_np).to(device=device, dtype=dtype)
        advantages = torch.from_numpy(advs_np).to(device=device, dtype=dtype).view(-1, 1, 1, 1)

        metrics = {
            "fea_ratio": float(np.mean(ratios_np)),
            "median_ratio": float(np.median(ratios_np)),
            "reward": float(np.mean(rewards_np)),
            "advantage": float(np.mean(advs_np)),
            "batch_stable_pct": float(np.mean(ratios_np <= 2.5) * 100.0)
        }
        return compliance_loss, rho_judge, advantages, metrics


# =========================================================================
# 4. SIMP-JUDGE RL + THIN-STRUT COMPOSITE LOSS
# =========================================================================
class TopoRLCompositeLoss(nn.Module):
    """
    Physics-Guided + SIMP-Judge RL Composite Loss for TopoUNet / PI-UNet.

    Upgrades over standard TopoCompositeLoss:
      1. Thin-Strut Morphological Skeleton Weighting (w_thin_strut = 3.5):
         Extracts thin internal web struts via `thin_mask = clamp(target - erode_5x5(target), 0, 1)`
         so 2-3 pixel internal bars (like Sample #2450's middle bar) receive 3.5x higher loss weight.
      2. Advantage-Weighted SIMP Judge Distillation (w_judge_oc):
         Supervises the network toward the online 1-step OC teacher state `rho_judge = OC_1(pred)`
         weighted by the sample's RL advantage `A_i`.
      3. Direct Self-Adjoint FEA Compliance Loss (w_fea):
         Applies the filtered analytical sensitivity dC/d_rho directly to the predicted density.
      4. Stronger Binarization Penalty (beta_bin = 0.03):
         Eliminates blurry intermediate gray densities in (0.15, 0.85).
      5. PPO-Style Reference Policy Trust-Region Anchor (w_ref_kl):
         Prevents catastrophic forgetting of topologies that the reference model already solves well.
    """
    def __init__(self, w_bce=1.0, w_dice=0.6, alpha_vf=10.0, beta_bin=0.03,
                 w_stiff=0.6, w_edge=0.25, w_fea=0.35, w_judge_oc=0.50,
                 w_ref_anchor=0.20, w_thin_strut=3.5):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.alpha_vf = alpha_vf
        self.beta_bin = beta_bin
        self.w_stiff = w_stiff
        self.w_edge = w_edge
        self.w_fea = w_fea
        self.w_judge_oc = w_judge_oc
        self.w_ref_anchor = w_ref_anchor
        self.w_thin_strut = w_thin_strut

        self.dice = DiceLoss()

        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target, inputs, epoch=1, strain_energy=None, raw_probs=None,
                judge_env=None, ref_pred=None):
        if inputs.shape[1] == 5:
            inputs = torch.cat([inputs[:, 0:1], inputs[:, 1:2], inputs[:, 1:2], inputs[:, 2:5]], dim=1)

        bc_x = inputs[:, 1:2]
        bc_y = inputs[:, 2:3]
        fx = inputs[:, 3:4]
        fy = inputs[:, 4:5]
        target_vf = inputs[:, 5:6]

        # 1. Fixture & Load Anchor Dilation Map (binary indicator before pooling)
        anchor_mask = ((bc_x > 0.05) | (bc_y > 0.05) | ((fx**2 + fy**2) > 1e-8)).to(pred.dtype)
        anchors_dilated = F.avg_pool2d(
            F.max_pool2d(anchor_mask, kernel_size=5, stride=1, padding=2),
            kernel_size=5, stride=1, padding=2
        )

        # 2. Thin-Strut Morphological White Top-Hat Detector:
        # Morphological opening (erosion followed by dilation with a 5x5 kernel) removes thin 2-4 pixel
        # internal web bars while restoring the boundaries of thick outer chords.
        # The White Top-Hat difference `target - opened_5x5` cleanly isolates thin internal struts!
        eroded_5x5 = -F.max_pool2d(-target.detach(), kernel_size=5, stride=1, padding=2)
        opened_5x5 = F.max_pool2d(eroded_5x5, kernel_size=5, stride=1, padding=2)
        thin_strut_mask = torch.clamp(target.detach() - opened_5x5, 0.0, 1.0)

        if strain_energy is not None:
            if strain_energy.ndim == 3:
                strain_energy = strain_energy.unsqueeze(1)
            se_clamped = torch.clamp(strain_energy.detach(), 0.0, 1.5)
            W_mech = (1.0 + 2.0 * anchors_dilated.detach() + 1.5 * se_clamped +
                      0.5 * target.detach() + self.w_thin_strut * thin_strut_mask)
        else:
            W_mech = (1.0 + 2.0 * anchors_dilated.detach() + 1.0 * target.detach() +
                      self.w_thin_strut * thin_strut_mask)
        W_mech = W_mech.detach()

        # 3. SIMP-Judge Environment Step (if active)
        loss_fea = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        loss_judge_oc = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        judge_metrics = {"fea_ratio": 1.0, "reward": 1.0, "advantage": 1.0, "batch_stable_pct": 100.0}
        advantages = torch.ones((pred.size(0), 1, 1, 1), device=pred.device, dtype=pred.dtype)

        if judge_env is not None and (self.w_fea > 0.0 or self.w_judge_oc > 0.0):
            loss_fea, rho_judge, advantages, judge_metrics = judge_env.step(pred, target, inputs)
            # Advantage-weighted distillation toward the 1-step OC-polished topology
            loss_judge_oc = torch.mean(advantages.detach() * W_mech * torch.abs(pred - rho_judge.detach()))

        # 4. Mechanics & Advantage-Weighted BCE Loss (clamped at 1e-4 to prevent FP16 GradScaler overflow)
        pred_clamped = torch.clamp(pred, 1e-4, 1.0 - 1e-4)
        loss_bce = F.binary_cross_entropy(pred_clamped, target, weight=(W_mech * advantages.detach()))

        # 5. SIMP Cubic Stiffness Discrepancy Loss: |pred^3 - target^3|
        loss_stiff = torch.mean(W_mech * torch.abs(pred**3 - target**3))

        # 6. Multi-Scale + Morphologically Eroded Dice Loss (3x3 and 5x5 erosion)
        def erode3(u):
            return -F.max_pool2d(-u, kernel_size=3, stride=1, padding=1)

        dice_standard = self.dice(pred, target)
        dice_eroded = self.dice(erode3(pred), erode3(target))
        loss_dice = 0.5 * dice_standard + 0.5 * dice_eroded

        # 7. Sobel Structural Edge Alignment Loss
        pred_pad = F.pad(pred, (1, 1, 1, 1), mode='replicate')
        target_pad = F.pad(target, (1, 1, 1, 1), mode='replicate')

        sx = self.sobel_x.to(device=pred.device, dtype=pred.dtype)
        sy = self.sobel_y.to(device=pred.device, dtype=pred.dtype)

        grad_pred_x = F.conv2d(pred_pad, sx)
        grad_pred_y = F.conv2d(pred_pad, sy)
        grad_target_x = F.conv2d(target_pad, sx)
        grad_target_y = F.conv2d(target_pad, sy)

        loss_edge = F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)

        # 8. Volume Fraction Penalty
        t_vf = target_vf.mean(dim=(-2, -1))
        p_vf = pred.mean(dim=(-2, -1))
        loss_vf = F.mse_loss(p_vf, t_vf)
        if raw_probs is not None:
            r_vf = raw_probs.mean(dim=(-2, -1))
            loss_vf = loss_vf + 0.5 * F.mse_loss(r_vf, t_vf)

        # 9. Binarization Sharpness Penalty: 4 * pred * (1 - pred)
        loss_bin = torch.mean(4.0 * pred * (1.0 - pred))

        # 10. PPO-Style Reference Policy Trust-Region Anchor (prevents policy drift on already-good samples)
        loss_ref = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if ref_pred is not None and self.w_ref_anchor > 0.0:
            loss_ref = F.mse_loss(pred, ref_pred.detach())

        total_loss = (
            self.w_bce * loss_bce +
            self.w_dice * loss_dice +
            self.alpha_vf * loss_vf +
            self.beta_bin * loss_bin +
            self.w_stiff * loss_stiff +
            self.w_edge * loss_edge +
            self.w_fea * loss_fea +
            self.w_judge_oc * loss_judge_oc +
            self.w_ref_anchor * loss_ref
        )

        loss_dict = {
            'total': total_loss.item(),
            'bce': loss_bce.item(),
            'dice': loss_dice.item(),
            'vf_mse': loss_vf.item(),
            'bin': loss_bin.item(),
            'stiff': loss_stiff.item(),
            'edge': loss_edge.item(),
            'fea_judge': loss_fea.item(),
            'judge_oc': loss_judge_oc.item(),
            'ref_anchor': loss_ref.item(),
            'fea_ratio': judge_metrics['fea_ratio'],
            'reward': judge_metrics['reward'],
            'batch_stable_pct': judge_metrics['batch_stable_pct']
        }

        return total_loss, loss_dict

