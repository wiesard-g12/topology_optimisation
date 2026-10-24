"""
evaluate_stability_100_rl_ver.py - 100-Sample Random Structural Stability Benchmark (RL Version)
Copy of evaluate_stability_100.py configured to evaluate checkpoints/best_unet_rl.pth
against the exact same 100 random validation datapoints (seed=42).

Outputs:
  - Visual 6-panel scorecard: images/stability_100_rl_report.png
  - Full JSON diagnostics:    checkpoints/stability_100_rl_report.json
"""

import argparse
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from unet_model import TopoUNet
from losses_rl_ver import check_structural_stability, physics_aware_binarize


def load_model(checkpoint_path="checkpoints/best_unet_rl.pth", device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/best_unet_3000.pth"
    model = TopoUNet(in_channels=6, out_channels=1)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device, checkpoint_path


def plot_100_stability_report(records, summary, output_path="images/stability_100_rl_report.png"):
    fig = plt.figure(figsize=(20, 11), dpi=160)
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.25)

    # 1. Sub-criterion & Overall Stability Accuracy Bar Chart
    ax0 = fig.add_subplot(gs[0, 0])
    categories = [
        'Load Path\nConnected',
        'FEA Stiffness\n(No Hinge)',
        'Yield Stress\nSafe (FoS)',
        'Buckling\nStable (λ₁≥1)',
        'OVERALL\nSTABLE'
    ]
    accs = [
        summary['connectivity_accuracy_pct'],
        summary['stiffness_accuracy_pct'],
        summary['yielding_accuracy_pct'],
        summary['buckling_accuracy_pct'],
        summary['overall_stability_accuracy_pct']
    ]
    colors = ['#1976D2', '#0288D1', '#7B1FA2', '#F57C00', '#2E7D32']
    bars = ax0.bar(categories, accs, color=colors, width=0.58, edgecolor='k', lw=0.8)
    ax0.set_ylim(0, 112)
    ax0.axhline(100.0, color='gray', linestyle=':', lw=1.2)
    ax0.set_ylabel("Accuracy / Pass Rate (%)", fontsize=10, fontweight='bold')
    ax0.set_title("1. RL-Trained Stability Accuracy Breakdown (N=100)", fontsize=11, fontweight='bold', pad=8)
    ax0.grid(True, axis='y', linestyle=':', alpha=0.6)
    for b, val in zip(bars, accs):
        ax0.annotate(f"{val:.1f}%", (b.get_x() + b.get_width() / 2, val + 2.0),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. FEA Compliance Ratio Distribution (Stiffness Parity)
    ax1 = fig.add_subplot(gs[0, 1])
    ratios = [min(r['fea_ratio'], 5.0) for r in records]
    ax1.hist(ratios, bins=np.linspace(0.8, 5.0, 22), color='#0288D1', edgecolor='white', alpha=0.85)
    ax1.axvline(2.5, color='#D32F2F', linestyle='--', lw=2, label='Stability Limit (2.5x)')
    ax1.axvline(summary['median_fea_ratio'], color='#2E7D32', linestyle='-', lw=2,
                label=f"Median Ratio: {summary['median_fea_ratio']:.2f}x")
    ax1.set_title("2. FEA Compliance Ratio Distribution ($C_{pred} / C^*$)", fontsize=11, fontweight='bold', pad=8)
    ax1.set_xlabel("Compliance Ratio (Clipped at 5.0x)", fontsize=10)
    ax1.set_ylabel("Sample Count", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 3. Yield Factor of Safety (FoS) Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    fos_vals = [min(r['fos_yield'], 10.0) for r in records]
    ax2.hist(fos_vals, bins=np.linspace(0, 10, 21), color='#7B1FA2', edgecolor='white', alpha=0.85)
    ax2.axvline(1.0, color='#D32F2F', linestyle='--', lw=2, label='Yield Break Limit (FoS=1.0)')
    ax2.axvline(summary['median_fos_yield'], color='#2E7D32', linestyle='-', lw=2,
                label=f"Median FoS: {summary['median_fos_yield']:.2f}")
    ax2.set_title("3. Material Yield Factor of Safety (FoS)", fontsize=11, fontweight='bold', pad=8)
    ax2.set_xlabel("Yield Factor of Safety (Clipped at 10)", fontsize=10)
    ax2.set_ylabel("Sample Count", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 4. In-Plane Buckling Multiplier (lambda_1) Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    lam_vals = [min(r['lambda_1'], 25.0) for r in records]
    ax3.hist(lam_vals, bins=np.linspace(0, 25, 21), color='#F57C00', edgecolor='white', alpha=0.85)
    ax3.axvline(1.0, color='#D32F2F', linestyle='--', lw=2, label=r'Critical Buckling Limit ($\lambda_1=1.0$)')
    ax3.set_title(r"4. In-Plane Buckling Load Factor ($\lambda_1$)", fontsize=11, fontweight='bold', pad=8)
    ax3.set_xlabel(r"Critical Buckling Multiplier $\lambda_1$ (Clipped at 25)", fontsize=10)
    ax3.set_ylabel("Sample Count", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # 5. Dice Score vs Stability Scatter Plot
    ax4 = fig.add_subplot(gs[1, 1])
    stable_pts = [r for r in records if r['is_stable']]
    unstable_pts = [r for r in records if not r['is_stable']]
    if stable_pts:
        ax4.scatter([r['dice'] for r in stable_pts],
                    [min(r['fea_ratio'], 5.0) for r in stable_pts],
                    c='#2E7D32', s=45, alpha=0.8, edgecolors='k', lw=0.5,
                    label=f"STABLE ({len(stable_pts)} samples)")
    if unstable_pts:
        ax4.scatter([r['dice'] for r in unstable_pts],
                    [min(r['fea_ratio'], 5.0) for r in unstable_pts],
                    c='#D32F2F', marker='X', s=70, alpha=0.9, edgecolors='k', lw=0.6,
                    label=f"CAN BREAK ({len(unstable_pts)} samples)")
    ax4.axhline(2.5, color='#D32F2F', linestyle='--', lw=1.5)
    ax4.set_title("5. Dice Overlap vs. FEA Stiffness Ratio", fontsize=11, fontweight='bold', pad=8)
    ax4.set_xlabel("Binarized Dice Similarity Score", fontsize=10)
    ax4.set_ylabel("Compliance Ratio $C_{pred}/C^*$ (Clipped at 5x)", fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, linestyle=':', alpha=0.6)

    # 6. Summary Verdict Card
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    card_text = (
        f"SIMP-JUDGE RL 100-SAMPLE VERDICT\n"
        f"{'='*40}\n\n"
        f"• Overall Stable (Will Not Break):  {summary['overall_stability_accuracy_pct']:.1f}% ({summary['n_stable']}/{summary['num_samples']})\n"
        f"• Can Break / Unstable:             {100.0 - summary['overall_stability_accuracy_pct']:.1f}% ({summary['n_unstable']}/{summary['num_samples']})\n\n"
        f"SUB-CRITERION ACCURACIES:\n"
        f"  1. Load-to-Support Connected:     {summary['connectivity_accuracy_pct']:.1f}%\n"
        f"  2. Stiffness Stability (≤2.5x):   {summary['stiffness_accuracy_pct']:.1f}%\n"
        f"  3. Yield Stress Safe (FoS≥1.0):   {summary['yielding_accuracy_pct']:.1f}%\n"
        f"  4. Buckling Stable (λ₁≥1.0):      {summary['buckling_accuracy_pct']:.1f}%\n\n"
        f"QUANTITATIVE BENCHMARKS:\n"
        f"  • Mean Dice Score:                {summary['mean_dice']:.4f}\n"
        f"  • Median Compliance Ratio:        {summary['median_fea_ratio']:.2f}x\n"
        f"  • Median Yield FoS:               {summary['median_fos_yield']:.2f}\n"
        f"  • Median Buckling λ₁:             {summary['median_lambda_1']:.2f}\n"
        f"  • Mean GPU Inference Latency:     {summary['mean_latency_ms']:.2f} ms"
    )
    ax5.text(0.03, 0.96, card_text, transform=ax5.transAxes, fontsize=10,
             va='top', ha='left', family='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F7FA', edgecolor='#2E7D32', lw=2.0))

    fig.suptitle(
        f"TopoUNet (SIMP-Judge RL) Structural Stability Audit — 100 Random Datapoints\n"
        f"Overall Stability Accuracy: {summary['overall_stability_accuracy_pct']:.1f}% STABLE | "
        f"Connectivity: {summary['connectivity_accuracy_pct']:.1f}% | "
        f"Yield Safety: {summary['yielding_accuracy_pct']:.1f}% | "
        f"Buckling Stability: {summary['buckling_accuracy_pct']:.1f}%",
        fontsize=13.5, fontweight='bold', y=0.99
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[Saved Visual Report] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL-Trained TopoUNet Stability Accuracy on 100 Random Samples")
    parser.add_argument("--dataset", type=str, default="topo_dataset_3000.npz", help="Path to .npz dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_unet_rl.pth", help="Path to RL model checkpoint")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of random samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--threshold", type=float, default=0.35, help="Binarization cutoff threshold")
    parser.add_argument("--val_only", action="store_true", default=True, help="Sample from held-out validation split")
    parser.add_argument("--split_mode", type=str, default="tail", choices=["tail", "random_split"],
                        help="Validation pool selection: 'tail' (last 15%, matches baseline benchmark) or 'random_split' (matches dataset.py seed=42)")
    args = parser.parse_args()

    model, device, ckpt_used = load_model(args.checkpoint)
    data = np.load(args.dataset)
    inputs_all = data['inputs']
    targets_all = data['targets']
    total_n = len(inputs_all)

    rng = np.random.default_rng(args.seed)
    if args.val_only and total_n >= 200:
        if args.split_mode == "random_split":
            n_val = int(total_n * 0.15)
            n_train = total_n - n_val
            _, val_subset = torch.utils.data.random_split(
                range(total_n), [n_train, n_val],
                generator=torch.Generator().manual_seed(42)
            )
            candidate_indices = np.array(list(val_subset.indices), dtype=int)
        else:
            val_start = int(total_n * 0.85)
            candidate_indices = np.arange(val_start, total_n)
    else:
        candidate_indices = np.arange(total_n)
    if len(candidate_indices) < args.num_samples:
        candidate_indices = np.arange(total_n)

    chosen_indices = np.sort(rng.choice(candidate_indices, size=args.num_samples, replace=False))

    print("=" * 80)
    print("  TOPOUNET (SIMP-JUDGE RL) 100-SAMPLE STRUCTURAL STABILITY BENCHMARK")
    print(f"  Dataset:    {args.dataset} (Total={total_n}, Pool=[{candidate_indices[0]}..{candidate_indices[-1]}])")
    print(f"  Checkpoint: {ckpt_used} on {device}")
    print(f"  Samples:    {args.num_samples} random datapoints (seed={args.seed}, threshold={args.threshold})")
    print("=" * 80)

    dummy = torch.zeros(1, 6, 60, 120, device=device)
    with torch.inference_mode():
        for _ in range(5):
            _ = model(dummy)

    records = []

    for idx_pos, sample_idx in enumerate(chosen_indices, start=1):
        inp = inputs_all[sample_idx]
        simp = targets_all[sample_idx]

        x_tensor = torch.from_numpy(inp).unsqueeze(0).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            pred_tensor = model(x_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred = pred_tensor.squeeze().cpu().numpy()
        bin_pred = physics_aware_binarize(pred, inp, threshold=args.threshold, polish_steps=5)
        bin_simp = (simp >= 0.5).astype(float)

        inter = np.sum(bin_pred * bin_simp)
        dice = float((2.0 * inter + 1e-6) / (np.sum(bin_pred) + np.sum(bin_simp) + 1e-6))

        stab = check_structural_stability(
            pred_density=pred,
            inp_np=inp,
            simp_density=simp,
            threshold=args.threshold,
            bin_pred_override=bin_pred
        )

        rec = {
            "sample_idx": int(sample_idx),
            "dice": dice,
            "latency_ms": float(latency_ms),
            **stab
        }
        records.append(rec)

        if idx_pos % 10 == 0 or idx_pos == args.num_samples:
            n_ok = sum(1 for r in records if r['is_stable'])
            print(f"  [{idx_pos:3d}/{args.num_samples}] Running Stability Accuracy: "
                  f"{n_ok}/{idx_pos} ({100.0*n_ok/idx_pos:.1f}%) | "
                  f"Last Sample #{sample_idx}: {stab['short_badge']} "
                  f"(Ratio={stab['fea_ratio']:.2f}x, FoS={stab['fos_yield']:.2f}, lambda_1={stab['lambda_1']:.1f})",
                  flush=True)

    n = len(records)
    n_stable = sum(1 for r in records if r['is_stable'])
    n_unstable = n - n_stable
    n_conn = sum(1 for r in records if r['is_connected'])
    n_stiff = sum(1 for r in records if r['stiff_pass'])
    n_yield = sum(1 for r in records if r['yield_pass'])
    n_buckle = sum(1 for r in records if r['buckle_pass'])

    finite_ratios = [r['fea_ratio'] for r in records if np.isfinite(r['fea_ratio'])]
    finite_fos = [r['fos_yield'] for r in records if np.isfinite(r['fos_yield'])]
    finite_lam = [r['lambda_1'] for r in records if np.isfinite(r['lambda_1'])]

    summary = {
        "num_samples": n,
        "seed": args.seed,
        "threshold": args.threshold,
        "checkpoint": ckpt_used,
        "n_stable": n_stable,
        "n_unstable": n_unstable,
        "overall_stability_accuracy_pct": 100.0 * n_stable / n,
        "connectivity_accuracy_pct": 100.0 * n_conn / n,
        "stiffness_accuracy_pct": 100.0 * n_stiff / n,
        "yielding_accuracy_pct": 100.0 * n_yield / n,
        "buckling_accuracy_pct": 100.0 * n_buckle / n,
        "mean_dice": float(np.mean([r['dice'] for r in records])),
        "std_dice": float(np.std([r['dice'] for r in records])),
        "median_fea_ratio": float(np.median(finite_ratios)) if finite_ratios else float('inf'),
        "mean_fea_ratio_stable": float(np.mean([r['fea_ratio'] for r in records if r['is_stable']])) if n_stable > 0 else 0.0,
        "median_fos_yield": float(np.median(finite_fos)) if finite_fos else 0.0,
        "median_lambda_1": float(np.median(finite_lam)) if finite_lam else 0.0,
        "mean_latency_ms": float(np.mean([r['latency_ms'] for r in records])),
        "unstable_sample_indices": [r['sample_idx'] for r in records if not r['is_stable']],
        "failure_breakdown": {
            "disconnected_load_path": n - n_conn,
            "excessive_compliance_or_hinge": n - n_stiff,
            "yield_stress_overload": n - n_yield,
            "buckling_instability": n - n_buckle
        }
    }

    print("\n" + "=" * 80)
    print("  FINAL 100-SAMPLE STRUCTURAL STABILITY ACCURACY REPORT (RL VERSION)")
    print("=" * 80)
    print(f"  1. Load-to-Support Connectivity Accuracy:  {summary['connectivity_accuracy_pct']:6.2f}% ({n_conn}/{n})")
    print(f"  2. FEA Stiffness Stability Accuracy:       {summary['stiffness_accuracy_pct']:6.2f}% ({n_stiff}/{n})")
    print(f"  3. Yield Stress Safety Accuracy (FoS):     {summary['yielding_accuracy_pct']:6.2f}% ({n_yield}/{n})")
    print(f"  4. In-Plane Buckling Stability Accuracy:   {summary['buckling_accuracy_pct']:6.2f}% ({n_buckle}/{n})")
    print("-" * 80)
    print(f"  >> OVERALL STRUCTURAL STABILITY ACCURACY:  {summary['overall_stability_accuracy_pct']:6.2f}% ({n_stable}/{n} STABLE)")
    print(f"  >> UNSTABLE / CAN BREAK RATE:              {100.0 - summary['overall_stability_accuracy_pct']:6.2f}% ({n_unstable}/{n} CAN BREAK)")
    print("-" * 80)
    print(f"  Mean Binarized Dice Score:                 {summary['mean_dice']:.4f} +/- {summary['std_dice']:.4f}")
    print(f"  Median FEA Compliance Ratio (C_pred/C*):   {summary['median_fea_ratio']:.2f}x (Stable Mean: {summary['mean_fea_ratio_stable']:.2f}x)")
    print(f"  Median Yield Factor of Safety (FoS):       {summary['median_fos_yield']:.2f}")
    print(f"  Median Buckling Load Factor (lambda_1):    {summary['median_lambda_1']:.2f}")
    print(f"  Mean GPU Inference Latency:                {summary['mean_latency_ms']:.2f} ms")
    if summary['unstable_sample_indices']:
        print(f"  Unstable Sample IDs:                       {summary['unstable_sample_indices']}")
    print("=" * 80)

    os.makedirs("checkpoints", exist_ok=True)
    json_path = "checkpoints/stability_100_rl_report.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"[Saved JSON Report] {json_path}")

    plot_100_stability_report(records, summary, output_path="images/stability_100_rl_report.png")


if __name__ == "__main__":
    main()
