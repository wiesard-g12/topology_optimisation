"""
train_and_evaluate_pipeline.py - Automated End-to-End Orchestrator
Waits for the 1000-sample dataset generation to finish, then trains TopoUNet,
benchmarks performance, and refreshes all demonstration figures.
"""

import os
import time
import argparse
import numpy as np
import torch

from train_unet import train as run_training
from predict_unet import benchmark_and_evaluate
from interactive_demo import run_demo


def wait_for_dataset(dataset_path="topo_dataset_1000.npz", target_samples=1000, check_interval=10):
    print(f"Monitoring dataset generation for {dataset_path} (Target: {target_samples} samples)...", flush=True)
    last_count = 0
    t0 = time.time()
    
    while True:
        if os.path.exists(dataset_path):
            try:
                data = np.load(dataset_path)
                count = len(data['inputs'])
                if count != last_count:
                    rate = count / (time.time() - t0) if (time.time() - t0) > 0 else 0
                    print(f"  [Dataset Progress] {count}/{target_samples} samples generated ({count/target_samples*100:.1f}%) | {rate:.2f} samples/s", flush=True)
                    last_count = count
                if count >= target_samples:
                    print(f"Target count {target_samples} reached successfully!", flush=True)
                    break
            except Exception:
                pass
        time.sleep(check_interval)


def main():
    parser = argparse.ArgumentParser(description="Automated Train & Evaluation Pipeline")
    parser.add_argument("--dataset", type=str, default="topo_dataset_1000.npz")
    parser.add_argument("--target_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--wait", action="store_true", help="Wait for dataset generation to reach target")
    args = parser.parse_args()
    
    if args.wait:
        wait_for_dataset(args.dataset, target_samples=args.target_samples)
        
    print("\n" + "=" * 65, flush=True)
    print(f"  Starting TopoUNet Training on {args.dataset}", flush=True)
    print("=" * 65, flush=True)
    
    # 1. Train model
    class TrainArgs:
        dataset = args.dataset
        epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr
        val_split = 0.15
        base_filters = 64
        w_bce = 1.0
        w_dice = 0.5
        alpha_vf = 10.0
        beta_bin = 1e-3
        no_augment = False
        cpu = False
        save_dir = args.save_dir

    run_training(TrainArgs)
    
    # 2. Benchmark & Evaluate
    best_ckpt = os.path.join(args.save_dir, "best_unet.pth")
    print("\n" + "=" * 65, flush=True)
    print("  Evaluating Model on Test Split & Generating Publication Visualizations", flush=True)
    print("=" * 65, flush=True)
    
    benchmark_and_evaluate(
        checkpoint_path=best_ckpt,
        dataset_path=args.dataset,
        output_dir="images",
        num_visualizations=5,
        device_str='auto'
    )
    
    # 3. Refresh Benchmark Demos
    print("\n" + "=" * 65, flush=True)
    print("  Refreshing Interactive Benchmark Demonstration Figures", flush=True)
    print("=" * 65, flush=True)
    for case in ['cantilever', 'mbb', 'bridge', 'crane']:
        run_demo(case_name=case, volfrac=0.40, checkpoint_path=best_ckpt)
        
    print("\n" + "=" * 65, flush=True)
    print("  Pipeline Completed Successfully! All Models & Demos Refreshed.", flush=True)
    print("=" * 65, flush=True)


if __name__ == '__main__':
    main()
