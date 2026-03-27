# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

"""
Fake training script for demonstrating PresetTuner with the "demo" preset.

This script simulates a training workload whose execution time varies
depending on --batch-size, --num-workers, and OMP_NUM_THREADS.
It outputs "Total training time: X.XXs" which the demo preset's
RegexEvaluator extracts as the objective value.

Usage:
    python fake_train.py --epochs 3
    python fake_train.py --epochs 3 --batch-size 64 --num-workers 4
"""

import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Fake training script")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    return parser.parse_args()


def simulate_training(epochs, batch_size, num_workers):
    """Simulate training time based on parameters.

    The simulated time is designed so that:
    - Larger batch_size is generally faster (fewer iterations)
    - num_workers has a sweet spot around 4-8
    - OMP_NUM_THREADS has a sweet spot around 4-8
    """
    omp_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))

    # Base time per epoch
    base_time = 0.5

    # Batch size effect: larger batches are faster
    batch_factor = 32.0 / max(batch_size, 1)

    # Worker effect: sweet spot around 4-8
    worker_factor = 1.0 + 0.1 * abs(num_workers - 6)

    # Thread effect: sweet spot around 4-8
    thread_factor = 1.0 + 0.05 * abs(omp_threads - 6)

    total_time = 0.0
    for epoch in range(epochs):
        epoch_time = base_time * batch_factor * worker_factor * thread_factor
        # Add small random-like variation based on epoch number
        epoch_time *= 1.0 + 0.02 * (epoch % 3 - 1)
        time.sleep(epoch_time)
        total_time += epoch_time
        print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.3f}s")

    return total_time


def main():
    args = parse_args()

    print(
        f"Starting training: epochs={args.epochs}, "
        f"batch_size={args.batch_size}, num_workers={args.num_workers}, "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(not set)')}"
    )

    total_time = simulate_training(args.epochs, args.batch_size, args.num_workers)

    print(f"Total training time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
