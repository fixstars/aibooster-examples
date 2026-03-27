#!/usr/bin/env python3
# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

"""Fake Megatron training script for testing MegatronPreset/Strategy/Evaluator.

Simulates ms-swift/Megatron training output with a simple throughput model.
Parameters are passed as CLI arguments using Megatron's actual flag names
(hyphen-separated, e.g. --tensor-model-parallel-size).
MegatronPreset injects these flags via CommandBuilder.

OOM thresholds (max MBS before OOM):
    selective: 8 // (tp * ep)
    full:     16 // (tp * ep)

TFLOP/s model:
    base      = 80 / tp
    mbs_boost = 1 + 0.2 * log2(mbs)    (larger batch -> higher throughput)
    act_factor = 0.95 if selective else 1.0
    ep_factor  = 0.95 ^ (ep - 1)       (EP communication overhead)
    tflops = base * mbs_boost * act_factor * ep_factor

Usage:
    python fake_megatron_train.py
    python fake_megatron_train.py --tensor-model-parallel-size 2 --expert-model-parallel-size 4 \
        --micro-batch-size 2 --recompute-granularity selective
"""

import argparse
import math
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fake Megatron training script")
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        default=1,
        help="Pipeline parallel size (default: 1)",
    )
    parser.add_argument(
        "--context-parallel-size",
        type=int,
        default=1,
        help="Context parallel size (default: 1)",
    )
    parser.add_argument(
        "--expert-model-parallel-size",
        type=int,
        default=1,
        help="Expert parallel size (default: 1)",
    )
    parser.add_argument(
        "--expert-tensor-parallel-size",
        type=int,
        default=1,
        help="Expert tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Micro batch size (default: 1)",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=16,
        help="Global batch size (default: 16)",
    )
    parser.add_argument(
        "--recompute-granularity",
        choices=["full", "selective"],
        default="full",
        help="Recompute granularity (default: 'full')",
    )
    parser.add_argument(
        "--recompute-method",
        choices=["uniform", "block"],
        default="uniform",
        help="Recompute method (default: 'uniform')",
    )
    parser.add_argument(
        "--recompute-num-layers",
        type=int,
        default=1,
        help="Number of layers to recompute (default: 1)",
    )
    parser.add_argument(
        "--vit-gradient-checkpointing",
        choices=["true", "false"],
        default="false",
        help="ViT gradient checkpointing (default: 'false')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tp = args.tensor_model_parallel_size
    ep = args.expert_model_parallel_size
    mbs = args.micro_batch_size
    activation = args.recompute_granularity

    time.sleep(0.05)  # simulate startup overhead

    # OOM check
    if activation == "selective":
        max_mbs = 8 // (tp * ep)
    else:
        max_mbs = 16 // (tp * ep)
    max_mbs = max(max_mbs, 1)

    if mbs > max_mbs:
        print(
            f"CUDA out of memory. (tp={tp} ep={ep} mbs={mbs} activation={activation})"
        )
        sys.exit(1)

    # TFLOP/s calculation
    base = 80.0 / tp
    mbs_boost = 1.0 + 0.2 * math.log2(mbs) if mbs > 1 else 1.0
    act_factor = 0.95 if activation == "selective" else 1.0
    ep_factor = 0.95 ** (ep - 1)
    tflops = round(base * mbs_boost * act_factor * ep_factor, 2)

    print(f"throughput per GPU (TFLOP/s/GPU): {tflops}")
    sys.exit(0)


if __name__ == "__main__":
    main()
