# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

import argparse
import os
import re
from typing import Dict, Optional, Union

from optuna.trial import Trial

from aibooster.intelligence.zenith_tune import CommandOutputTuner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--study-name")
    parser.add_argument(
        "--n-trials",
        default=10,
        help="The number of optimization steps.",
        type=int,
    )
    return parser.parse_args()


def command_generator(
    trial: Trial,
    dist_info: Dict[str, Union[int, str]],
    **kwargs,
) -> str:
    omp_num_threads = trial.suggest_categorical("omp_num_threads", [1, 2, 4, 8, 16, 32])
    num_workers = trial.suggest_int("num_workers", low=4, high=16)
    prefetch_factor = trial.suggest_int("prefetch_factor", low=1, high=10)
    available_cpus = trial.suggest_int("available_cpus", low=1, high=14)
    os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    # Limit iterations to 100 for optimization
    batch_size = 32
    limited_iterations = 100
    limited_samples = limited_iterations * batch_size * dist_info["world_size"]

    command = f"""\
python src/train.py src/resnet_config.py --cfg-options \
train_dataloader.num_workers={num_workers} \
train_dataloader.prefetch_factor={prefetch_factor} \
train_dataloader.sampler.type=zenith_tune.LimitedSampler \
train_dataloader.sampler.limited_samples={limited_samples} \
train_dataloader.worker_init_fn.type=zenith_tune.worker_affinity_init_fn \
train_dataloader.worker_init_fn.available_cpus={available_cpus}\
"""
    return command


def value_extractor(log_path: str) -> Optional[float]:
    with open(log_path) as f:
        lines = f.readlines()
    time_samples = []
    for line in lines:
        if "Epoch(train) [1]" in line:
            match = re.findall(
                r"Epoch\(train\) \[1\]\[\s*(\d+)/.*time: (\d+\.\d+)  data_time", line
            )[0]
            iterations = int(match[0])
            time_val = float(match[1])
            if 10 < iterations and iterations <= 100:
                time_samples.append(time_val)
    if len(time_samples) > 0:
        return sum(time_samples)
    else:
        return None


def main():
    args = parse_args()
    tuner = CommandOutputTuner(args.output_dir, args.study_name)
    tuner.optimize(
        command_generator,
        value_extractor,
        args.n_trials,
        default_params={
            "omp_num_threads": 16,
            "num_workers": 8,
            "prefetch_factor": 2,
            "available_cpus": 14,
        },
    )
    tuner.analyze()


if __name__ == "__main__":
    main()
