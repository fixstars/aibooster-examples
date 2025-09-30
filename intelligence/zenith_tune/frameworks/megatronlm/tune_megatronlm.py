# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

import argparse
import os
import re
import statistics
from typing import Dict, Optional, Union

import optunahub
import torch
from optuna.trial import Trial

from aibooster.intelligence.zenith_tune import CommandOutputTuner
from aibooster.intelligence.zenith_tune.utils import replace_params_to_file


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
    parser.add_argument(
        "--use-ingo",
        action="store_true",
        help="Use ingo sampler for optimization.",
    )
    return parser.parse_args()


def command_generator(
    trial: Trial,
    trial_id: int,
    study_dir: str,
    dist_info: Dict[str, Union[int, str]],
    **kwargs,
) -> str:
    global_batch_size = 1024
    micro_batch_size = trial.suggest_int("micro_batch_size", low=1, high=4)
    sequence_length = trial.suggest_int(
        "sequence_length", low=1024, high=8192, step=1024
    )
    tensor_model_parallel_size = trial.suggest_categorical(
        "tensor_model_parallel_size", [1, 2, 4, 8]
    )
    world_size = dist_info["world_size"]
    if world_size % tensor_model_parallel_size != 0:
        return None
    data_parallel_size = world_size // tensor_model_parallel_size
    if global_batch_size % data_parallel_size != 0:
        return None
    maximum_micro_batches = global_batch_size // data_parallel_size
    if maximum_micro_batches % micro_batch_size != 0:
        return None

    tuning_script_path = os.path.join(study_dir, f"train_{trial_id}.sh")
    if dist_info["rank"] == 0:
        replace_params_to_file(
            "train_gpt3_7b_mpi_template.sh",
            tuning_script_path,
            {
                "micro_batch_size": micro_batch_size,
                "sequence_length": sequence_length,
                "tensor_model_parallel_size": tensor_model_parallel_size,
                "global_batch_size": global_batch_size,
            },
        )
    torch.distributed.barrier()  # wait creating script

    command = f"bash {tuning_script_path} ./tensorboard ./gpt2/vocab.json ./gpt2/merges.txt ./arxiv_text_document"
    return command


def value_extractor(log_path: str) -> Optional[float]:
    with open(log_path) as f:
        lines = f.readlines()

    sample_flops = []
    for line in lines:
        if "throughput per GPU" in line:
            match = re.findall(
                r"iteration\s+(\d+)/\s+\d+.*throughput per GPU \(TFLOP/s/GPU\): (\d+\.\d+)",
                line,
            )[0]
            iteration = int(match[0])
            tflops_per_gpu = float(match[1])
            if 20 <= iteration and iteration <= 100:
                sample_flops.append(tflops_per_gpu)
    return statistics.harmonic_mean(sample_flops)


def main():
    args = parse_args()

    torch.distributed.init_process_group("mpi")

    sampler = None
    if args.use_ingo:
        mod = optunahub.load_module("samplers/implicit_natural_gradient")
        sampler = mod.ImplicitNaturalGradientSampler(sigma0=0.5, seed=0)

    tuner = CommandOutputTuner(
        args.output_dir, args.study_name, sampler=sampler, maximize=True
    )
    tuner.optimize(command_generator, value_extractor, args.n_trials)
    tuner.analyze()


if __name__ == "__main__":
    main()
