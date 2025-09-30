# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

import argparse
import json
import os
from typing import Optional

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
    return parser.parse_args()


def command_generator(trial: Trial, trial_id: int, study_dir: str, **kwargs) -> str:
    overlap_comm = trial.suggest_categorical("overlap_comm", ["true", "false"])
    contiguous_gradients = trial.suggest_categorical(
        "contiguous_gradients", ["true", "false"]
    )
    sub_group_size = trial.suggest_int("sub_group_size", 1, 1e10, log=True)
    reduce_bucket_size = trial.suggest_int("reduce_bucket_size", 1, 1e10, log=True)
    stage3_prefetch_bucket_size = trial.suggest_int(
        "stage3_prefetch_bucket_size", 1, 1e10, log=True
    )
    stage3_param_persistence_threshold = trial.suggest_int(
        "stage3_param_persistence_threshold", 1, 8000000
    )
    stage3_max_live_parameters = trial.suggest_int(
        "stage3_max_live_parameters", 1, 1e10
    )
    stage3_max_reuse_distance = trial.suggest_int("stage3_max_reuse_distance", 1, 1e10)
    per_device_batch_size = trial.suggest_categorical(
        "per_device_batch_size", [1, 2, 4, 8, 16, 32]
    )

    tuning_ds_config_path = os.path.join(study_dir, f"ds_config_{trial_id}.json")
    replace_params_to_file(
        "ds_config_template.json",
        tuning_ds_config_path,
        {
            "overlap_comm": overlap_comm,
            "contiguous_gradients": contiguous_gradients,
            "sub_group_size": sub_group_size,
            "reduce_bucket_size": reduce_bucket_size,
            "stage3_prefetch_bucket_size": stage3_prefetch_bucket_size,
            "stage3_param_persistence_threshold": stage3_param_persistence_threshold,
            "stage3_max_live_parameters": stage3_max_live_parameters,
            "stage3_max_reuse_distance": stage3_max_reuse_distance,
        },
    )

    command = f"""\
deepspeed \
--hostfile=/dev/null \
src/run_trainer.py \
config/trainer/Llama-3-8b.yaml \
dataset.data_files=data/OpenCL_API.jsonl \
training_args.logging_steps=1 \
training_args.deepspeed={tuning_ds_config_path} \
training_args.per_device_train_batch_size={per_device_batch_size} \
training_args.per_device_eval_batch_size={per_device_batch_size}\
"""
    return command


def value_extractor(log_path: str) -> Optional[float]:
    with open(log_path) as f:
        lines = f.readlines()
    for line in lines:
        if "train_runtime" in line:
            result_json = json.loads(line.replace("'", '"'))
            train_runtime = float(result_json["train_runtime"])
            return train_runtime
    return None


def main():
    args = parse_args()
    tuner = CommandOutputTuner(args.output_dir, args.study_name)
    tuner.optimize(command_generator, value_extractor, args.n_trials)
    tuner.analyze()


if __name__ == "__main__":
    main()
