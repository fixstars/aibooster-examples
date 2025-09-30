# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

"""Example script for tuning OMP_NUM_THREADS using sequential 1-job-1-trial approach."""

import argparse
import re
from typing import Optional

from optuna.trial import Trial

from aibooster.intelligence.zenith_tune.command import CommandBuilder
from aibooster.intelligence.zenith_tune.integration.kubernetes import (
    PyTorchJob,
    PyTorchJobTuner,
)


def job_converter(trial: Trial, job: PyTorchJob) -> PyTorchJob:
    """
    Update job definition with different OMP_NUM_THREADS and num_workers settings.

    Args:
        trial: Optuna trial object for suggesting parameters
        job: PyTorchJob object to update

    Returns:
        Updated PyTorchJob object
    """
    # Suggest number of threads and workers
    num_threads = trial.suggest_int("omp_num_threads", 1, 8)
    num_workers = trial.suggest_int("num_workers", 0, 4)

    # Set environment variable using convenient API
    job.set_env("OMP_NUM_THREADS", str(num_threads))

    # Update command to include num_workers argument using CommandBuilder
    current_command = job.get_command()
    assert (
        current_command
        and len(current_command) >= 3
        and current_command[0] == "sh"
        and current_command[1] == "-c"
    ), f"Expected ['sh', '-c', 'command'] format, got: {current_command}"

    # Modify only the actual command part (index 2)
    actual_command = current_command[2]
    builder = CommandBuilder(actual_command)
    builder.append(f"--num-workers {num_workers}")

    # Replace the command part while keeping sh -c wrapper
    new_command = current_command.copy()
    new_command[2] = builder.get_command()
    job.set_command(new_command)

    print(
        f"Trial {trial.number}: OMP_NUM_THREADS={num_threads}, num_workers={num_workers}"
    )

    return job


def value_extractor(log_path: str) -> Optional[float]:
    """Extract objective value from log file."""
    with open(log_path, "r") as f:
        logs = f.read()

    # Look for the line with elapsed time
    match = re.search(r"Elapsed time: ([0-9.]+) seconds", logs)
    if match:
        elapsed_time = float(match.group(1))
        return elapsed_time
    else:
        print(f"Could not find elapsed time in {log_path}")
        return None


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Tune OMP_NUM_THREADS for a PyTorchJob using sequential optimization"
    )
    parser.add_argument(
        "--job-name",
        type=str,
        required=True,
        help="Name of the PyTorchJob to use as template",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Kubernetes namespace (default: default)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=10, help="Number of trials to run (default: 10)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to the database file for Optuna study persistence (default: None)",
    )

    args = parser.parse_args()

    # Initialize the tuner
    tuner = PyTorchJobTuner(
        job_name=args.job_name,
        get_namespace=args.namespace,
        submit_namespace=args.namespace,
        db_path=args.db_path,
    )

    # Run optimization
    tuner.optimize(
        job_converter=job_converter,
        value_extractor=value_extractor,
        n_trials=args.n_trials,
    )

    # Analyze results
    tuner.analyze()


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
