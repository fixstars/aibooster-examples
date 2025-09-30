# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

"""
GeneralTuner example

Basic function optimization example.
Find the minimum value of a quadratic function.
"""

from aibooster.intelligence.zenith_tune import GeneralTuner


def objective(trial, target_x=2.0, **kwargs):
    """
    Objective function to optimize

    Args:
        trial: Optuna trial object
        target_x: x value where minimum occurs (default: 2.0)
        **kwargs: Additional arguments

    Returns:
        float: Objective function value
    """
    # Define parameter search space
    x = trial.suggest_float("x", -10.0, 10.0)
    n_layers = trial.suggest_int("n_layers", 1, 5)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])

    # Quadratic function: (x - target_x)^2
    # For target_x=2.0, minimum value 0 occurs at x=2
    base_value = (x - target_x) ** 2

    # Add penalty based on n_layers (optimal: 2-3 layers)
    layer_penalty = abs(n_layers - 2.5) * 0.1

    # Add penalty based on optimizer (adam is optimal)
    optimizer_penalty = 0.0 if optimizer == "adam" else 0.2

    value = base_value + layer_penalty + optimizer_penalty

    return value


def main():
    """Main function"""
    # Initialize tuner
    tuner = GeneralTuner(
        output_dir="outputs",
        study_name="general_tuner_example3",
    )

    # Run optimization
    n_trials = 10
    tuner.optimize(objective, n_trials=n_trials)

    # Analyze results
    tuner.analyze()


if __name__ == "__main__":
    main()
