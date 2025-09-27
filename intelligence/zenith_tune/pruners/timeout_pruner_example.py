"""
Example of using TimeoutPruner with CommandOutputTuner.

This example demonstrates how to use TimeoutPruner to automatically
terminate training jobs that exceed a specified time limit, which is useful for
hyperparameter optimization where certain parameter combinations might
cause extremely slow convergence or infinite loops.
"""

from aibooster.intelligence.zenith_tune import CommandOutputTuner
from aibooster.intelligence.zenith_tune.auto_pruners import TimeoutPruner

COMPUTE_SCRIPT = """
import time
import sys

# Get parameters from command line
num_tasks = int(sys.argv[1])
task_duration = float(sys.argv[2])

print(f"Processing {num_tasks} tasks, each taking {task_duration} seconds...")

total_time = 0
for i in range(num_tasks):
    print(f"Task {i+1}/{num_tasks}")
    time.sleep(task_duration)
    total_time += task_duration

# Output performance metric (lower total time = better performance)
performance = 1.0 / total_time if total_time > 0 else 0.0
print("Performance: " + str(performance))
"""


def command_generator(trial, **_):
    """Generate a command with time-dependent parameters."""
    num_tasks = trial.suggest_int("num_tasks", 0, 100)
    task_duration = trial.suggest_float("task_duration", 0.0, 1.0)

    return f"python -c '{COMPUTE_SCRIPT}' {num_tasks} {task_duration}"


def value_extractor(log_path):
    """Extract performance metric from training log."""
    try:
        with open(log_path) as f:
            content = f.read()

        # Look for the performance metric in the log
        for line in content.split("\n"):
            if "Performance:" in line:
                performance = float(line.split("Performance:")[1].strip())
                return performance

    except Exception as e:
        print("Error reading log file: " + str(e))

    return None


def main():
    """Run parameter optimization with timeout monitoring."""
    # Create timeout pruner (10 seconds timeout)
    timeout_pruner = TimeoutPruner(timeout=10.0)

    # Create tuner with timeout pruner
    tuner = CommandOutputTuner(
        output_dir="outputs",
        study_name="timeout_pruner_example",
        auto_pruners=[timeout_pruner],
        maximize=True,
    )

    # Run optimization
    best_value, best_params = tuner.optimize(
        command_generator=command_generator,
        value_extractor=value_extractor,
        n_trials=15,
    )

    print("Optimization completed!")
    print("Best performance: " + str(best_value))
    print("Best hyperparameters: " + str(best_params))


if __name__ == "__main__":
    main()
