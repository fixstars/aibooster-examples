"""
Example of using AIBoosterGPUMemoryUsedPruner with CommandOutputTuner.

This example demonstrates how to use AIBoosterGPUMemoryUsedPruner to automatically
terminate training jobs that consume excessive GPU memory, which is useful for
hyperparameter optimization where batch size, model size, or other parameters
can cause memory allocation failures.
"""

import socket

from aibooster.intelligence.zenith_tune.auto_pruners import AIBoosterGPUMemoryUsedPruner
from aibooster.intelligence.zenith_tune.tuners.command_output import CommandOutputTuner

TRAINING_SCRIPT = """
import torch
import sys

# Get hyperparameters from command line
batch_size = int(sys.argv[1])
model_dim = int(sys.argv[2])
num_layers = int(sys.argv[3])

device = torch.device("cuda:0")

# Create model layers
layers = []
for i in range(num_layers):
    layer = torch.nn.Linear(model_dim, model_dim).to(device)
    layers.append(layer)

# Create training data batch
inputs = torch.randn(batch_size, model_dim).to(device)
targets = torch.randn(batch_size, model_dim).to(device)

# Training loop
for epoch in range(100):
    x = inputs
    for layer in layers:
        x = torch.relu(layer(x))

    loss = torch.nn.functional.mse_loss(x, targets)
    loss.backward()

    for layer in layers:
        layer.zero_grad()

    print(f"Epoch {epoch}/100, Loss: {loss.item():.4f}")

# Output dummy performance metric (larger model = better performance)
# This is a synthetic metric for demonstration purposes
performance = batch_size * model_dim * num_layers / 10000.0
print("Performance: " + str(performance))
"""


def command_generator(trial, **kwargs):
    """Generate a training command with memory-dependent hyperparameters."""
    batch_size = trial.suggest_int("batch_size", 512, 8192, step=512)
    model_dim = trial.suggest_int("model_dim", 512, 8192, step=512)
    num_layers = trial.suggest_int("num_layers", 4, 128, step=4)

    return f"python -c '{TRAINING_SCRIPT}' {batch_size} {model_dim} {num_layers}"


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
    """Run hyperparameter optimization with GPU memory monitoring."""
    # Create GPU memory pruner
    current_hostname = socket.gethostname()
    gpu_memory_pruner = AIBoosterGPUMemoryUsedPruner(
        aibooster_server_address="http://localhost:16697",
        threshold=12800.0,  # Prune if GPU memory usage > 12800 MB
        agent_gpu_filter={
            current_hostname: [0],  # Monitor GPU 0 on current host
        },
    )

    # Create tuner with GPU memory pruner
    tuner = CommandOutputTuner(
        output_dir="outputs",
        study_name="gpu_memory_pruner_example",
        auto_pruners=[gpu_memory_pruner],
        maximize=True,
    )

    # Run optimization
    best_value, best_params = tuner.optimize(
        command_generator=command_generator,
        value_extractor=value_extractor,
        n_trials=10,
    )

    print("Optimization completed!")
    print("Best performance: " + str(best_value))
    print("Best hyperparameters: " + str(best_params))


if __name__ == "__main__":
    main()
