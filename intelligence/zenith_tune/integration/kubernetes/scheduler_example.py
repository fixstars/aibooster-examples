# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

"""Run PyTorchJob tuning scheduler with annotation-based configuration."""

from aibooster.intelligence.zenith_tune.integration.kubernetes import (
    JobFilter,
    PyTorchJobTuningScheduler,
    TuningConfig,
    TuningRule,
)

# Only target jobs with zenith-tune/optimization-config annotation
tuning_rules = [
    TuningRule(
        job_filter=JobFilter(
            annotations={"zenith-tune/optimization-config": None}  # Key existence check
        ),
        tuning_config=TuningConfig(),
    )
]

scheduler = PyTorchJobTuningScheduler(
    tuning_rules=tuning_rules,
    max_concurrent_tuning_per_namespace=2,
)
scheduler.run()
