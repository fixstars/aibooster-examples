"""Run PyTorchJob tuning scheduler with annotation-based configuration."""

from aibooster.intelligence.zenith_tune.integration.kubernetes import (
    JobFilter,
    PyTorchJobTuningScheduler,
)

# Only target jobs with zenith-tune/optimization-config annotation
job_filter = JobFilter(
    annotations={"zenith-tune/optimization-config": None}  # Key existence check
)

scheduler = PyTorchJobTuningScheduler(
    submit_namespace="default",
    job_filter=job_filter,
    max_concurrent_tuning=1,
)
scheduler.run()
