# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

"""AIBooster Client usage example."""

from datetime import datetime, timedelta, timezone

from aibooster.intelligence.zenith_tune.integration.aibooster import AIBoosterClient


def main():
    # Initialize client
    client = AIBoosterClient("http://localhost:16697")

    # Get data from last hour (UTC)
    end_time = datetime.now(timezone.utc)
    begin_time = end_time - timedelta(hours=1)

    # Get mean GPU utilization for the last hour
    value = client.get_dcgm_metrics_reduction(
        "DCGM_FI_DEV_GPU_UTIL", "mean", begin_time=begin_time, end_time=end_time
    )

    if value is not None:
        print(f"Average GPU utilization (last hour): {value:.2f}%")
    else:
        print("No GPU utilization data available for the last hour")


if __name__ == "__main__":
    main()
