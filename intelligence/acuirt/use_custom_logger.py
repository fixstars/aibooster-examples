# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

import logging
import shutil
import tempfile
from datetime import datetime

import torch
from aibooster.intelligence.acuirt.convert.convert import convert_model
from aibooster.intelligence.acuirt.inference.inference import load_runtime_modules
from aibooster.intelligence.acuirt import AcuiRTDefaultLogger
from tensorrt import ILogger
from torchvision.models import resnet50


def main():
    local_dt = datetime.now().astimezone()
    local_utc = local_dt.strftime("%z")

    logger = AcuiRTDefaultLogger(name="AcuiRT", min_severity=ILogger.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s]:%(name)s:%(levelname)s:%(message)s",
        datefmt=f"User Defined:%Y/%m/%d-%H:%M:%S,UTC{local_utc}",
    )
    logger.set_formatter(formatter)

    resnet = resnet50()
    resnet = resnet.cuda().eval()

    config = {
        "rt_mode": "onnx",
        "auto": True,
        "int8": True,
        "fp16": True,
    }
    path = tempfile.mkdtemp()

    data = [{"x": torch.randn(1, 3, 224, 224)} for _ in range(10)]

    # If applying a custom logger, pass the logger as an argument
    summary = convert_model(resnet, config, path, False, data, logger=logger)
    model = load_runtime_modules(resnet, summary, path, logger=logger)
    batch = data[0]["x"]
    model(batch.cuda())

    shutil.rmtree(path)


if __name__ == "__main__":
    main()
