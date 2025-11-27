# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

import tempfile

import torch
from torch import nn

from typing import cast
from numbers import Real
from dataclasses import asdict

from aibooster.intelligence.acuirt import ConversionWorkflow
from aibooster.intelligence.acuirt.dataclasses import AcuiRTONNXConversionConfig
from aibooster.intelligence.acuirt.utils.report_exporter import LoggingExporter
from aibooster.intelligence.acuirt.utils.logger import AcuiRTDefaultLogger


# Non Convertible Model
class NonConvertible(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.randn(5, 5))
        self.param2 = nn.Parameter(torch.randn(5, 5))

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            # raise 'NonConvertible' object has no attribute 'param3' if in onnx export.
            return self.param3
        else:
            return x + self.param + self.param2


# Dummy Class contains Non Convertible Model
# Requires fallback to convert model with TensorRT
class DummyClass(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
        self.norm = nn.BatchNorm2d(3)
        self.silu = nn.SiLU()
        self.non_convertible = NonConvertible()
        self.softmax = nn.Softmax(1)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.norm(x)
        x = self.silu(x)
        x = self.non_convertible(x)
        x = x.flatten(1, -1)

        return self.softmax(x)


# Dummy Evaluator. return accuracy: 1.0
class Evaluator:
    def __init__(self):
        self.count = 0
        self.matched = 0

    def update(self, result):
        self.count += 1
        self.matched += 1

    def aggregate(self):
        return {"accuracy": cast(Real, self.matched / self.count)}

    def reset(self):
        self.count = 0
        self.matched = 0


model = DummyClass()

config = AcuiRTONNXConversionConfig(rt_mode="onnx", auto=True, children=None)

path = tempfile.mkdtemp()

dataset = [(torch.randn(1, 3, 5, 5, device="cuda"), {})]


def post_process(batch):
    args, kwargs = batch
    return ((args,), kwargs)


# create conversion workflow and run conversion
workflow = ConversionWorkflow(
    model,
    Evaluator(),
    dataset,
    post_process,
    exporters=[LoggingExporter(AcuiRTDefaultLogger("logger"), "accuracy")],
)

# accelerated model and report are returned
# report containes performance, conversion summary and profile information
model_trt, report = workflow.run(
    asdict(config),
    path,
)
