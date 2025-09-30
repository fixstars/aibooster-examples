# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

import torch.nn.functional as F
import torchvision
import torchvision.transforms as tvt
from mmengine.dataset.base_dataset import Compose
from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.registry import DATASETS, METRICS, MODELS, TRANSFORMS


@MODELS.register_module(name="ResNet")
class MMResNet50(BaseModel):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == "loss":
            return {"loss": F.cross_entropy(x, labels)}
        elif mode == "predict":
            return x, labels


@METRICS.register_module(name="Accuracy")
class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append(
            {
                "batch_size": len(gt),
                "correct": (score.argmax(dim=1) == gt).sum().cpu(),
            }
        )

    def compute_metrics(self, results):
        total_correct = sum(item["correct"] for item in results)
        total_size = sum(item["batch_size"] for item in results)
        return dict(accuracy=100 * total_correct / total_size)


@DATASETS.register_module(name="Cifar10")
def build_torchvision_cifar10(transform=None, **kwargs):
    if isinstance(transform, dict):
        transform = [transform]
    if isinstance(transform, (list, tuple)):
        transform = Compose(transform)
    return torchvision.datasets.CIFAR10(**kwargs, transform=transform)


TRANSFORMS.register_module("RandomCrop", module=tvt.RandomCrop)
TRANSFORMS.register_module("RandomHorizontalFlip", module=tvt.RandomHorizontalFlip)
TRANSFORMS.register_module("ToTensor", module=tvt.ToTensor)
TRANSFORMS.register_module("Normalize", module=tvt.Normalize)
