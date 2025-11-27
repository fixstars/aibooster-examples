# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

import glob
import os
from typing import List, Tuple, cast
from numbers import Real

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

from aibooster.intelligence.acuirt import ConversionWorkflow


def get_imagenet_classes():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        class_list = torch.hub.load_state_dict_from_url(url)
    except Exception:
        import urllib.request

        with urllib.request.urlopen(url) as f:
            class_list = [
                line.decode("utf-8").strip().split(" ")[0] for line in f.readlines()
            ]
    return class_list


def get_image_paths(root: str):
    jpeg_files = glob.glob(os.path.join(root, "**", "*.JPEG"), recursive=True)

    return jpeg_files


class ImageNetDataset:
    def __init__(self, image_paths: List[str]):
        assert len(image_paths) > 0, "No images found in the dataset"
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image = preprocess(image)
        return (
            image.unsqueeze(0),
            os.path.splitext(os.path.basename(image_path).split("_")[1])[0],
        )  # add batch dim

    @staticmethod
    def post_process(inputs: Tuple[torch.Tensor, str]):
        return ((inputs[0].cuda(),), {})


# Evaluator for ImageNet classification
# Calculates Top-1 accuracy
class ImageNetEvaluator:
    def __init__(self, dataset):
        self.correct = 0
        self.total = 0
        self.labels = [label for _, label in dataset]
        self.image_classes = get_imagenet_classes()

    def update(self, result: torch.Tensor):
        predicted_class = self.image_classes[result.argmax(dim=-1).cpu().item()]

        if predicted_class == self.labels[self.total]:
            self.correct += 1
        self.total += 1

    def aggregate(self):
        accuracy = self.correct / self.total if self.total > 0 else 0.0
        return {"accuracy": cast(Real, accuracy)}

    def reset(self):
        self.correct = 0
        self.total = 0


def main():
    resnet = resnet50(pretrained=True)
    resnet = resnet.cuda().eval()

    # int8 quantization (PTQ) settings for conversion to TensorRT
    config = {
        "rt_mode": "onnx",
        "auto": True,
        "int8": True,
    }

    # export path to save the converted model
    path = "./model"

    image_paths = get_image_paths("./imagenet-sample-images")

    # create a dataset with ImageNet sample images
    dataset = ImageNetDataset(image_paths)

    # set profiler settings
    profiler_settings = {
        "activities": [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        "profile_memory": False,
        "schedule": torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    }

    # create conversion workflow
    workflow = ConversionWorkflow(
        resnet,
        ImageNetEvaluator(dataset),
        dataset,
        dataset.post_process,
        settings_torch_profiler=profiler_settings,
        eval_non_converted_model=True,
    )

    # run conversion workflow and get the converted model and report
    resnet_trt, report = workflow.run(config, path)

    assert report.non_converted_performance is not None

    print(
        f"PyTorch: Top-1 Accuracy: {report.non_converted_performance.accuracy['accuracy'] * 100:.2f}%, Average Inference Time: {report.non_converted_performance.latency}"
    )
    print(
        f"AcuiRT: Top-1 Accuracy: {report.performance.accuracy['accuracy'] * 100:.2f}%, Average Inference Time: {report.performance.latency}"
    )


if __name__ == "__main__":
    main()
