# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

import glob
import os
import time
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

from aibooster.intelligence.acuirt.convert import convert_model
from aibooster.intelligence.acuirt.inference import load_runtime_modules


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

    # 両方のリストを結合
    all_jpeg_files = jpeg_files

    return all_jpeg_files


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
        )  # バッチ次元を追加

    @staticmethod
    def post_process(inputs: Tuple[torch.Tensor, str]):
        return ((inputs[0],), {})


def main():
    resnet = resnet50(pretrained=True)
    resnet = resnet.cuda().eval()

    # int8量子化(PTQ)でTensorRTに変換する場合の設定
    config = {
        "rt_mode": "onnx",
        "auto": True,
        "int8": True,
    }

    # 変換後のモデルを保存するパスを指定してください。
    path = "./model"

    image_paths = get_image_paths("./imagenet-sample-images")
    dataset = ImageNetDataset(image_paths)
    image_classes = get_imagenet_classes()

    # ダミーのデータセットを作成します
    # Iterableなデータセットを渡すことで、calibrationが自動的に行われます。

    # TensorRTへの変換と、calibrationを実行します。
    # 変換後のモデルはpathに保存されます。
    # また、summaryというdict型の変数に変換されたモデルの情報が格納されます。
    summary = convert_model(resnet, config, path, False, dataset, dataset.post_process)
    resnet = resnet.cuda()

    # TensorRT用の推論エンジンをロードします。
    resnet_trt = load_runtime_modules(resnet, summary, path)

    # 推論を実行します。
    correct, correct_trt, total = 0, 0, len(dataset)
    process_time, process_time_trt = [], []
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        args = [image.cuda()]

        start = time.perf_counter_ns()
        outputs_trt = resnet_trt(*args).argmax(dim=-1)
        end = time.perf_counter_ns()
        inference_time_trt = (end - start) / 1_000
        process_time_trt.append(inference_time_trt)
        predicted_class_trt = image_classes[outputs_trt.cpu().item()]

        if predicted_class_trt == label:
            correct_trt += 1

        start = time.perf_counter_ns()
        with torch.no_grad():
            outputs = resnet(*args).argmax(dim=-1)
        end = time.perf_counter_ns()

        inference_time = (end - start) / 1_000  # μ秒に変換
        process_time.append(inference_time)
        predicted_class = image_classes[outputs.cpu().item()]

        if predicted_class == label:
            correct += 1

    print(
        f"PyTorch: Top-1 Accuracy: {correct}/{total} ({100.0 * correct / total:.2f}%), Average Inference Time: {sum(process_time) / total:.2f}μs"
    )
    print(
        f"AcuiRT: Top-1 Accuracy: {correct_trt}/{total} ({100.0 * correct_trt / total:.2f}%), Average Inference Time: {sum(process_time_trt) / total:.2f}μs"
    )


if __name__ == "__main__":
    main()
