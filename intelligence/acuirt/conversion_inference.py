import shutil
import tempfile

import torch
from torchvision.models import resnet50

from aibooster.intelligence.acuirt.convert.convert import convert_model
from aibooster.intelligence.acuirt.inference.inference import load_runtime_modules


def main():
    resnet = resnet50()
    resnet = resnet.cuda().eval()
    # enable auto conversion with int8, fp16 quantization.
    # implicit calibration is performed automatically inside of the convert_model function.
    config = {
        "rt_mode": "auto",
        "int8": True,
        "fp16": True,
    }
    # create temporary directory to save converted model.
    # if you need to get converted model, change directory.
    path = tempfile.mkdtemp()

    data = [{"x": torch.randn(1, 3, 224, 224)} for _ in range(10)]

    summary = convert_model(resnet, config, path, False, data)
    model = load_runtime_modules(resnet, summary, path)
    batch = data[0]["x"]  # TRTInferenceEngine only accept positional argument.
    model(batch.cuda())

    # remove temporary directory.
    shutil.rmtree(path)


if __name__ == "__main__":
    main()
