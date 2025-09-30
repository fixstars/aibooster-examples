# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

import argparse

import modules  # noqa
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner  # noqa

import aibooster.intelligence.zenith_tune  # noqa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Overwrite config options.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    runner = RUNNERS.build(cfg)
    runner.train()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
