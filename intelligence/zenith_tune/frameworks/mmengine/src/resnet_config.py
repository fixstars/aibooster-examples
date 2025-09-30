# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

model = dict(type="ResNet")

norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])

train_dataloader = dict(
    batch_size=32,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="Cifar10",
        root="data/cifar10",
        train=True,
        download=True,
        transform=[
            dict(type="RandomCrop", size=32, padding=4),
            dict(type="RandomHorizontalFlip"),
            dict(type="ToTensor"),
            dict(type="Normalize", **norm_cfg),
        ],
    ),
    collate_fn=dict(type="default_collate"),
)

val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type="Cifar10",
        root="data/cifar10",
        train=False,
        download=True,
        transform=[
            dict(type="ToTensor"),
            dict(type="Normalize", **norm_cfg),
        ],
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
    collate_fn=dict(type="default_collate"),
)

optim_wrapper = dict(optimizer=dict(type="SGD", lr=0.001, momentum=0.9))

train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)
val_cfg = dict()
val_evaluator = dict(type="Accuracy")

env_cfg = dict(dist_cfg=dict(backend="nccl"))
launcher = "pytorch"
work_dir = "work_dir"
