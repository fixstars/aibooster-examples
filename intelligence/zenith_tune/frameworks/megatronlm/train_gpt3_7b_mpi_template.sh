#!/bin/bash
# Copyright (c) 2025 Fixstars Corporation
# SPDX-License-Identifier: MIT

# Runs the "7B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
# Change for multinode config
export RANK=$OMPI_COMM_WORLD_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=14535 # 適当な値
export LOCAL_RANK=$((OMPI_COMM_WORLD_RANK % GPUS_PER_NODE))
export OMP_NUM_THREADS=28

TENSORBOARD_LOGS_PATH=$1 #<Specify path>
VOCAB_FILE=$2 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$3 #<Specify path to file>/gpt2-merges.txt
DATA_PATH=$4 #<Specify path and file prefix>_text_document

GPT_MODEL_ARGS=(
    --num-layers 30
    --hidden-size 4096
    --num-attention-heads 32
    --seq-length {{sequence_length}}
    --max-position-embeddings {{sequence_length}}
)

TRAINING_ARGS=(
    --micro-batch-size {{micro_batch_size}}
    --global-batch-size {{global_batch_size}}
    --train-iters 100
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --fp16
    --lr 6.0e-5
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction .001
    --lr-decay-iters 430000
    # fp8
    --transformer-impl transformer_engine
    --fp8-format hybrid
    --fp8-interval 1024
    --fp8-amax-history-len 1
    --fp8-amax-compute-algo max
    # compute method
    --swiglu
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --disable-bias-linear
    --no-masked-softmax-fusion
    --untie-embeddings-and-output-weights
    # distribution algorithm
    --use-distributed-optimizer
    --overlap-grad-reduce
    --sequence-parallel
    # legacy settings
    --use-legacy-models
    --attention-softmax-in-fp32
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size {{tensor_model_parallel_size}}
    --pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 0
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --log-throughput
    --log-memory-to-tensorboard
)

python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
