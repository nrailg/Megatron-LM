#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NNODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2
	--pipeline-model-parallel-size 2
)

CHECKPOINT_PATH=${PWD}/ckpt
EVAL_AND_LOGGING_ARGS=(
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
)

readonly WORK_DIR="examples/cli-arg-yaml-cfgs"

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
  --cli-arg-yaml-cfgs $WORK_DIR/trainer.yaml $WORK_DIR/data.yaml $WORK_DIR/gpt.yaml \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${EVAL_AND_LOGGING_ARGS[@]}
