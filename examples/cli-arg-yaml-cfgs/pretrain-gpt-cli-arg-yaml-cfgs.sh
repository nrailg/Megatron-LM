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

CHECKPOINT_PATH=${PWD}/ckpt
TENSORBOARD_LOGS_PATH=${PWD}/tb
VOCAB_FILE=../oscar-data/gpt2-vocab.json
MERGE_FILE=../oscar-data/gpt2-merges.txt
DATA_PATH=../oscar-data/my-gpt2_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NNODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

readonly MODEL_YAML="examples/cli-arg-yaml-cfgs/gpt.yaml"

readonly TRAINING_YAML="examples/cli-arg-yaml-cfgs/trainer.yaml"

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2
	--pipeline-model-parallel-size 2
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

export WANDB_BASE_URL=https://wandb.lubanml.woa.com
export WANDB_API_KEY=local-f6267444e79de80aa84dedff1d214b8e090f389b

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --wandb-project nrwu
    --wandb-exp-name no-name
    --wandb-save-dir ${PWD}/wandb
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
  --cli-arg-yaml-cfgs $TRAINING_YAML $MODEL_YAML \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${EVAL_AND_LOGGING_ARGS[@]}
