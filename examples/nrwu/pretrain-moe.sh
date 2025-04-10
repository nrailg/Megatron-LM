#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=${PWD}/ckpt
TENSORBOARD_LOGS_PATH=${PWD}/tb
VOCAB_FILE=../oscar-data/gpt2-vocab.json
MERGE_FILE=../oscar-data/gpt2-merges.txt
DATA_PATH=../oscar-data/my-gpt2_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

MP_PP0_LAYERS=3
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
NUM_LAYERS=8
INTERMEDIATE_SIZE=11264
NUM_SHARED_EXPERTS=2
MOE_INTERMEDIATE_SIZE=1408
SEQ_LEN=2048
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=0
Q_LORA_RANK=None
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=50000
NUM_EXPERTS=64
ROUTER_TOPK=6
FIRST_K_DENSE_REPLACE=1
RMS_NORM_EPS=1e-5
TOPK_SCALING_FACTOR=2.446

GPT_MODEL_ARGS=(
    --attention-backend flash # Can use (flash/fused/unfused/local)
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --num-attention-heads ${NUM_ATTN_HEADS}
    --ffn-hidden-size ${INTERMEDIATE_SIZE}
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS}
    --swiglu
    --normalization RMSNorm
    --norm-epsilon ${RMS_NORM_EPS}
    --use-rotary-position-embeddings
    --no-bias-swiglu-fusion
    --no-rope-fusion
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --rotary-base ${ROPE_THETA}
    --kv-channels ${V_HEAD_DIM}
    --qk-layernorm
    # moe args
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE}
    --moe-router-topk ${ROUTER_TOPK}
    --num-experts ${NUM_EXPERTS}
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} ))
    --moe-router-num-groups 1
    --moe-router-group-topk 1
    --moe-router-enable-expert-bias
    --moe-router-score-function sigmoid
    --moe-router-topk-scaling-factor ${TOPK_SCALING_FACTOR}
    --moe-router-load-balancing-type seq_aux_loss
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 32 
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --recompute-method uniform
    --recompute-num-layers 1
    --recompute-granularity full
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
  --expert-model-parallel-size 4 \
  --context-parallel-size 2 \
  --sequence-parallel \
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
    --wandb-exp-name moonlight
    --wandb-save-dir ${PWD}/wandb
)

PYTHONPATH="../gcore:$PYTHONPATH" \
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
