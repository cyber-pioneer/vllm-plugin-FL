#!/bin/bash
#export USE_FLAGGEMS=0
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
#export ASCEND_VISIBLE_DEVICES=4,5,6,7
export TRITON_ALL_BLOCKS_PARALLEL=1
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

cd /workspace/vllm-plugin-FL
vllm serve /models/Qwen3.5-35B-A3B \
    --served-model-name "qwen3.5" \
    --host 0.0.0.0 \
    --port 8020 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 32768 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 8 \
    --block-size 384 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --trust-remote-code \
    --allowed-local-media-path / \
    --mm-processor-cache-gb 0 \
    --additional-config '{"enable_cpu_binding":true}'
