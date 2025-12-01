#!/bin/bash

NPROC_PER_GPU=2

GPUS_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count()* $NPROC_PER_GPU )")

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-9000}


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS feature_fake_image.py --nproc_per_gpu $NPROC_PER_GPU --iters 15000 --step 4 --Lpipsvalue 500 1000 \
 --image_dir "/home/zhangjianwei/photoguard/init_images/mix-27" --output_dir "/mnt/algorithm/user_dir/zhangjianwei/photoguard/output"
