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
MASTER_PORT=${MASTER_PORT:-6000}


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

init_path="init_images/lego-minifigure-faces/"
model_path='stable-diffusion-v1-5'



image_list=$(python3 - <<END
import os
import json
init_path='$init_path'
model_path='$model_path'
image_list=[]
for _,_,files in os.walk(init_path):
    for f in files:
        if f.split('.')[-1].lower() not in ['jpg','png','jpeg']:
            continue
        file_path = f
        image_dict={"image_name": file_path}
        image_list.append(image_dict)
print(json.dumps(image_list))
END
)


#image_list='[{"image_name": "mix-27/16.png"}]'

out_dir="output/lego-minifigure-faces/"

torchrun $DISTRIBUTED_ARGS notebooks/distribution_adv_tgt.py\
    --image_dir $init_path\
    --image_list "$image_list"\
     --mode 'protect'\
    --out_dir $out_dir\
    --model_path $model_path\
    --maxDis 20 \
    --rewrite False\
    --nproc_per_gpu $NPROC_PER_GPU\
    --eps 0.08\
    --tgttype "latent"