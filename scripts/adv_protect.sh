#!/bin/bash
set -euo pipefail

GPUS_PER_NODE=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-7000}

MODEL_PATH=${MODEL_PATH:-model/stable-diffusion-v1-5}
CLIP_MODEL_PATH=${CLIP_MODEL_PATH:-model/clip-vit-base-patch32}
IMAGE_ROOT=${IMAGE_ROOT:-init_images}
IMAGE_DIRNAME=${IMAGE_DIRNAME:-lego-minifigure-faces}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/outputs}
EXP_ID=${EXP_ID:-}
METHOD=${METHOD:-chocolatent}
STAGE=${STAGE:-A}
DATASET_TYPE=${DATASET_TYPE:-D_UNKNOWN}
TASK_TYPE=${TASK_TYPE:-style_imitation}
FINETUNE_METHOD=${FINETUNE_METHOD:-LoRA}
SRC_MODEL=${SRC_MODEL:-SD1.5}
TGT_MODEL=${TGT_MODEL:-SD1.5}
BUDGET_L2_GRID=${BUDGET_L2_GRID:-4/255,8/255,12/255}
BUDGET_LPIPS_GRID=${BUDGET_LPIPS_GRID:-0.1,0.2,0.5}
ITERS=${ITERS:-600}
INITIAL_LR=${INITIAL_LR:-4.0}
EPS=${EPS:-0.1}
TARGET_IMAGE_PATH=${TARGET_IMAGE_PATH:-/root/chocolatent/MIST.png}
GLAZE_STYLE_BACKEND=${GLAZE_STYLE_BACKEND:-onnx_mosaic}
GLAZE_STYLE_ONNX_PATH=${GLAZE_STYLE_ONNX_PATH:-model/style_transfer/mosaic-9.onnx}
GLAZE_STYLE_IMAGE_PATH=${GLAZE_STYLE_IMAGE_PATH:-}
GLAZE_STYLE_ALPHA=${GLAZE_STYLE_ALPHA:-1.0}
SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
LOG_EVERY=${LOG_EVERY:-50}

REWRITE=${REWRITE:-true}
SHOW_PROGRESS=${SHOW_PROGRESS:-true}
STRICT_LPIPS_PROJECTION=${STRICT_LPIPS_PROJECTION:-true}
COLLECT_METRICS=${COLLECT_METRICS:-false}

if [[ "${REWRITE}" == "true" ]]; then
  REWRITE_FLAG="--rewrite"
else
  REWRITE_FLAG="--no-rewrite"
fi

if [[ "${SHOW_PROGRESS}" == "true" ]]; then
  SHOW_PROGRESS_FLAG="--show_progress"
else
  SHOW_PROGRESS_FLAG="--no-show_progress"
fi

if [[ "${STRICT_LPIPS_PROJECTION}" == "true" ]]; then
  STRICT_LPIPS_FLAG="--strict_lpips_projection"
else
  STRICT_LPIPS_FLAG="--no-strict_lpips_projection"
fi

EXTRA_FLAGS=()
if [[ "${COLLECT_METRICS}" == "true" ]]; then
  EXTRA_FLAGS+=("--collect_metrics")
fi

if [[ -n "${EXP_ID}" ]]; then
  EXTRA_FLAGS+=("--exp_id" "${EXP_ID}")
fi

if [[ "${METHOD}" == "mist" ]]; then
  if [[ -z "${TARGET_IMAGE_PATH}" ]]; then
    echo "METHOD=mist requires TARGET_IMAGE_PATH=/path/to/target.png"
    exit 1
  fi
  EXTRA_FLAGS+=("--target_image_path" "${TARGET_IMAGE_PATH}")
fi

if [[ "${METHOD}" == "glaze" ]]; then
  if [[ "${GLAZE_STYLE_BACKEND}" == "adain" ]] && [[ -z "${GLAZE_STYLE_IMAGE_PATH}" ]]; then
    echo "METHOD=glaze backend=adain requires GLAZE_STYLE_IMAGE_PATH=/path/to/style_ref.png"
    exit 1
  fi
  if [[ "${GLAZE_STYLE_BACKEND}" == "onnx_mosaic" ]] && [[ ! -f "${GLAZE_STYLE_ONNX_PATH}" ]]; then
    echo "METHOD=glaze backend=onnx_mosaic requires ONNX file: ${GLAZE_STYLE_ONNX_PATH}"
    exit 1
  fi
  EXTRA_FLAGS+=("--glaze_style_backend" "${GLAZE_STYLE_BACKEND}")
  EXTRA_FLAGS+=("--glaze_style_onnx_path" "${GLAZE_STYLE_ONNX_PATH}")
  EXTRA_FLAGS+=("--glaze_style_image_path" "${GLAZE_STYLE_IMAGE_PATH}")
  EXTRA_FLAGS+=("--glaze_style_alpha" "${GLAZE_STYLE_ALPHA}")
fi

accelerate launch \
    --num_processes "$GPUS_PER_NODE" \
    --num_machines "$NNODES" \
    --machine_rank "$NODE_RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    code/distribution_adv_tgt.py \
    --model_path "$MODEL_PATH" \
    --clip_model_path "$CLIP_MODEL_PATH" \
    --image_root "$IMAGE_ROOT" \
    --image_dirname "$IMAGE_DIRNAME" \
    --output_root "$OUTPUT_ROOT" \
    --method "$METHOD" \
    --stage "$STAGE" \
    --dataset_type "$DATASET_TYPE" \
    --task_type "$TASK_TYPE" \
    --finetune_method "$FINETUNE_METHOD" \
    --src_model "$SRC_MODEL" \
    --tgt_model "$TGT_MODEL" \
    --budget_l2_grid "$BUDGET_L2_GRID" \
    --budget_lpips_grid "$BUDGET_LPIPS_GRID" \
    --iters "$ITERS" \
    --initial_lr "$INITIAL_LR" \
    --eps "$EPS" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --pin_memory \
    --persistent_workers \
    --mixed_precision fp16 \
    --log_every "$LOG_EVERY" \
    "$REWRITE_FLAG" \
    "$SHOW_PROGRESS_FLAG" \
    "$STRICT_LPIPS_FLAG" \
    "${EXTRA_FLAGS[@]}"
