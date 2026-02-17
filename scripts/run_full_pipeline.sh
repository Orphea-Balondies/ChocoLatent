#!/bin/bash
set -euo pipefail

EXP_ID=${EXP_ID:-full-pipeline-$(date +%Y%m%d-%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/outputs}
MODEL_PATH=${MODEL_PATH:-model/stable-diffusion-v1-5}
CLIP_MODEL_PATH=${CLIP_MODEL_PATH:-model/clip-vit-base-patch32}
IMAGE_ROOT=${IMAGE_ROOT:-init_images}
IMAGE_DIRNAME=${IMAGE_DIRNAME:-lego-minifigure-faces}
PROMPT_FILE=${PROMPT_FILE:-}
SEED_FILE=${SEED_FILE:-}
SEEDS=${SEEDS:-}
METHODS=${METHODS:-chocolatent,glaze,photoguard,robust-ldm,mist}
BUDGET_L2_GRID=${BUDGET_L2_GRID:-8/255,12/255,24/255}
BUDGET_LPIPS_GRID=${BUDGET_LPIPS_GRID:-0.1,0.2,0.5}
CAPTION_TEMPLATE=${CAPTION_TEMPLATE:-}
PROTECT_ITERS=${PROTECT_ITERS:-300}
PROTECT_INITIAL_LR=${PROTECT_INITIAL_LR:-1.0}
PROTECT_EPS=${PROTECT_EPS:-0.1}
PROTECT_BATCH_SIZE=${PROTECT_BATCH_SIZE:-4}
PROTECT_NUM_WORKERS=${PROTECT_NUM_WORKERS:-4}
PROTECT_MIXED_PRECISION=${PROTECT_MIXED_PRECISION:-fp16}
MIST_TARGET_IMAGE_PATH=${MIST_TARGET_IMAGE_PATH:-/root/chocolatent/MIST.png}
GLAZE_STYLE_BACKEND=${GLAZE_STYLE_BACKEND:-onnx_mosaic}
GLAZE_STYLE_ONNX_PATH=${GLAZE_STYLE_ONNX_PATH:-model/style_transfer/mosaic-9.onnx}
GLAZE_STYLE_IMAGE_PATH=${GLAZE_STYLE_IMAGE_PATH:-}
GLAZE_STYLE_ALPHA=${GLAZE_STYLE_ALPHA:-1.0}
PROTECT_NAN_LR_DECAY=${PROTECT_NAN_LR_DECAY:-0.5}
PROTECT_NAN_MIN_LR=${PROTECT_NAN_MIN_LR:-1e-4}
PROTECT_NAN_MAX_RECOVERIES=${PROTECT_NAN_MAX_RECOVERIES:-8}
STREAM_CHILD_LOGS=${STREAM_CHILD_LOGS:-true}
HEARTBEAT_SEC=${HEARTBEAT_SEC:-120}
SKIP_PROTECT=${SKIP_PROTECT:-}
REWRITE_PROTECT=${REWRITE_PROTECT:-}
OVERWRITE_METADATA=${OVERWRITE_METADATA:-}
OVERWRITE_LORA=${OVERWRITE_LORA:-}
OVERWRITE_GENERATE=${OVERWRITE_GENERATE:-}
OVERWRITE_METRICS=${OVERWRITE_METRICS:-}
STOP_ON_ERROR=${STOP_ON_ERROR:-}
STRICT_PREFLIGHT=${STRICT_PREFLIGHT:-}
STRICT_METRICS=${STRICT_METRICS:-}
COMPUTE_FID=${COMPUTE_FID:-}

append_bool_flag() {
  local value="$1"
  local flag="$2"
  if [[ -z "${value}" ]]; then
    return
  fi
  case "${value}" in
    1|true|TRUE|True|yes|YES|on|ON)
      CMD+=(--"${flag}")
      ;;
    0|false|FALSE|False|no|NO|off|OFF)
      CMD+=(--no-"${flag}")
      ;;
    *)
      echo "Invalid boolean for ${flag}: ${value}"
      exit 1
      ;;
  esac
}

if [[ -z "${PROMPT_FILE}" ]]; then
  echo "PROMPT_FILE is required. Example: PROMPT_FILE=exp_plan/prompts_style.txt"
  exit 1
fi

if [[ "${METHODS}" == *"mist"* ]] && [[ -z "${MIST_TARGET_IMAGE_PATH}" ]]; then
  echo "MIST is enabled in METHODS but MIST_TARGET_IMAGE_PATH is empty."
  echo "Either set MIST_TARGET_IMAGE_PATH=/path/to/target.png or remove mist from METHODS."
  exit 1
fi

if [[ "${METHODS}" == *"glaze"* ]]; then
  if [[ "${GLAZE_STYLE_BACKEND}" == "adain" ]] && [[ -z "${GLAZE_STYLE_IMAGE_PATH}" ]]; then
    echo "GLAZE backend=adain requires GLAZE_STYLE_IMAGE_PATH=/path/to/style_ref.png."
    exit 1
  fi
  if [[ "${GLAZE_STYLE_BACKEND}" == "onnx_mosaic" ]] && [[ ! -f "${GLAZE_STYLE_ONNX_PATH}" ]]; then
    echo "GLAZE backend=onnx_mosaic requires ONNX file: ${GLAZE_STYLE_ONNX_PATH}"
    exit 1
  fi
fi

CMD=(
  python experiments/scripts/run_full_pipeline.py
  --exp_id "${EXP_ID}"
  --output_root "${OUTPUT_ROOT}"
  --model_path "${MODEL_PATH}"
  --clip_model_path "${CLIP_MODEL_PATH}"
  --image_root "${IMAGE_ROOT}"
  --image_dirname "${IMAGE_DIRNAME}"
  --methods "${METHODS}"
  --budget_l2_grid "${BUDGET_L2_GRID}"
  --budget_lpips_grid "${BUDGET_LPIPS_GRID}"
  --protect_iters "${PROTECT_ITERS}"
  --protect_initial_lr "${PROTECT_INITIAL_LR}"
  --protect_eps "${PROTECT_EPS}"
  --protect_batch_size "${PROTECT_BATCH_SIZE}"
  --protect_num_workers "${PROTECT_NUM_WORKERS}"
  --protect_mixed_precision "${PROTECT_MIXED_PRECISION}"
  --prompt_file "${PROMPT_FILE}"
)

if [[ -n "${SEED_FILE}" ]]; then
  CMD+=(--seed_file "${SEED_FILE}")
elif [[ -n "${SEEDS}" ]]; then
  CMD+=(--seeds "${SEEDS}")
fi

if [[ -n "${MIST_TARGET_IMAGE_PATH}" ]]; then
  CMD+=(--mist_target_image_path "${MIST_TARGET_IMAGE_PATH}")
fi

if [[ -n "${CAPTION_TEMPLATE}" ]]; then
  CMD+=(--caption_template "${CAPTION_TEMPLATE}")
fi

CMD+=(--glaze_style_backend "${GLAZE_STYLE_BACKEND}")
CMD+=(--glaze_style_onnx_path "${GLAZE_STYLE_ONNX_PATH}")
CMD+=(--glaze_style_image_path "${GLAZE_STYLE_IMAGE_PATH}")
CMD+=(--glaze_style_alpha "${GLAZE_STYLE_ALPHA}")
CMD+=(--protect_nan_lr_decay "${PROTECT_NAN_LR_DECAY}")
CMD+=(--protect_nan_min_lr "${PROTECT_NAN_MIN_LR}")
CMD+=(--protect_nan_max_recoveries "${PROTECT_NAN_MAX_RECOVERIES}")
CMD+=(--heartbeat_sec "${HEARTBEAT_SEC}")
append_bool_flag "${SKIP_PROTECT}" "skip_protect"
append_bool_flag "${REWRITE_PROTECT}" "rewrite_protect"
append_bool_flag "${OVERWRITE_METADATA}" "overwrite_metadata"
append_bool_flag "${OVERWRITE_LORA}" "overwrite_lora"
append_bool_flag "${OVERWRITE_GENERATE}" "overwrite_generate"
append_bool_flag "${OVERWRITE_METRICS}" "overwrite_metrics"
append_bool_flag "${STOP_ON_ERROR}" "stop_on_error"
append_bool_flag "${STRICT_PREFLIGHT}" "strict_preflight"
append_bool_flag "${STRICT_METRICS}" "strict_metrics"
append_bool_flag "${COMPUTE_FID}" "compute_fid"

if [[ "${STREAM_CHILD_LOGS}" == "0" || "${STREAM_CHILD_LOGS}" == "false" || "${STREAM_CHILD_LOGS}" == "False" ]]; then
  CMD+=(--no-stream_child_logs)
else
  CMD+=(--stream_child_logs)
fi

echo "[Run] ${CMD[*]}"
"${CMD[@]}"
