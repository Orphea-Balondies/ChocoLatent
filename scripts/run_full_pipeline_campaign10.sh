#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CAMPAIGN_ID=${CAMPAIGN_ID:-campaign10-$(date +%Y%m%d-%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/outputs}
PIPELINE_METHODS=${PIPELINE_METHODS:-chocolatent,glaze,photoguard,robust-ldm,mist}
PIPELINE_BUDGET_L2_GRID=${PIPELINE_BUDGET_L2_GRID:-8/255,12/255,24/255}
PIPELINE_BUDGET_LPIPS_GRID=${PIPELINE_BUDGET_LPIPS_GRID:-0.1,0.2,0.5}
BORIS_EXP_ID=${BORIS_EXP_ID:-wikiart-boris-full}

PROTECT_ITERS=${PROTECT_ITERS:-300}
PROTECT_INITIAL_LR=${PROTECT_INITIAL_LR:-0.5}
PROTECT_BATCH_SIZE=${PROTECT_BATCH_SIZE:-2}
PROTECT_NAN_LR_DECAY=${PROTECT_NAN_LR_DECAY:-0.7}
PROTECT_NAN_MIN_LR=${PROTECT_NAN_MIN_LR:-1e-5}
PROTECT_NAN_MAX_RECOVERIES=${PROTECT_NAN_MAX_RECOVERIES:-32}
HEARTBEAT_SEC=${HEARTBEAT_SEC:-60}

REWRITE_PROTECT=${REWRITE_PROTECT:-false}
SKIP_PROTECT=${SKIP_PROTECT:-false}
OVERWRITE_METADATA=${OVERWRITE_METADATA:-false}
OVERWRITE_LORA=${OVERWRITE_LORA:-false}
OVERWRITE_GENERATE=${OVERWRITE_GENERATE:-false}
OVERWRITE_METRICS=${OVERWRITE_METRICS:-false}
STOP_ON_ERROR=${STOP_ON_ERROR:-false}
STRICT_PREFLIGHT=${STRICT_PREFLIGHT:-false}
STRICT_METRICS=${STRICT_METRICS:-false}
COMPUTE_FID=${COMPUTE_FID:-false}
STREAM_CHILD_LOGS=${STREAM_CHILD_LOGS:-true}

MIST_TARGET_IMAGE_PATH=${MIST_TARGET_IMAGE_PATH:-/root/chocolatent/MIST.png}
GLAZE_STYLE_BACKEND=${GLAZE_STYLE_BACKEND:-onnx_mosaic}
GLAZE_STYLE_ONNX_PATH=${GLAZE_STYLE_ONNX_PATH:-model/style_transfer/mosaic-9.onnx}
GLAZE_STYLE_ALPHA=${GLAZE_STYLE_ALPHA:-1.0}

PROMPT_ROOT=${PROMPT_ROOT:-exp_plan/prompt_sets}
SEED_ROOT=${SEED_ROOT:-exp_plan/seed_sets}
mkdir -p "${PROMPT_ROOT}/concept" "${PROMPT_ROOT}/wikiart" "${SEED_ROOT}"

PLAN_FILE="${OUTPUT_ROOT}/${CAMPAIGN_ID}/campaign_plan.csv"
mkdir -p "$(dirname "${PLAN_FILE}")"
cat > "${PLAN_FILE}" <<'CSV'
dataset_kind,image_root,image_dirname,exp_id,caption_template,prompt_file,seed_file
CSV

declare -a EXP_IDS=()
declare -a FAILED_GROUPS=()

sanitize_name() {
  local raw="$1"
  echo "${raw//_/ }"
}

ensure_seed_file() {
  local seed_file="$1"
  local offset="$2"
  if [[ -f "${seed_file}" ]]; then
    return
  fi
  mkdir -p "$(dirname "${seed_file}")"
  local base=(42 3407 2025 314159 8675309)
  : > "${seed_file}"
  for seed in "${base[@]}"; do
    echo "$((seed + offset))" >> "${seed_file}"
  done
}

ensure_concept_prompt_file() {
  local group="$1"
  local prompt_file="$2"
  if [[ -f "${prompt_file}" ]]; then
    return
  fi
  mkdir -p "$(dirname "${prompt_file}")"
  local readable
  readable="$(sanitize_name "${group}")"
  cat > "${prompt_file}" <<EOF
a studio photo of ${readable}, highly detailed, soft lighting
a close-up photo of ${readable}, shallow depth of field
a ${readable} on a wooden table, natural light
a ${readable} in an outdoor scene, cinematic composition
a clean catalog photo of ${readable} on white background
a ${readable} in a modern indoor environment, high detail
EOF
}

ensure_wikiart_prompt_file() {
  local group="$1"
  local prompt_file="$2"
  if [[ -f "${prompt_file}" ]]; then
    return
  fi
  mkdir -p "$(dirname "${prompt_file}")"
  local readable
  readable="$(sanitize_name "${group}")"
  cat > "${prompt_file}" <<EOF
a painting in the style of ${readable}, bustling city street, oil on canvas
a painting in the style of ${readable}, portrait with warm rim lighting
a painting in the style of ${readable}, dramatic seascape and clouds
a painting in the style of ${readable}, still life with flowers and cloth
a painting in the style of ${readable}, village scene at sunset
a painting in the style of ${readable}, festival crowd with vivid palette
EOF
}

run_one_group() {
  local dataset_kind="$1"
  local image_root="$2"
  local image_dirname="$3"
  local exp_id="$4"
  local caption_template="$5"
  local prompt_file="$6"
  local seed_file="$7"

  echo "[Group] kind=${dataset_kind} name=${image_dirname} exp_id=${exp_id}"
  echo "${dataset_kind},${image_root},${image_dirname},${exp_id},${caption_template},${prompt_file},${seed_file}" >> "${PLAN_FILE}"
  EXP_IDS+=("${exp_id}")

  if ! EXP_ID="${exp_id}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    IMAGE_ROOT="${image_root}" \
    IMAGE_DIRNAME="${image_dirname}" \
    PROMPT_FILE="${prompt_file}" \
    SEED_FILE="${seed_file}" \
    CAPTION_TEMPLATE="${caption_template}" \
    METHODS="${PIPELINE_METHODS}" \
    BUDGET_L2_GRID="${PIPELINE_BUDGET_L2_GRID}" \
    BUDGET_LPIPS_GRID="${PIPELINE_BUDGET_LPIPS_GRID}" \
    PROTECT_ITERS="${PROTECT_ITERS}" \
    PROTECT_INITIAL_LR="${PROTECT_INITIAL_LR}" \
    PROTECT_BATCH_SIZE="${PROTECT_BATCH_SIZE}" \
    PROTECT_NAN_LR_DECAY="${PROTECT_NAN_LR_DECAY}" \
    PROTECT_NAN_MIN_LR="${PROTECT_NAN_MIN_LR}" \
    PROTECT_NAN_MAX_RECOVERIES="${PROTECT_NAN_MAX_RECOVERIES}" \
    HEARTBEAT_SEC="${HEARTBEAT_SEC}" \
    REWRITE_PROTECT="${REWRITE_PROTECT}" \
    SKIP_PROTECT="${SKIP_PROTECT}" \
    OVERWRITE_METADATA="${OVERWRITE_METADATA}" \
    OVERWRITE_LORA="${OVERWRITE_LORA}" \
    OVERWRITE_GENERATE="${OVERWRITE_GENERATE}" \
    OVERWRITE_METRICS="${OVERWRITE_METRICS}" \
    STOP_ON_ERROR="${STOP_ON_ERROR}" \
    STRICT_PREFLIGHT="${STRICT_PREFLIGHT}" \
    STRICT_METRICS="${STRICT_METRICS}" \
    COMPUTE_FID="${COMPUTE_FID}" \
    STREAM_CHILD_LOGS="${STREAM_CHILD_LOGS}" \
    MIST_TARGET_IMAGE_PATH="${MIST_TARGET_IMAGE_PATH}" \
    GLAZE_STYLE_BACKEND="${GLAZE_STYLE_BACKEND}" \
    GLAZE_STYLE_ONNX_PATH="${GLAZE_STYLE_ONNX_PATH}" \
    GLAZE_STYLE_ALPHA="${GLAZE_STYLE_ALPHA}" \
    bash scripts/run_full_pipeline.sh; then
    FAILED_GROUPS+=("${dataset_kind}:${image_dirname}")
  fi
}

declare -a CONCEPT_GROUPS=(
  car
  cat
  dog
  purse
  person
)

declare -a WIKIART_GROUPS=(
  boris_kustodiev
  claude_monet
  camille_pissarro
  vincent_van_gogh
  john_singer_sargent
)

seed_offset=0
for group in "${CONCEPT_GROUPS[@]}"; do
  seed_offset=$((seed_offset + 1000))
  prompt_file="${PROMPT_ROOT}/concept/prompts_${group}.txt"
  seed_file="${SEED_ROOT}/seeds_concept_${group}.txt"
  ensure_concept_prompt_file "${group}" "${prompt_file}"
  ensure_seed_file "${seed_file}" "${seed_offset}"
  run_one_group \
    "concept" \
    "datasets/partial/Concept" \
    "${group}" \
    "${CAMPAIGN_ID}-concept-${group}" \
    "a photo of {name}" \
    "${prompt_file}" \
    "${seed_file}"
done

for group in "${WIKIART_GROUPS[@]}"; do
  seed_offset=$((seed_offset + 1000))
  seed_file="${SEED_ROOT}/seeds_wikiart_${group}.txt"
  ensure_seed_file "${seed_file}" "${seed_offset}"

  if [[ "${group}" == "boris_kustodiev" && -f "datasets/partial/WikiArt_hot40_top15_artist/boris_kustodiev/prompts_boris_kustodiev.txt" ]]; then
    prompt_file="datasets/partial/WikiArt_hot40_top15_artist/boris_kustodiev/prompts_boris_kustodiev.txt"
    exp_id="${BORIS_EXP_ID}"
  else
    prompt_file="${PROMPT_ROOT}/wikiart/prompts_${group}.txt"
    ensure_wikiart_prompt_file "${group}" "${prompt_file}"
    exp_id="${CAMPAIGN_ID}-wikiart-${group}"
  fi

  run_one_group \
    "wikiart" \
    "datasets/partial/WikiArt_hot40_top15_artist" \
    "${group}" \
    "${exp_id}" \
    "a painting in the style of {name}" \
    "${prompt_file}" \
    "${seed_file}"
done

EXP_IDS_CSV="$(IFS=,; echo "${EXP_IDS[*]}")"
ANALYSIS_DIR="${OUTPUT_ROOT}/${CAMPAIGN_ID}/analysis_10groups"
mkdir -p "${ANALYSIS_DIR}"

if ! python experiments/scripts/run_campaign_analysis.py \
  --exp_ids "${EXP_IDS_CSV}" \
  --output_root "${OUTPUT_ROOT}" \
  --output_dir "${ANALYSIS_DIR}" \
  --methods "${PIPELINE_METHODS}" \
  --budget_l2_grid "${PIPELINE_BUDGET_L2_GRID}" \
  --budget_lpips_grid "${PIPELINE_BUDGET_LPIPS_GRID}"; then
  echo "[Warning] campaign analysis failed. Please inspect logs and rerun analysis manually."
fi

if ((${#FAILED_GROUPS[@]} > 0)); then
  echo "[Campaign] finished with failed groups:"
  for item in "${FAILED_GROUPS[@]}"; do
    echo "  - ${item}"
  done
  exit 1
fi

echo "[Campaign] all groups finished."
echo "[Campaign] plan file: ${PLAN_FILE}"
echo "[Campaign] analysis dir: ${ANALYSIS_DIR}"
