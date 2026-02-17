#!/usr/bin/env bash
set -euo pipefail

cd /root/chocolatent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate choco

python experiments/scripts/run_full_pipeline.py \
  --exp_id wikiart-boris-full \
  --output_root experiments/outputs \
  --model_path model/stable-diffusion-v1-5 \
  --clip_model_path model/clip-vit-base-patch32 \
  --image_root datasets/partial/WikiArt_hot40_top15_artist \
  --image_dirname boris_kustodiev \
  --methods glaze,photoguard,robust-ldm,mist \
  --budget_l2_grid 4/255,8/255,12/255 \
  --budget_lpips_grid 0.1,0.2,0.5 \
  --mist_target_image_path /root/chocolatent/MIST.png \
  --glaze_style_backend onnx_mosaic \
  --glaze_style_onnx_path model/style_transfer/mosaic-9.onnx \
  --glaze_style_alpha 1.0 \
  --protect_iters 300 \
  --protect_initial_lr 0.5 \
  --protect_batch_size 2 \
  --protect_nan_lr_decay 0.7 \
  --protect_nan_min_lr 1e-5 \
  --protect_nan_max_recoveries 32 \
  --prompt_file datasets/partial/WikiArt_hot40_top15_artist/boris_kustodiev/prompts_boris_kustodiev.txt \
  --seed_file exp_plan/seeds_eval.txt \
  --caption_template "a painting in the style of {name}" \
  --heartbeat_sec 60 \
  --stream_child_logs \
  --skip_protect
