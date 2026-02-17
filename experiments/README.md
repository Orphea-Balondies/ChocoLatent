# Protocol-Oriented Experiment Entrypoints

This folder follows `exp_plan/experiment_protocol_latent_protection.md`.

## Scripts

- `experiments/scripts/run_protect.py`
  - Generate protected images with `B2 x Bp` sweep.
  - Writes `protected_images/`, `metrics/<run_id>/{metrics.json,per_image.csv,config.yaml}`, `manifest.csv`, and logs.
- `experiments/scripts/run_lora_train.py`
  - Wrapper for `code/train_text_to_image_lora.py`.
- `experiments/scripts/run_generate_eval.py`
  - Wrapper for `code/text2image_generate.py`.
- `experiments/scripts/run_metrics.py`
  - Computes paired `PGG_lpips` and `PGG_clip` between `generated_clean` and `generated_adv`.
- `experiments/scripts/run_robustness.py`
  - Generates transformed datasets for robustness tests (JPEG/resize/crop/noise).
- `experiments/scripts/run_analysis.py`
  - Aggregates all `metrics/*/metrics.json` into one CSV.
- `experiments/scripts/run_conclusion_metrics.py`
  - Computes protocol-aligned summary metrics for one `(clean, adv)` generated pair set:
  - `PGG_lpips`, `PGG_clip`, `MLS_clean`, `MLS_adv`, `PG`, `CLIP-T`, `QRR` (+ optional FID).
- `experiments/scripts/run_full_pipeline.py`
  - End-to-end one-command pipeline:
  - `protect (all methods x all budgets) -> train clean/adv LoRA -> generate clean/adv by multi prompts+seeds -> conclusion metrics -> summary`.
- `experiments/scripts/run_campaign_analysis.py`
  - Cross-experiment aggregation for multiple `exp_id`s.
  - Filters by target method and budget grid, then exports merged CSV/JSON and heatmap plots.

## Example

```bash
python experiments/scripts/run_protect.py \
  --model_path model/stable-diffusion-v1-5 \
  --image_root init_images \
  --image_dirname lego-minifigure-faces \
  --method chocolatent \
  --budget_l2_grid 4/255,8/255,12/255 \
  --budget_lpips_grid 0.1,0.2,0.5 \
  --output_root experiments/outputs \
  --exp_id stageA-demo
```

## Full Pipeline Example

```bash
bash scripts/setup_experiment_env.sh

python experiments/scripts/run_full_pipeline.py \
  --exp_id full-demo \
  --output_root experiments/outputs \
  --model_path model/stable-diffusion-v1-5 \
  --clip_model_path model/clip-vit-base-patch32 \
  --image_root init_images \
  --image_dirname lego-minifigure-faces \
  --methods chocolatent,glaze,photoguard,robust-ldm \
  --glaze_style_backend onnx_mosaic \
  --glaze_style_onnx_path model/style_transfer/mosaic-9.onnx \
  --budget_l2_grid 4/255,8/255,12/255 \
  --budget_lpips_grid 0.1,0.2,0.5 \
  --prompt_file exp_plan/prompts_style.txt \
  --num_random_seeds 8
```

Run in background:

```bash
GLAZE_STYLE_BACKEND=onnx_mosaic \
GLAZE_STYLE_ONNX_PATH=model/style_transfer/mosaic-9.onnx \
nohup bash scripts/run_full_pipeline.sh \
  > experiments/outputs/full-run.log 2>&1 &
```

## Campaign (10 groups)

Run 5 Concept groups + 5 WikiArt artist groups (includes `boris_kustodiev`) with auto-resume:

```bash
bash scripts/run_full_pipeline_campaign10.sh
```

Useful overrides:

```bash
CAMPAIGN_ID=campaign10-v1 \
PIPELINE_BUDGET_L2_GRID=8/255,12/255 \
PIPELINE_BUDGET_LPIPS_GRID=0.1,0.2,0.5 \
PIPELINE_METHODS=chocolatent,glaze,photoguard,robust-ldm,mist \
bash scripts/run_full_pipeline_campaign10.sh
```
