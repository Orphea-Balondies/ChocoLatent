EXP_ID=wikiart-boris-full \
IMAGE_ROOT=datasets/partial/WikiArt_hot40_top15_artist \
IMAGE_DIRNAME=boris_kustodiev \
PROMPT_FILE=datasets/partial/WikiArt_hot40_top15_artist/boris_kustodiev/prompts_boris_kustodiev.txt \
SEED_FILE=exp_plan/seeds_eval.txt \
CAPTION_TEMPLATE="a painting in the style of {name}" \
PROTECT_INITIAL_LR=0.5 \
PROTECT_BATCH_SIZE=2 \
PROTECT_NAN_LR_DECAY=0.7 \
PROTECT_NAN_MIN_LR=1e-5 \
PROTECT_NAN_MAX_RECOVERIES=32 \
HEARTBEAT_SEC=60 \
STOP_ON_ERROR=false \
SKIP_PROTECT=true \
METHODS=chocolatent \
bash scripts/run_full_pipeline.sh
