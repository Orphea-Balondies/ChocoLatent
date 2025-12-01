export MODEL_NAME="model/stable-diffusion-v1-5"
export DATASET_NAME="exp/04lr400e-8fc-01id02il15md01ml/adv_images/lego-minifigure-faces"

accelerate launch code/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME --caption_column="caption" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="exp/04lr400e-8fc-01id02il15md01ml/lora/lego-minifigure-faces" \
  --validation_prompt="lego-minifigure-faces" --validation_epochs=10