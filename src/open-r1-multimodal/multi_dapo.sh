#!/bin/bash

# Move to repo root if needed
cd ~/R1-V/src/open-r1-multimodal

# Debug logging
export DEBUG_MODE="true"
export LOG_PATH="./debug_log_dapo_2p5vl.txt"

# Dataset and model config
DATASET_NAME="kxxinDave/GEOVQ_Qwen2_5_Geo_Description_Subset_500"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="checkpoints/Qwen2_5VL_3B_DAPO"
RUN_NAME="Qwen2_5VL_3B_DAPO"

# Launch distributed training
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=12345 \
    -m open_r1.dapo \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --max_prompt_length 512 \
    --max_completion_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --logging_steps 1 \
    --disable_tqdm false \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 12845056 \
    --min_pixels 3136 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 2 \
    --epsilon_low 0.20 \
    --epsilon_high 0.28 \
    --optim "adamw_torch_fused"
