python src/open_r1/grpo.py \
    --output_dir checkpoints/testQwenRun \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name leonardPKU/clevr_cogen_a_train \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name qwen-test-on-clevr \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 2

