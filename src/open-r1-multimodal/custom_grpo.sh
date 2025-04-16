# cd src/open-r1-multimodal

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt"

HF_DATASET = 'leonardPKU/clevr_cogen_a_train'
OUTPUT_DIR = 'checkpoints/testQwenRun'
RUN_NAME = 'qwen-test-on-clevr-multiGPU'

CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4" torchrun --nproc_per_node=3 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=12345 \
    src/open_r1/grpo.py \
    --use_vllm True \
    --output_dir checkpoints/testQwenRun \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name leonardPKU/clevr_cogen_a_train \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name qwen-test-on-clevr-multiGPU \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 3
