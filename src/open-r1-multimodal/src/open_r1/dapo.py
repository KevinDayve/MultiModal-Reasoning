import os
import gc
import re
import time
import torch
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, ProgressCallback
from peft import TaskType
from trl import GRPOConfig, ModelConfig, TrlParser, get_peft_config
from math_verify import parse, verify
from open_r1.trainer.dapo_trainer import Qwen2VLDapoTrainer
from open_r1.trainer.dapoConfig import DAPOConfig
from datasets import load_dataset

@dataclass
class DAPOScriptArguments:
    dataset_name: str = field(default="kxxinDave/GEOVQ_Qwen2_5_Geo_Description_Subset_500")
    dataset_config: Optional[str] = field(default=None)
    dataset_train_split: str = field(default="train")
    dataset_test_split: str = field(default="test")
    reward_funcs: list[str] = field(default_factory=lambda: ["accuracy", "format"])
    max_pixels: Optional[int] = field(default=12845056)
    min_pixels: Optional[int] = field(default=3136)
    use_vllm: bool = field(default=False)


def clear_memory():
    vars_to_clear = ["inputs", "model", "processor", "trainer", "peft_model", "bnb_config"]
    for var in vars_to_clear:
        if var in globals():
            del globals()[var]

    time.sleep(2)
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


clear_memory()

def accuracy_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass

        if reward == 0.0:
            try:
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                reward = -1.0

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "dapo_log.txt")
            with open(log_path, "a") as f:
                f.write(f"\n[{current_time}] Accuracy reward: {reward}\n")
                f.write(f"Content: {content}\nSolution: {sol}\n")

    return rewards

def format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else -1.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


def make_conversation_image(example):
    prompt = f"{example['problem']}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    }


def main():
    parser = TrlParser((DAPOScriptArguments, DAPOConfig, ModelConfig))
    scriptArgs, trainingArgs, modelArgs = parser.parse_args_and_config()

    reward_funcs = [reward_funcs_registry[func] for func in scriptArgs.reward_funcs]
    dataset = load_dataset(scriptArgs.dataset_name, name=scriptArgs.dataset_config)

    if "image" in dataset[scriptArgs.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)
    else:
        raise ValueError("Non-image datasets not yet supported in this script.")

    trainer_cls = Qwen2VLDapoTrainer if not scriptArgs.use_vllm else None
    if trainer_cls is None:
        raise ValueError("We don't support vLLM for DAPO as of now.")
    print("Using:", trainer_cls)

    trainer = trainer_cls(
        model=modelArgs.model_name_or_path,
        reward_funcs=reward_funcs,
        args=trainingArgs,
        train_dataset=dataset[scriptArgs.dataset_train_split],
        eval_dataset=dataset[scriptArgs.dataset_test_split] if trainingArgs.eval_strategy != "no" else None,
        peft_config=get_peft_config(modelArgs),
        attn_implementation=modelArgs.attn_implementation,
        max_pixels=scriptArgs.max_pixels,
        min_pixels=scriptArgs.min_pixels
    )

    trainer.train()
    trainer.save_model(trainingArgs.output_dir)
    if trainingArgs.push_to_hub:
        trainer.push_to_hub(dataset_name=scriptArgs.dataset_name)


if __name__ == "__main__":
    main()