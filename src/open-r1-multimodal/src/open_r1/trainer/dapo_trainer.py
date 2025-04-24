import os
import textwrap
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Optional, Sized, Union

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoModelForVision2Seq,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    ProgressCallback,
    is_wandb_available,
)
import gc
import types
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from types import NoneType
from transformers.utils import is_peft_available
from trl.trainer import GRPOConfig
from peft import PeftConfig
from open_r1.trainer.dapoConfig import DAPOConfig
from open_r1.trainer.padding_helper import enforceLeftAttention, enforceLeftPad
from trl.data_utils import apply_chat_template, maybe_apply_chat_template, is_conversational
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
import copy
from open_r1.trainer.dapoComponents import DAPOAdvantage, DAPOLoss

if is_peft_available():
    from peft import PeftConfig, get_peft_model
if is_wandb_available():
    import wandb
import sys
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLDapoTrainer(Trainer):
    """
    Trainer for the Dynamic Sampling optimisation method (Without Dynamic Sampling - Trust me, the irony is not lost on me). This algorithm was proposed by ByteDance and Tsinghua University.

        Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
        epsilon_low (float): Lower clipping bound.
        epsilon_high (float): Upper clipping bound.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: DAPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: Optional[tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional[PeftConfig] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = 'flash_attention_2',
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.28, #Clipping bounds for the reward function.
        overlong_penalty_enabled: bool = True,
        overlong_buffer_len: int = 8,
        overlong_penalty_factor: float = 1.0, #Penalty factor for the overlong penalty. Can try gentler penalties.
    ):
        #Arguments.
        if args is None:
            modelName = model if isinstance(model, str) else model.name_or_path
            modelName = modelName.split('/')[-1]
            args = DAPOConfig(modelName=modelName)

        #models.
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs['attn_implementation'] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `DAPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                model.config.use_cache = False
                model.gradient_checkpointing_enable()
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                model.config.use_cache = False
                model.gradient_checkpointing_enable()
            elif "SmolVLM" in model_id:
                model = AutoModelForVision2Seq.from_pretrained(model_id, **model_init_kwargs)
                model.config.use_cache = False
                model.gradient_checkpointing_enable()
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config.name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed model init kwargs to the Trainer, but you passed a pre-trained model to the Trainer."
                )
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        #Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                # self.ref_model.config.use_cache = False
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                # self.ref_model.config.use_cache = False
            elif "SmolVLM" in model_id:
                self.ref_model = AutoModelForVision2Seq.from_pretrained(model_id, **model_init_kwargs)
                # self.ref_model.config.use_cache = False
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            #If config is not provided, we will choose the base model as the reference model.
            self.ref_model = create_reference_model(model)
        else:
            #If config is used, we can disable the adapters to use the base model instead.
            self.ref_model = None

        #Define the processor class.
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "SmolVLM" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id, padding_side="left")
                processing_class.tokenizer.padding_side = "left"
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id
                processing_class.tokenizer.padding_side = "left"



        #Sanity print.
        print(f'Tokenizer padding side: {processing_class.tokenizer.padding_side}')
        #Remove the monkey patch if it causes conflict outside of deepspeed.
        # print(f'Padding side of the model: {model.model._update_causal_mask.__self__.config.padding_side}')
        #Reward functions.
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(reward_func, num_labels=1, **model_init_kwargs)

        self.reward_funcs = reward_funcs

        #Reward processing classes.
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        # print(f'Before the troublesome loop, reward processing classes are of the type: {type(reward_processing_classes)}')
        # if isinstance(reward_processing_classes, NoneType):
        #     print('Reward processing classes are indeed None. Fixing that.')
        #     reward_processing_classes = [None] * len(reward_funcs)

        # print(f'After the troublesome loop, reward processing classes are of the type: {type(reward_processing_classes)}')


        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel): #If the reward function is a pretrained model, and the processing classes have not been defined, we use the AutoTokenizer class provided by Huggingface.
                print(f'Processing class for rewards is Indeed an LLM.')
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                    #The reward model computes the reward for the latest non-padded tokens in the input sequence.
                    #Therefore, we need to set the pad token to the eos token.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # print(f'The type of a Reward function in GRPO setup: {type(self.reward_processing_classes[0])}')


        #Data collator.
        def data_collator(features):
            return features #Not required as no data collation is needed in DAPO.

        #Training arguments.
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1, #Hacking, to maximise exploration.
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
            # padding_side='left', #Doesn't work for the generation config.
        )
        self.epsilon_low = epsilon_low #Lower clipping bound
        self.epsilon_high = epsilon_high #Upper clipping bound
        self.overlong_penalty_enabled = overlong_penalty_enabled
        self.overlong_buffer_len = overlong_buffer_len
        self.overlong_penalty_factor = overlong_penalty_factor

        model.warnings_issued['estimate_tokens'] = True

        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    def _prepare_inputs(self, inputs):
        return inputs

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        if image_grid_thw is None: #Huggingface SMOL models don't have this attribute.
            logits = model(input_ids, attention_mask=attention_mask).logits
        else:
            logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits
        logits = logits[:, :-1]
        input_ids = input_ids[:, 1:]
        #Compute the log probabilities, use a loop to avoid memory issues.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=-1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("DAPOTrainer does not support returning outputs.")

        device = self.accelerator.device
        prompts = [x['prompt'] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)['prompt'] for example in inputs]
        images = [x['image'] for x in inputs]

        #some more tiptoeing to make this compatible with SMOL faimly.
        model_id = self.model.config._name_or_path.lower()
        is_smol = any(key in model_id for key in ['smolvlm', 'idefics'])
        img_token_id = getattr(self.processing_class.tokenizer, "image_token_id", None)

        # print(f'Status of the image token ID: {img_token_id}')

        if is_smol and img_token_id is not None:
            print(f'Injecting <image> tokens for compatibility with SMOL.')
            for i in range(len(prompts_text)):
                image_count_in_prompt = prompts_text[i].count("<image>")
                img_entry = inputs[i]['image']
                image_count = len(img_entry) if isinstance(img_entry, list) else 1
                if image_count_in_prompt < image_count:
                    num_to_inject = image_count - image_count_in_prompt
                    prompts_text[i] = ("<image> " * num_to_inject).strip() + " " + prompts_text[i]


        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors='pt',
            padding="longest",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        if hasattr(prompt_inputs, "input_ids") and isinstance(prompt_inputs, dict):
            self.processing_class.tokenizer.padding_side = "left" 
        #Diagnostic print.
        # imageTokenId = getattr(self.processing_class.tokenizer, "image_token_id", None)
        # if imageTokenId is not None:
        #     print(f"Contains image token ID? {imageTokenId in prompt_inputs['input_ids'][0]}")
        # else:
        #     print("image_token_id not defined for this tokenizer.")


        prompt_ids, prompt_mask = prompt_inputs['input_ids'], prompt_inputs['attention_mask']
        pixel_values, image_grid_thw = prompt_inputs['pixel_values'], prompt_inputs.get('image_grid_thw', None)

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        # print(f'Diagnostic Print: While using DeepSpeed, right before tokenisation. This is the padding side of the model: {self.processing_class.tokenizer.padding_side}')
        #If the below couple of lines are not compatible with torch run, please remove them.
        # pad_token_id = self.processing_class.tokenizer.pad_token_id
        # prompt_inputs['input_ids'] = enforceLeftPad(prompt_inputs['input_ids'], pad_token_id)
        # prompt_inputs['attention_mask'] = enforceLeftAttention(prompt_inputs['input_ids'], pad_token_id)
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
            )
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Corrected completion mask (includes EOS)
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.int64, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)

        if image_grid_thw is not None:
            image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)

        per_token_logps = self._get_per_token_logps(
            model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
        )[:, prompt_length - 1:]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
        self.processing_class.tokenizer.padding_side = "left"
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        repeated_prompts = [p for p in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(repeated_prompts), len(self.reward_funcs), device=device)

        # assert all(callable(rpc) for rpc in self.reward_processing_classes), "One of the reward tokenizers is None or invalid!"
        torch.cuda.empty_cache()
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(repeated_prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(repeated_prompts, completions)]
                torch.cuda.empty_cache()
                gc.collect()
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=repeated_prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)
        #Soft overlong penalty support. ()

        if hasattr(self, "overlong_penalty_enabled") and self.overlong_penalty_enabled:
            overlong_buffer_len = getattr(self, "overlong_buffer_len", 8)
            penalty_factor = getattr(self, "overlong_penalty_factor", 1.0)

            expected_length = self.max_completion_length - overlong_buffer_len
            actual_length = completion_mask.sum(1).float()
            exceed_length = actual_length - expected_length

            #Calculate the penalty
            overlong_penalty = torch.zeros_like(rewards, device=device)
            mask = (exceed_length > 0) & (exceed_length <= overlong_buffer_len)
            overlong_penalty[mask] = (exceed_length[mask] / overlong_buffer_len) * penalty_factor
            overlong_penalty[exceed_length > overlong_buffer_len] = -penalty_factor
            rewards += overlong_penalty
            self._metrics["overlong_penalty"].append(overlong_penalty.mean().item())

        #Compute advantages.
        #Empty the cache
        torch.cuda.empty_cache()
        advantages = DAPOAdvantage(rewards, self.num_generations)
        advantages = advantages.to(device)
        gc.collect()
        #Clear cache before we go to compute the loss
        torch.cuda.empty_cache()
        #Compute Loss
        loss = DAPOLoss(
            log_probs=per_token_logps,
            log_probs_ref=ref_per_token_logps,
            advantage=advantages,
            mask=completion_mask,
            epsilon_low=0.2,
            epsilon_high=0.28
        )
        gc.collect()
        torch.cuda.empty_cache()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics['reward_std'].append(rewards.std().item())

        #Logging metrics (for notebook) You can uncomment this if you want to see the output.
        # print(f'Mean Reward: {rewards.mean().item()}')
        # print(f'Reward Deviation: {rewards.std().item()}')

        # print(f'DAPO Loss: {loss}')

        # # Print the prompt-completion pair at every logging step. #Uncomment to see what the model is doing.
        # if self.args.logging_steps and self.state.global_step % self.args.logging_steps == 0:
        #     print(f'Prompt: {repeated_prompts[0]}')
        #     print(f'Completion: {completions[0][0]["content"]}' if isinstance(completions[0], list) else completions[0])


        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        #printing custom metrics whicj are relevamt.
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        #Uncomment below code to see Global level DAPO stats.
        if self.args.local_rank in [-1, 0]:  # Only print once in distributed
            loss_val = logs.get("loss", None)
            std_val = logs.get("reward_std", None)

            print("\n======= DAPO Training Stats =======")
            if loss_val is not None:
                print(f"DAPO Loss: {loss_val:.4f}")
            if std_val is not None:
                print(f"Reward Deviation: {std_val:.4f}")
            print("===================================\n")
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)

        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))