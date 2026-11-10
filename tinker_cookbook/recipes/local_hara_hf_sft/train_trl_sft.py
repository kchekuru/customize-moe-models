from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import chz
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from trl import SFTTrainer

from tinker_cookbook import model_info
from tinker_cookbook.renderers import Message, Role, get_renderer
from tinker_cookbook.supervised.data import TrainOnWhat, conversation_to_datum

IGNORE_INDEX = -100

# Fallback renderer mapping used when TINKER_API_KEY is unavailable and
# model_info.get_recommended_renderer_name cannot reach the Tinker service.
_LOCAL_RENDERER_FALLBACKS: dict[str, str] = {
    "qwen2.5": "qwen3",
    "qwen1.5-moe": "qwen3",
    "qwen3": "qwen3",
    "mixtral": "llama3",
    "llama-3": "llama3",
}


def resolve_renderer_name(model_id: str) -> str:
    """Resolve a renderer name; works with or without TINKER_API_KEY.

    Prefers the Tinker-authoritative lookup (convention #4: never hardcode
    renderer names) when a key is set, falling back to a local offline
    mapping otherwise.
    """
    if os.environ.get("TINKER_API_KEY"):
        try:
            return model_info.get_recommended_renderer_name(model_id)
        except Exception:
            pass  # fall through to local mapping below

    lowered = model_id.lower()
    for key, renderer_name in _LOCAL_RENDERER_FALLBACKS.items():
        if key in lowered:
            return renderer_name

    raise ValueError(
        f"No renderer available for model_id={model_id!r}: TINKER_API_KEY is "
        "not set (or the lookup failed) and no local fallback mapping exists. "
        "Add an entry to _LOCAL_RENDERER_FALLBACKS, or set TINKER_API_KEY."
    )


@chz.chz
class LocalHFSFTConfig:
    """Config for local, API-key-free SFT via Hugging Face `trl`.

    Mirrors the shape of `SupervisedDatasetBuilder`-style configs elsewhere in
    the cookbook, but targets a local HF/`peft`/`trl` training backend instead
    of the Tinker service.
    """

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    train_jsonl_path: str = ""
    eval_jsonl_path: str = ""
    output_dir: str = "./outputs/local_hf_sft"
    max_length: int = 2048
    train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def load_conversations_jsonl(path: str) -> list[list[dict[str, str]]]:
    """Load a JSONL file of `{"messages": [...]}` records."""
    conversations: list[list[dict[str, str]]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            conversations.append(record["messages"])
    return conversations


def conversations_to_hf_dataset(
    conversations: list[list[dict[str, str]]],
    tokenizer: PreTrainedTokenizerBase,
    renderer_name: str,
    max_length: int,
    train_on_what: TrainOnWhat,
) -> Dataset:
    # get_renderer's `tokenizer` param is typed Any; cast avoids a stub
    # resolution conflict with an unrelated `tokenizers.Tokenizer` symbol.
    renderer = get_renderer(renderer_name, cast(Any, tokenizer))
    rows: list[dict[str, list[int]]] = []

    for conv in conversations:
        messages = [Message(role=Role(m["role"]), content=m["content"]) for m in conv]

        datum = conversation_to_datum(
            conversation=messages,
            renderer=renderer,
            max_length=max_length,
            train_on_what=train_on_what,
        )

        input_ids_T = datum.model_input.to_ints()
        weights_T = datum.loss_fn_inputs["weights"].to_numpy()
        target_ids_T = datum.loss_fn_inputs["target_tokens"].to_numpy()

        n = min(len(input_ids_T), len(weights_T), len(target_ids_T))
        labels_T = [int(target_ids_T[i]) if weights_T[i] > 0.0 else IGNORE_INDEX for i in range(n)]

        rows.append(
            {
                "input_ids": list(input_ids_T[:n]),
                "labels": labels_T,
                "attention_mask": [1] * n,
            }
        )

    return Dataset.from_list(rows)


def resolve_device_dtype() -> tuple[str | None, torch.dtype]:
    if torch.cuda.is_available():
        return "auto", torch.bfloat16
    if torch.backends.mps.is_available():
        return None, torch.float32
    return None, torch.float32


def run(config: LocalHFSFTConfig) -> None:
    if not config.train_jsonl_path:
        raise ValueError("train_jsonl_path is required")

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Works with or without TINKER_API_KEY (see resolve_renderer_name).
    renderer_name = resolve_renderer_name(config.model_id)

    conversations = load_conversations_jsonl(config.train_jsonl_path)
    train_dataset = conversations_to_hf_dataset(
        conversations=conversations,
        tokenizer=tokenizer,
        renderer_name=renderer_name,
        max_length=config.max_length,
        train_on_what=config.train_on_what,
    )
    eval_dataset = None
    if config.eval_jsonl_path:
        eval_conversations = load_conversations_jsonl(config.eval_jsonl_path)
        eval_dataset = conversations_to_hf_dataset(
            conversations=eval_conversations,
            tokenizer=tokenizer,
            renderer_name=renderer_name,
            max_length=config.max_length,
            train_on_what=config.train_on_what,
        )

    device_map, torch_dtype = resolve_device_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.lora_target_modules),
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=1,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )

    trainer.train()

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with (Path(config.output_dir) / "training_log.json").open("w", encoding="utf-8") as log_file:
        json.dump(trainer.state.log_history, log_file, indent=2)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    chz.nested_entrypoint(run)
