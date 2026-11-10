# Local HF/TRL Supervised Fine-Tuning (No Tinker API Key Required)

This recipe fine-tunes a local Hugging Face causal LM using `trl.SFTTrainer` +
`peft` LoRA, while reusing `tinker-cookbook`'s renderer and dataset-building
logic. **No `TINKER_API_KEY` or network calls to the Tinker service are made.**

## What's reused vs. replaced

| Layer | Source | Notes |
|---|---|---|
| Renderer selection | `tinker_cookbook.model_info.get_recommended_renderer_name` | Never hardcode renderer names (see `AGENTS.md`) |
| Prompt formatting + loss mask | `tinker_cookbook.renderers.get_renderer(...).build_supervised_example` (via `conversation_to_datum`) | Matches exact chat-template + assistant-only masking used in Tinker-backed training |
| Datum construction | `tinker_cookbook.supervised.data.conversation_to_datum` | Cookbook helper, not manual `Datum` construction |
| Training backend | `trl.SFTTrainer` + `peft.LoraConfig` | Runs on local CPU/GPU/MPS — **replaces** Tinker's `training_client`/`service_client` |

## Cost model

- No Tinker API billing (no `TINKER_API_KEY` needed anywhere in this path).
- No Hugging Face Hub billing for standard public model downloads/inference —
  you are only downloading open model weights and training locally.
- Only cost is your own compute (CPU/GPU/MPS time, electricity).

## Prerequisites

```bash
.venv/bin/python -m pip install transformers trl peft datasets accelerate bitsandbytes chz
```

## Agricultural Domain Data format

Training data is a JSONL file, one conversation per line:

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

See `sample_train.jsonl` in this directory for a minimal agricultural-domain
example (soil pH, crop rotation, irrigation, and farm machinery maintenance
Q&A pairs) used for pipeline smoke-testing.

## Running locally

```bash
.venv/bin/python - <<'PY'
from tinker_cookbook.recipes.local_hf_sft.train_trl_sft import LocalHFSFTConfig, run

cfg = LocalHFSFTConfig(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    train_jsonl_path="tinker_cookbook/recipes/local_hf_sft/sample_train.jsonl",
    output_dir="./outputs/local_hf_sft_agri",
    max_length=2048,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
)

run(cfg)
PY
```

## Hardware notes (Apple Silicon)

- CPU-only training is slow; prefer small models (<=1B params) for local
  smoke tests.
- MPS (Metal) is used automatically when available via `torch.backends.mps`;
  no separate flag is needed (`transformers.TrainingArguments` no longer has
  a `use_mps_device` option — device placement is automatic).
- Use LoRA (already configured via `LoraConfig` in `train_trl_sft.py`) to
  keep memory usage low — full fine-tuning of even small models is
  memory-intensive on unified memory Macs.

## Verifying the config loads correctly

```bash
.venv/bin/python -c "from tinker_cookbook.recipes.local_hf_sft.train_trl_sft import LocalHFSFTConfig; print(LocalHFSFTConfig())"
```

## Extending to Mixture-of-Experts (MoE) models

To fine-tune MoE models (e.g. `Qwen/Qwen1.5-MoE-A2.7B`,
`mistralai/Mixtral-8x7B-v0.1`) instead of the small dense model above, update
`LocalHFSFTConfig.model_id` and set `lora_target_modules` to match that
model's MoE expert/gate projection layer names (e.g. `gate_proj`, `up_proj`,
`down_proj`, or `w1`/`w2`/`w3` depending on architecture).