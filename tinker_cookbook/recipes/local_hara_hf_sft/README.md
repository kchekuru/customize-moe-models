# Local HF/TRL Supervised Fine-Tuning (Works With or Without TINKER_API_KEY)

This recipe fine-tunes a local Hugging Face causal LM using `trl.SFTTrainer` +
`peft` LoRA, while reusing `tinker-cookbook`'s renderer and dataset-building
logic. It works in two modes:

- **No `TINKER_API_KEY` set**: renderer name is resolved from a local offline
  mapping (`_LOCAL_RENDERER_FALLBACKS` in `train_trl_sft.py`). No network
  calls to the Tinker service occur anywhere in this path.
- **`TINKER_API_KEY` set**: renderer name is resolved via the
  Tinker-authoritative `model_info.get_recommended_renderer_name`, per
  `AGENTS.md` convention #4 (never hardcode renderer names). If that lookup
  fails for any reason, it falls back to the local mapping automatically.

Either way, model training itself always runs locally via `trl`/`peft` — the
Tinker service is never used for the actual training compute.

## What's reused vs. replaced

| Layer | Source | Notes |
|---|---|---|
| Renderer selection | `resolve_renderer_name` (this recipe) | Tries `model_info.get_recommended_renderer_name` first, falls back to a local mapping |
| Prompt formatting + loss mask | `tinker_cookbook.renderers.get_renderer(...).build_supervised_example` (via `conversation_to_datum`) | Matches exact chat-template + assistant-only masking used in Tinker-backed training |
| Datum construction | `tinker_cookbook.supervised.data.conversation_to_datum` | Cookbook helper, not manual `Datum` construction |
| Training backend | `trl.SFTTrainer` + `peft.LoraConfig` | Always runs on local CPU/GPU/MPS — **replaces** Tinker's `training_client`/`service_client` |

## Cost model

- No Tinker API billing required (works fully offline, no `TINKER_API_KEY`).
- If `TINKER_API_KEY` is set, it's only used for an optional renderer-name
  lookup — not for any billed training/sampling calls.
- No Hugging Face Hub billing for standard public model downloads/inference.
- Only cost is your own local compute (CPU/GPU/MPS time, electricity).

## Prerequisites

```bash
.venv/bin/python -m pip install transformers trl peft datasets accelerate bitsandbytes chz
```

## Data format

Training data is a JSONL file, one conversation per line:

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

See `sample_train.jsonl` in this directory for a minimal agricultural-domain
example (soil pH, crop rotation, irrigation, and farm machinery maintenance
Q&A pairs) used for pipeline smoke-testing.

## Running locally (no API key)

```bash
unset TINKER_API_KEY
.venv/bin/python - <<'PY'
from tinker_cookbook.recipes.local_hara_hf_sft.train_trl_sft import LocalHFSFTConfig, run

cfg = LocalHFSFTConfig(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    train_jsonl_path="tinker_cookbook/recipes/local_hara_hf_sft/sample_train.jsonl",
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

## Verifying renderer resolution in both modes

```bash
unset TINKER_API_KEY
.venv/bin/python -c "from tinker_cookbook.recipes.local_hara_hf_sft.train_trl_sft import resolve_renderer_name; print(resolve_renderer_name('Qwen/Qwen2.5-0.5B-Instruct'))"

export TINKER_API_KEY=your_key_here
.venv/bin/python -c "from tinker_cookbook.recipes.local_hara_hf_sft.train_trl_sft import resolve_renderer_name; print(resolve_renderer_name('Qwen/Qwen2.5-0.5B-Instruct'))"
```

Both should print `qwen3`.

## Hardware notes (Apple Silicon)

- CPU-only training is slow; prefer small models (<=1B params) for local
  smoke tests.
- MPS (Metal) is used automatically when available via `torch.backends.mps`;
  device placement is automatic in current `transformers` (no
  `use_mps_device` flag needed).
- LoRA (already configured via `LoraConfig`) keeps memory usage low — full
  fine-tuning even of small models is memory-intensive on unified memory Macs.

## Extending to Mixture-of-Experts (MoE) models

To fine-tune MoE models (e.g. `Qwen/Qwen1.5-MoE-A2.7B`,
`mistralai/Mixtral-8x7B-v0.1`), update `LocalHFSFTConfig.model_id`, add an
entry to `_LOCAL_RENDERER_FALLBACKS` if needed, and set
`lora_target_modules` to match that model's MoE expert/gate projection layer
names (e.g. `gate_proj`, `up_proj`, `down_proj`, or `w1`/`w2`/`w3` depending
on architecture).
