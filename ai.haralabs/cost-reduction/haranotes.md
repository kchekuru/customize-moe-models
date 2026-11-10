#cTinker API free alternatives
pytest tinker_cookbook


#wont-pay-for-api-key
1) leverage existing cookbook chapters related to renderes, dataset builders, and loss mask logic
2) use trl 'fine tunning' capabilities

3) leverage on agricultural machinery

#progress-summary (2026-09-14)

1) Verified local unit tests run with zero Tinker API/network calls:
   - `pytest tinker_cookbook/` → 1729 passed, 63 skipped, 0 unexpected warnings
   - Fixed missing optional deps: `modal`, `litellm`, `textarena`
   - Fixed real deprecation: `asyncio.iscoroutinefunction` -> `inspect.iscoroutinefunction` in `tinker_cookbook/tool_use/tools.py`
   - Added scoped `filterwarnings` to `pyproject.toml` for known/expected test-generated warnings only

2) Built a local, API-key-free SFT training path (Option 3: HF/TRL migration):
   - New recipe: `tinker_cookbook/recipes/local_hara_hf_sft/train_trl_sft.py`
   - Reuses cookbook artifacts that require no network access:
     - `tinker_cookbook.renderers.get_renderer` + `build_supervised_example` (chat template + loss mask)
     - `tinker_cookbook.supervised.data.conversation_to_datum` (Datum construction)
   - Replaces Tinker's `training_client`/`service_client` with `trl.SFTTrainer` + `peft.LoraConfig` (local CPU/GPU/MPS)
   - `resolve_renderer_name()` works with or without `TINKER_API_KEY`:
     - if key set: tries `model_info.get_recommended_renderer_name` (Tinker-authoritative)
     - always falls back to a local offline renderer mapping (`_LOCAL_RENDERER_FALLBACKS`) if no key or lookup fails
   - Sample agri-domain data fixture: `tinker_cookbook/recipes/local_hara_hf_sft/sample_train.jsonl`
   - Docs: `tinker_cookbook/recipes/local_hara_hf_sft/README.md`

3) Confirmed no billed API usage anywhere in this path:
   - No `TINKER_API_KEY` required
   - No Hugging Face paid API calls (only local model download + local training)
   - Only cost is local compute (CPU/GPU/MPS time)

4) Environment note: use `uv run` to avoid conda `base` env shadowing `.venv`'s `python`/`pip` in `$PATH`.

5) Result: first local training run (1 epoch) on `Qwen/Qwen2.5-0.5B-Instruct` completed successfully using
   `tinker_cookbook/recipes/local_hara_hf_sft/sample_train.jsonl`, output saved to `./outputs/local_hf_sft_agri`.

Next: increase epochs, analyze loss curve, then consider scaling to a real MoE model
(update `model_id`, extend `_LOCAL_RENDERER_FALLBACKS` if needed, and set `lora_target_modules`
to match the MoE model's expert/gate projection layer names).

