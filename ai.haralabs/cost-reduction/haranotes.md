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

#progress-summary (2026-11-27)

1) Full no-key test suite is green:
   - `unset TINKER_API_KEY; .venv/bin/python -m pytest -q`
   - 2247 passed, 248 skipped (API-key smoke tests only)
   - Fixed SDK trace tripwire: instrument `TrainingClient.save_weights_external_async`
     in `tinker_cookbook/utils/trace.py` (tinker 0.29.0).
   - Fixed `tests/conftest.py` so missing-key skips apply only to `tests/` smoke tests;
     `tinker_cookbook/` unit tests and `tests/downstream_compat/` now run without a key.

2) Local SFT label alignment corrected in
   `tinker_cookbook/recipes/local_hara_hf_sft/train_trl_sft.py`:
   - A tinker `Datum` is already next-token shifted; HF/TRL shifts labels again
     internally, so dataset labels had to be realigned with the reconstructed full
     token sequence to avoid training one token off.
   - Added offline unit tests in
     `tinker_cookbook/recipes/local_hara_hf_sft/train_trl_sft_test.py`
     covering no-key renderer resolution, fallback on lookup failure, JSONL loading,
     and HF label alignment.

3) End-to-end offline fine-tuning verified with no key and no network:
   - `unset TINKER_API_KEY; HF_HUB_OFFLINE=1 .venv/bin/python ... run(LocalHFSFTConfig(...))`
   - `Qwen/Qwen2.5-0.5B-Instruct`, 2 sample rows, train + eval, 1 epoch
   - train_loss 2.155, eval_loss 1.197, `NO_KEY_SFT_SMOKE_OK`
   - LoRA adapter, tokenizer, chat template and `training_log.json` saved.
   - Default `train_on_what` is now `LAST_ASSISTANT_MESSAGE` (offline qwen3
     fallback does not satisfy the sequence extension property).

4) Documentation paths corrected in `ai.haralabs/cost-reduction/cost-reduction-readme.md`
   (`local_hf_sft` -> `local_hara_hf_sft`), with no-key verification commands and
   `HF_HUB_OFFLINE=1` guidance for fully offline runs.

5) Repo-wide lint/format + environment hygiene:
   - `ruff format .` cleaned 25 pre-existing unformatted docs/tutorial files;
     `ruff format --check .` → 544 files already formatted.
   - Fixed two `RUF036` union-order errors in `tinker_cookbook/recipes/math_rl/math_env.py`.
   - Added `local-sft` optional dependency extra (`accelerate`, `peft`, `trl`) to
     `pyproject.toml` and documented `uv pip install --python .venv/bin/python -e '.[local-sft]'`.
   - Local SFT tests now skip cleanly when the optional extra is not installed.
   - VS Code/Pylance now uses `.venv/bin/python` (old setting was a Windows path);
     pyright/ty checks on `tinker_cookbook/utils/trace.py` are clean.

