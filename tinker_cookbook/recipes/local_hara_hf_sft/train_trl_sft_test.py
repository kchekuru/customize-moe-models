"""Tests for the local, API-key-free HF/TRL SFT path."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

# The local SFT recipe uses optional training dependencies.
pytest.importorskip("peft")
pytest.importorskip("trl")

from tinker_cookbook import model_info
from tinker_cookbook.recipes.local_hara_hf_sft.train_trl_sft import (
    IGNORE_INDEX,
    conversations_to_hf_dataset,
    load_conversations_jsonl,
    resolve_renderer_name,
)
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.tokenizer_utils import Tokenizer


class _FakeTokenizer:
    """Deterministic, offline tokenizer standing in for a HF tokenizer."""

    bos_token = None
    eos_token_id = None
    name_or_path = "fake-tokenizer"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) % 128 for char in text]

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(token) for token in tokens)


def test_resolve_renderer_name_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    assert resolve_renderer_name("Qwen/Qwen2.5-0.5B-Instruct") == "qwen3"
    assert resolve_renderer_name("Qwen/Qwen1.5-MoE-A2.7B") == "qwen3"


def test_resolve_renderer_name_rejects_unknown_model_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="no local fallback mapping exists"):
        resolve_renderer_name("some-unknown-model")


def test_resolve_renderer_name_uses_authoritative_lookup_with_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    looked_up: list[str] = []

    def fake_lookup(model_id: str) -> str:
        looked_up.append(model_id)
        return "qwen3_instruct"

    monkeypatch.setattr(model_info, "get_recommended_renderer_name", fake_lookup)
    assert resolve_renderer_name("Qwen/Qwen3-8B") == "qwen3_instruct"
    assert looked_up == ["Qwen/Qwen3-8B"]


def test_resolve_renderer_name_falls_back_when_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    def failing_lookup(model_id: str) -> str:
        raise RuntimeError("Tinker service unreachable")

    monkeypatch.setattr(model_info, "get_recommended_renderer_name", failing_lookup)
    assert resolve_renderer_name("Qwen/Qwen2.5-0.5B-Instruct") == "qwen3"


def test_load_conversations_jsonl_skips_blank_lines(tmp_path: Any) -> None:
    path = tmp_path / "train.jsonl"
    first = {"messages": [{"role": "user", "content": "hi"}]}
    second = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
    }
    path.write_text("\n".join([json.dumps(first), "", json.dumps(second), ""]), encoding="utf-8")

    assert load_conversations_jsonl(str(path)) == [first["messages"], second["messages"]]


def test_conversations_to_hf_dataset_aligns_labels_for_hf_shift() -> None:
    conversations = [
        [
            {"role": "user", "content": "What is soil pH?"},
            {"role": "assistant", "content": "It measures acidity."},
        ]
    ]
    dataset = conversations_to_hf_dataset(
        conversations=conversations,
        tokenizer=cast(Tokenizer, _FakeTokenizer()),
        renderer_name="role_colon",
        max_length=128,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    row = dataset[0]
    input_ids = row["input_ids"]
    labels = row["labels"]
    assert len(input_ids) == len(labels) == len(row["attention_mask"])

    # HF causal LM loss uses labels[i + 1] to score the prediction from
    # input_ids[i], so a trained label at position j must match input_ids[j].
    # A double next-token shift would instead put tokens[j + 1] there.
    assert labels[0] == IGNORE_INDEX
    trained_positions = [idx for idx, label in enumerate(labels) if label != IGNORE_INDEX]
    assert trained_positions
    assert all(labels[idx] == input_ids[idx] for idx in trained_positions)
    assert len(trained_positions) < len(input_ids)
