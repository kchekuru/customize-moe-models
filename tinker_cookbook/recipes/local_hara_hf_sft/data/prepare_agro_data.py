"""Build one agriculture-only SFT corpus from confirmed source datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _procurement_record(row: list[str]) -> dict[str, list[dict[str, str]]]:
    district, commodity, season, quantity, farmers, amount = row
    question = f"What were the procurement figures for {commodity} in {district} during {season}?"
    answer = (
        f"For {commodity} in {district} during {season}, the recorded "
        f"procurement quantity was {quantity} metric tonnes, involving "
        f"{farmers} farmers, with a recorded amount of Rs {amount}."
    )
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer using the agricultural government statistics in the "
                    "source record. Preserve the source units and do not infer "
                    "missing values."
                ),
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def load_procurement(path: Path) -> list[dict[str, list[dict[str, str]]]]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError(f"Expected a data list in {path}")
    records: list[dict[str, list[dict[str, str]]]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 6:
            raise ValueError(f"Unexpected procurement row: {row!r}")
        records.append(_procurement_record([str(value) for value in row]))
    return records


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=True) + "\n")


def split_records(
    records: list[dict[str, list[dict[str, str]]]], validation_stride: int = 5
) -> tuple[list[dict[str, list[dict[str, str]]]], list[dict[str, list[dict[str, str]]]]]:
    validation = [record for index, record in enumerate(records) if index % validation_stride == 0]
    training = [record for index, record in enumerate(records) if index % validation_stride != 0]
    return training, validation


if __name__ == "__main__":
    data_dir = Path(__file__).parent
    mandi_train = load_jsonl(data_dir / "mandi_train.jsonl")
    mandi_validation = load_jsonl(data_dir / "mandi_validation.jsonl")
    procurement_train, procurement_validation = split_records(
        load_procurement(data_dir / "procurement-details.json")
    )
    train_records = mandi_train + procurement_train
    validation_records = mandi_validation + procurement_validation
    write_jsonl(data_dir / "agro_train.jsonl", train_records)
    write_jsonl(data_dir / "agro_validation.jsonl", validation_records)
    print(
        f"Wrote {len(train_records)} training and {len(validation_records)} "
        f"validation examples to {data_dir}"
    )
