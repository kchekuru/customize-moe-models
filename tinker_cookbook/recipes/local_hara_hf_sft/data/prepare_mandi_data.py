"""Convert downloaded mandi prices into local SFT conversations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TypedDict


class MandiRow(TypedDict):
    State: str
    District: str
    Market: str
    Commodity: str
    Variety: str
    Grade: str
    Arrival_Date: str
    Min_x0020_Price: str
    Max_x0020_Price: str
    Modal_x0020_Price: str


def _price(value: str) -> int:
    return int(value.replace(",", "").strip())


def row_to_record(row: MandiRow) -> dict[str, list[dict[str, str]]]:
    state = row["State"].strip()
    district = row["District"].strip()
    market = row["Market"].strip()
    commodity = row["Commodity"].strip()
    variety = row["Variety"].strip()
    grade = row["Grade"].strip()
    arrival_date = row["Arrival_Date"].strip()
    minimum = _price(row["Min_x0020_Price"])
    maximum = _price(row["Max_x0020_Price"])
    modal = _price(row["Modal_x0020_Price"])

    question = (
        f"What were the mandi prices for {commodity} at {market}, {district}, "
        f"{state} on {arrival_date}?"
    )
    answer = (
        f"On {arrival_date}, {commodity} ({variety}, {grade}) at {market} in "
        f"{district}, {state} had a minimum price of Rs {minimum} per quintal, "
        f"a maximum price of Rs {maximum} per quintal, and a modal price of "
        f"Rs {modal} per quintal."
    )
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def read_rows(input_path: Path) -> list[MandiRow]:
    with input_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_path}")

        required = {
            "State",
            "District",
            "Market",
            "Commodity",
            "Variety",
            "Grade",
            "Arrival_Date",
            "Min_x0020_Price",
            "Max_x0020_Price",
            "Modal_x0020_Price",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")

        return [row for row in reader if any(row.values())]


def write_records(output_path: Path, records: list[dict[str, list[dict[str, str]]]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=True) + "\n")


def convert(input_path: Path, output_path: Path) -> int:
    records = [row_to_record(row) for row in read_rows(input_path)]
    write_records(output_path, records)
    return len(records)


def convert_split(
    input_path: Path,
    train_path: Path,
    validation_path: Path,
    validation_stride: int = 5,
) -> tuple[int, int]:
    rows = read_rows(input_path)
    markets = sorted({row["Market"].strip() for row in rows})
    validation_markets = {
        market for index, market in enumerate(markets) if index % validation_stride == 0
    }
    if len(validation_markets) == len(markets):
        validation_markets.remove(markets[-1])

    train_records = [
        row_to_record(row) for row in rows if row["Market"].strip() not in validation_markets
    ]
    validation_records = [
        row_to_record(row) for row in rows if row["Market"].strip() in validation_markets
    ]
    write_records(train_path, train_records)
    write_records(validation_path, validation_records)
    return len(train_records), len(validation_records)


if __name__ == "__main__":
    recipe_dir = Path(__file__).parent
    train_count, validation_count = convert_split(
        recipe_dir / "mandi.csv",
        recipe_dir / "mandi_train.jsonl",
        recipe_dir / "mandi_validation.jsonl",
    )
    print(
        f"Wrote {train_count} training examples and {validation_count} validation "
        f"examples to {recipe_dir}"
    )
