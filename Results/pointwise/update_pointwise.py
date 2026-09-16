"""Aggregate only the dedicated new pointwise output directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = (
    ROOT / "LLM_prompt_attack" / "outputs" / "pointwise_new_all_models",
    ROOT / "LLM_prompt_attack" / "outputs" / "pointwise_new",
    ROOT / "LLM_prompt_attack" / "outputs" / "gptoss_pointwise_low_reasoning",
)
OUTPUT = ROOT / "Results" / "pointwise" / "pointwise.csv"
DATASETS = {
    "msmarco-passage/trec-dl-2019": "TREC-DL-2019",
    "msmarco-passage/trec-dl-2020": "TREC-DL-2020",
}
ATTACKS = {"so": "DOH", "sd": "DCH", "qi": "QI"}
MODELS = {
    "qwen3-4b": "Qwen3-4B",
    "qwen/qwen3-4b": "Qwen3-4B",
    "qwen.qwen3-32b-v1:0": "Qwen3-32B",
    "openai.gpt-oss-20b-1:0": "GPT-OSS-20B",
    "meta.llama3-8b-instruct-v1:0": "Llama-3-8B",
    "meta.llama3-70b-instruct-v1:0": "Llama3 70B",
}
FIELDS = (
    "Dataset", "Model", "Attack", "Prompt", "Requested", "Valid attacked",
    "Discarded", "Attack success", "Attack success (%)", "Date", "Source",
)
KEY_FIELDS = ("Dataset", "Model", "Attack", "Prompt")


def _row(record: dict, source: Path) -> dict | None:
    if record.get("ranking_scheme") != "pointwise":
        return None
    dataset = record.get("dataset_name")
    attack = record.get("attack_type")
    if dataset not in DATASETS or attack not in ATTACKS:
        return None
    if record.get("attack_position") != "back":
        return None
    requested = int(record.get("original_total_rankings", record.get("total_queries", 0)))
    valid = int(record.get("attacked_valid_rankings", record.get("total_queries", 0)))
    success = int(record.get("pointwise_flip_count", 0))
    if requested <= 0 or valid < 0 or success < 0 or success > valid:
        return None
    model_name = str(record.get("model_name", "Unknown")).lower()
    model = MODELS.get(model_name, str(record.get("model_name", "Unknown")))
    prompt = "Defense" if record.get("prompt_mode") == "defense" else "Default"
    return {
        "Dataset": DATASETS[dataset], "Model": model, "Attack": ATTACKS[attack],
        "Prompt": prompt, "Requested": requested, "Valid attacked": valid,
        "Discarded": requested - valid, "Attack success": success,
        "Attack success (%)": 100 * success / requested,
        "Date": str(record.get("date", "")),
        "Source": source.relative_to(ROOT).as_posix(),
    }


def _load_existing() -> dict[tuple[str, ...], dict]:
    if not OUTPUT.exists():
        return {}
    with OUTPUT.open(newline="", encoding="utf-8") as handle:
        return {tuple(row[field] for field in KEY_FIELDS): row for row in csv.DictReader(handle)}


def main(input_dirs: list[Path]) -> int:
    rows = _load_existing()
    for raw_input_dir in input_dirs:
        # CLI paths are commonly supplied relative to the repository root.
        # Normalize them before walking so Source can always be made relative
        # to ROOT, regardless of how the updater was invoked.
        input_dir = raw_input_dir if raw_input_dir.is_absolute() else ROOT / raw_input_dir
        if not input_dir.exists():
            continue
        for path in sorted(input_dir.rglob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = _row(json.loads(line), path)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    if row is not None:
                        rows[tuple(row[field] for field in KEY_FIELDS)] = row
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    print(f"Wrote {len(rows)} pointwise rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", type=Path, dest="input_dirs")
    args = parser.parse_args()
    raise SystemExit(main(args.input_dirs or list(DEFAULT_INPUTS)))
