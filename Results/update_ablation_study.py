"""Aggregate GPT-OSS reasoning-effort ablation summaries."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "LLM_prompt_attack" / "outputs" / "reasoning_ablation" / "gptoss-20b_reasoning_ablation_2019_1000"
DEFAULT_OUTPUT = ROOT / "Results" / "ablation_study.csv"
PATTERN = re.compile(r"pointwise_(so|sd)_(standard|defense)_reasoning_(low|medium|high)_n(\d+)")

FIELDS = ["Reasoning effort", "Dataset", "Model", "Attack", "Prompt", "Passages", "Requested", "Valid attacked", "Discarded", "Attack success", "Attack success (%)", "Date", "Source"]

def main(input_dir: Path = DEFAULT_INPUT, output: Path = DEFAULT_OUTPUT) -> int:
    rows = []
    for path in sorted(input_dir.glob("result_*.jsonl")):
        match = PATTERN.search(path.name)
        if not match:
            continue
        effort, attack, prompt, passages = match.groups()
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if not records:
            continue
        record = records[-1]
        requested = int(record.get("original_total_rankings", record.get("total_queries", 0)))
        success = int(record.get("pointwise_flip_count", 0))
        valid = int(record.get("attacked_valid_rankings", record.get("total_queries", 0)))
        rows.append({
            "Reasoning effort": effort,
            "Dataset": "TREC-DL-2019",
            "Model": "GPT-OSS-20B",
            "Attack": {"so": "DOH", "sd": "DCH"}[attack],
            "Prompt": "Defense" if prompt == "defense" else "Default",
            "Passages": passages,
            "Requested": requested,
            "Valid attacked": valid,
            "Discarded": requested - valid,
            "Attack success": success,
            "Attack success (%)": record.get("pointwise_flip_percentage", 100 * success / requested if requested else 0),
            "Date": record.get("date", ""),
            "Source": path.relative_to(ROOT).as_posix(),
        })
    rows.sort(key=lambda row: (row["Reasoning effort"], row["Attack"], row["Prompt"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} ablation rows to {output}")
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raise SystemExit(main(args.input_dir, args.output))
