"""Run paired McNemar tests directly on evaluation checkpoint files.

Checkpoint files preserve one record per query under ``clean`` and ``attacked``.
This script currently tests pointwise Yes/No labels; ranking checkpoints are
reported as skipped because their labels require qrels and attack-position
semantics that are not stored as binary outcomes in the checkpoint itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) / (2**n) for i in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail)


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def _metadata(path: Path, fingerprint: dict[str, Any]) -> dict[str, str]:
    dataset = str(fingerprint.get("dataset_name", "unknown"))
    year = re.search(r"20\d\d", dataset)
    model = str(fingerprint.get("model_name", "unknown"))
    model = {
        "openai.gpt-oss-20b-1:0": "GPT-OSS-20B",
        "meta.llama3-8b-instruct-v1:0": "Llama3-8B",
        "meta.llama3-70b-instruct-v1:0": "Llama3-70B",
    }.get(model, model)
    attack = {"so": "DOH", "sd": "DCH", "qi": "QI", "key_injection": "Keyword injection"}.get(
        str(fingerprint.get("attack_type", "unknown")),
        str(fingerprint.get("attack_type", "unknown")),
    )
    prompt_mode = str(fingerprint.get("prompt_mode", "unknown"))
    prompt = "Defense" if prompt_mode.startswith("defense") else "Default"
    setting = fingerprint.get("gpt_oss_reasoning_effort") or fingerprint.get("qwen_thinking_mode") or "default"
    return {
        "model": model,
        "dataset": f"TREC-DL-{year.group(0)}" if year else dataset,
        "paradigm": str(fingerprint.get("paradigm", "unknown")).capitalize(),
        "attack": attack,
        "prompt": prompt,
        "setting": str(setting),
    }


def _pointwise_row(path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    fingerprint = payload.get("fingerprint", {})
    metadata = _metadata(path, fingerprint)
    if metadata["paradigm"].casefold() != "pointwise":
        return None
    phases = payload.get("phases", {})
    clean = phases.get("clean", {})
    attacked = phases.get("attacked", {})
    ids = sorted(set(clean) & set(attacked), key=int)
    pairs = [
        (clean[index].get("label"), attacked[index].get("label"))
        for index in ids
    ]
    pairs = [(left, right) for left, right in pairs if left in {"Yes", "No"} and right in {"Yes", "No"}]
    b = sum(left == "No" and right == "Yes" for left, right in pairs)
    c = sum(left == "Yes" and right == "No" for left, right in pairs)
    return {
        **metadata,
        "paired": len(pairs),
        "clean_no_attacked_yes": b,
        "clean_yes_attacked_no": c,
        "p_value_exact": exact_mcnemar_p(b, c),
        "direction": "attack_increase" if b > c else "attack_decrease" if c > b else "tie",
        "source": path.as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("Results/ablation_server"))
    parser.add_argument("--output", type=Path, default=Path("Results/mcnemar_checkpoints.csv"))
    parser.add_argument("--min-paired", type=int, default=1)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    skipped = 0
    for path in sorted(args.checkpoints_dir.rglob("*.checkpoint.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            phases = payload.get("phases", {})
            if not payload.get("complete") or not phases.get("clean") or not phases.get("attacked"):
                skipped += 1
                continue
            row = _pointwise_row(path, payload)
            if row is None or row["paired"] < args.min_paired:
                skipped += 1
                continue
            rows.append(row)
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
            skipped += 1

    adjusted = holm([float(row["p_value_exact"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_value_holm"] = value
        row["significant_holm_0.05"] = value < 0.05

    fields = [
        "model", "dataset", "paradigm", "attack", "prompt", "setting", "paired",
        "clean_no_attacked_yes", "clean_yes_attacked_no", "p_value_exact",
        "p_value_holm", "significant_holm_0.05", "direction", "source",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} checkpoint McNemar rows to {args.output}; skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
