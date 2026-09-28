"""Exact McNemar tests for paired pointwise clean/attacked decisions.

The detailed JSON files written by ``pointwise_ranking_attack_openai.py`` contain
one ``original`` and one ``attacked`` record for each query/document instance.
This script compares correctness on those paired instances.  It excludes
invalid labels by default and writes one row per detailed result file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


FILE_RE = re.compile(
    r"detail_(?P<model>.+?)_trec-dl-(?P<year>20\d\d)_pointwise_"
    r"(?P<attack>[^_]+)_(?P<prompt>.+)\.json$"
)


def exact_mcnemar_p(b: int, c: int) -> float:
    """Return the two-sided exact McNemar p-value."""
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, i) / (2**discordant)
        for i in range(min(b, c) + 1)
    )
    return min(1.0, 2.0 * tail)


def _correct(record: dict, *, relevance: int) -> bool | None:
    label = str(record.get("label", "")).strip().casefold()
    if label not in {"yes", "no"}:
        return None
    # Pointwise evaluation samples non-relevant passages by default.  A No
    # prediction is therefore correct for relevance=0; invert for relevance=1.
    predicted_relevant = label == "yes"
    return predicted_relevant == bool(relevance)


def test_file(path: Path, relevance: int) -> dict | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return None
    original = {}
    attacked = {}
    for record in payload:
        if not isinstance(record, dict):
            continue
        key = (record.get("query"), record.get("doc_id"))
        if None in key:
            continue
        target = original if record.get("phase") == "original" else attacked
        if record.get("phase") in {"original", "attacked"}:
            target[key] = record

    a = b = c = d = 0
    paired = 0
    for key in original.keys() & attacked.keys():
        clean = _correct(original[key], relevance=relevance)
        attack = _correct(attacked[key], relevance=relevance)
        if clean is None or attack is None:
            continue
        paired += 1
        if clean and attack:
            a += 1
        elif clean and not attack:
            b += 1
        elif not clean and attack:
            c += 1
        else:
            d += 1

    if paired == 0:
        return None
    match = FILE_RE.search(path.name)
    metadata = match.groupdict() if match else {
        "model": "unknown", "year": "unknown", "attack": "unknown", "prompt": "unknown"
    }
    p_value = exact_mcnemar_p(b, c)
    return {
        "model": metadata["model"],
        "dataset": f"TREC-DL-{metadata['year']}",
        "attack": metadata["attack"],
        "prompt": metadata["prompt"],
        "paired": paired,
        "both_correct": a,
        "clean_correct_attack_wrong": b,
        "clean_wrong_attack_correct": c,
        "both_wrong": d,
        "p_value_exact": p_value,
        "direction": "degradation" if b > c else "improvement" if c > b else "tie",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--details-dir",
        type=Path,
        default=Path("LLM_prompt_attack/outputs"),
        help="Directory recursively containing pointwise detail JSON files.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("Results/mcnemar_pointwise.csv")
    )
    parser.add_argument("--relevance", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.details_dir.rglob("detail_*pointwise*.json")):
        try:
            row = test_file(path, args.relevance)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            print(f"Skipping {path}: {error}")
            continue
        if row is not None:
            row["source"] = path.as_posix()
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "model", "dataset", "attack", "prompt", "paired", "both_correct",
        "clean_correct_attack_wrong", "clean_wrong_attack_correct", "both_wrong",
        "p_value_exact", "direction", "source",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} McNemar rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
