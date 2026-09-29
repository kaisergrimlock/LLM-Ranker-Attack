"""Compare attack success rates with paired Cochran-Q/McNemar tests.

The script uses the newest complete detailed JSON for each model, dataset,
paradigm, attack, and prompt.  Tests are performed within each aligned
model/dataset/paradigm group; pooling unrelated query sets would invalidate
the paired tests.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

from mcnemar_all import _is_complete, _key, _metadata, exact_mcnemar_p, holm

try:
    from scipy.stats import chi2
except ImportError as error:  # pragma: no cover - environment dependent
    raise RuntimeError("scipy is required for Cochran's Q tests") from error


ATTACK_ALIASES = {
    "so": "DOH",
    "doh": "DOH",
    "doh_back": "DOH",
    "sd": "DCH",
    "dch": "DCH",
    "dch_back": "DCH",
    "qi": "Query injection",
    "qi_back": "Query injection",
    "query_injection": "Query injection",
    "key_injection": "Keyword injection",
    "keyword_injection": "Keyword injection",
}
ATTACK_ORDER = ("DOH", "DCH", "Query injection", "Keyword injection")


def _attack_name(value: str) -> str | None:
    key = re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")
    return ATTACK_ALIASES.get(key)


def _success_by_instance(path: Path) -> tuple[dict[tuple[Any, ...], bool], dict[str, str]]:
    """Return attack-success indicators aligned by the evaluator's instance key."""
    metadata = _metadata(path)
    paradigm = metadata["paradigm"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    clean = {}
    attacked = {}
    for record in payload:
        if not isinstance(record, dict) or record.get("phase") not in {"original", "attacked"}:
            continue
        key = _key(record, paradigm)
        if key is not None:
            (clean if record["phase"] == "original" else attacked)[key] = record

    result: dict[tuple[Any, ...], bool] = {}
    for key in clean.keys() & attacked.keys():
        original, attack = clean[key], attacked[key]
        if paradigm == "pointwise":
            left = str(original.get("label", "")).strip().casefold()
            right = str(attack.get("label", "")).strip().casefold()
            if left not in {"yes", "no"} or right not in {"yes", "no"}:
                continue
            value = left == "no" and right == "yes"
        elif paradigm == "pairwise":
            left = str(original.get("label", "")).strip().upper()
            right = str(attack.get("label", "")).strip().upper()
            if len(left) != 1 or len(right) != 1 or not left.isalpha() or not right.isalpha():
                continue
            value = left != right
        elif paradigm == "setwise":
            label = str(attack.get("label", "")).strip().upper()
            target = str(attack.get("attack_label", "")).strip().upper()
            if not label or not target:
                continue
            value = label == target
        elif paradigm == "listwise":
            labels = attack.get("labels")
            target = str(attack.get("attack_label", "")).strip().upper()
            if not isinstance(labels, list) or not labels or not target:
                continue
            value = str(labels[0]).strip().upper() == target
        else:
            continue
        result[key] = value
    return result, metadata


def _cochran_q(columns: list[dict[tuple[Any, ...], bool]], attacks: list[str]) -> tuple[int, float]:
    keys = set.intersection(*(set(column) for column in columns))
    n = len(keys)
    k = len(attacks)
    if n == 0 or k < 3:
        return n, 1.0
    rows = [[int(column[key]) for column in columns] for key in keys]
    row_totals = [sum(row) for row in rows]
    col_totals = [sum(row[j] for row in rows) for j in range(k)]
    total = sum(row_totals)
    denominator = k * total - sum(value * value for value in row_totals)
    if denominator == 0:
        return n, 1.0
    numerator = (k - 1) * (k * sum(value * value for value in col_totals) - total * total)
    statistic = max(0.0, numerator / denominator)
    return n, float(chi2.sf(statistic, k - 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details-dir", type=Path, default=Path("LLM_prompt_attack/outputs"))
    parser.add_argument("--output", type=Path, default=Path("Results/attack_significance.csv"))
    parser.add_argument("--prompts", nargs="+", default=["standard"], choices=["standard", "defense", "defense_qi", "filter_qi"])
    parser.add_argument("--attacks", nargs="+", default=list(ATTACK_ORDER), choices=list(ATTACK_ORDER))
    args = parser.parse_args()

    newest: dict[tuple[str, ...], Path] = {}
    for path in sorted(args.details_dir.rglob("detail_*.json")):
        metadata = _metadata(path)
        attack = _attack_name(metadata["attack"])
        if attack not in args.attacks or metadata["prompt"] not in args.prompts or not _is_complete(path):
            continue
        group = (metadata["model"], metadata["dataset"], metadata["paradigm"], attack, metadata["prompt"])
        old = newest.get(group)
        if old is None or (path.stat().st_mtime_ns, path.as_posix()) > (old.stat().st_mtime_ns, old.as_posix()):
            newest[group] = path

    records = {}
    for group, path in newest.items():
        indicators, metadata = _success_by_instance(path)
        records[group] = (indicators, metadata, path)

    omnibus = []
    pairwise = []
    grouped: dict[tuple[str, str, str, str], dict[str, tuple[dict, dict, Path]]] = {}
    for (model, dataset, paradigm, attack, prompt), value in records.items():
        grouped.setdefault((model, dataset, paradigm, prompt), {})[attack] = value

    for (model, dataset, paradigm, prompt), attacks in grouped.items():
        available = [attack for attack in args.attacks if attack in attacks]
        if len(available) < 2:
            continue
        columns = [attacks[attack][0] for attack in available]
        if len(available) >= 3:
            n, p_q = _cochran_q(columns, available)
            omnibus.append({
                "test": "cochran_q", "model": model, "dataset": dataset,
                "paradigm": paradigm, "prompt": prompt, "attack_a": "|".join(available),
                "attack_b": "", "paired": n, "discordant_b": "", "discordant_c": "",
                "p_value": p_q, "direction": "", "source_a": ";".join(attacks[a][2].as_posix() for a in available), "source_b": "",
            })
        for left, right in itertools.combinations(available, 2):
            a, b = attacks[left][0], attacks[right][0]
            keys = set(a) & set(b)
            discordant_b = sum(a[key] and not b[key] for key in keys)
            discordant_c = sum(not a[key] and b[key] for key in keys)
            pairwise.append({
                "test": "mcnemar", "model": model, "dataset": dataset,
                "paradigm": paradigm, "prompt": prompt, "attack_a": left,
                "attack_b": right, "paired": len(keys), "discordant_b": discordant_b,
                "discordant_c": discordant_c, "p_value": exact_mcnemar_p(discordant_b, discordant_c),
                "direction": "a_higher" if discordant_b > discordant_c else "b_higher" if discordant_c > discordant_b else "tie",
                "source_a": attacks[left][2].as_posix(), "source_b": attacks[right][2].as_posix(),
            })

    rows = omnibus + pairwise
    adjusted = holm([float(row["p_value"]) for row in rows]) if rows else []
    fields = ["test", "model", "dataset", "paradigm", "prompt", "attack_a", "attack_b", "paired", "discordant_b", "discordant_c", "p_value", "p_value_holm", "significant_holm_0.05", "direction", "source_a", "source_b"]
    for row, p in zip(rows, adjusted):
        row["p_value_holm"] = p
        row["significant_holm_0.05"] = p < 0.05
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Selected {len(newest)} newest complete runs; wrote {len(rows)} tests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
