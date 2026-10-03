#!/usr/bin/env python3
"""Paired defence tests within each attack type.

For every model/dataset/paradigm/attack stratum, this script compares the
available prompt conditions on the same query-passage units.  With two
conditions it runs an exact paired McNemar test.  With three or more it runs
Cochran's Q followed by pairwise exact McNemar tests with Holm correction.

The script deliberately reads per-query/per-passage detail JSON files rather
than aggregate ASR summaries.  It excludes ablation/reasoning sources.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Results"))
from build_attack_query_passage_table import (  # noqa: E402
    attack_target_label,
    detail_candidates,
    ranking_from_record,
    success_value,
)
from test_injection_effectiveness_all_models import (  # noqa: E402
    inferred_attack,
    official_query_map,
    record_key,
    source_outcomes,
)

ATTACKS = ("doh", "dch", "query injection", "keyword injection")
ATTACK_LABELS = {
    "doh": "DOH",
    "dch": "DCH",
    "query injection": "Query injection",
    "keyword injection": "Keyword injection",
}


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def chi_square_sf(statistic: float, df: int) -> float:
    try:
        from scipy.stats import chi2
        return float(chi2.sf(statistic, df))
    except ImportError:
        # Dependency-free regularized upper incomplete gamma for the
        # integer/half-integer shapes produced by chi-square distributions.
        x = max(0.0, statistic) / 2.0
        a = df / 2.0
        if x == 0.0:
            return 1.0
        if a.is_integer():
            return min(1.0, math.exp(-x) * sum(x**j / math.factorial(j) for j in range(int(a))))
        result = math.erfc(math.sqrt(x))
        current = 0.5
        while current < a:
            result += math.exp(-x + current * math.log(x) - math.lgamma(current + 1.0))
            current += 1.0
        return min(1.0, result)


def cochran_q(matrix: list[list[int]]) -> tuple[float, float]:
    n = len(matrix)
    k = len(matrix[0]) if matrix else 0
    if n == 0 or k < 3:
        return 0.0, 1.0
    column_totals = [sum(row[j] for row in matrix) for j in range(k)]
    row_totals = [sum(row) for row in matrix]
    total = sum(column_totals)
    denominator = k * total - sum(value * value for value in row_totals)
    if denominator == 0:
        return 0.0, 1.0
    statistic = (k - 1) * (
        k * sum(value * value for value in column_totals) - total * total
    ) / denominator
    return statistic, chi_square_sf(statistic, k - 1)


def is_ablation(source: str) -> bool:
    lowered = source.casefold()
    return "ablation" in lowered or "thinking_" in lowered or "reasoning_" in lowered


def normalize_condition(prompt: str, attack: str) -> str:
    value = " ".join(prompt.split()).strip()
    if value.casefold() in {"defense", "defense qi", "defense qwen"}:
        return "Defense QI" if attack in {"query injection", "keyword injection"} else "Defense"
    return value or "Unknown"


def query_passage_outcomes(
    rows: list[dict],
    missing_as_failure: bool,
    deduplicate_repeats: bool,
) -> tuple[dict, dict, list[str]]:
    """Return outcomes keyed by stratum and condition, plus source metadata."""
    selected = {}
    missing = []
    for source in rows:
        if is_ablation(source.get("Source", "")):
            continue
        attack = source.get("Attack", "").strip().casefold()
        if attack not in ATTACKS:
            continue
        inferred = inferred_attack(source.get("Source", ""))
        if inferred is not None and inferred != attack:
            missing.append(source.get("Source", ""))
            continue
        detail = next((p for p in detail_candidates(source["Source"]) if p.is_file()), None)
        if detail is None:
            missing.append(source["Source"])
            continue
        try:
            values = source_outcomes(source, detail, missing_as_failure, deduplicate_repeats)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            missing.append(source["Source"])
            continue
        # The original GPT-OSS pointwise DOH/DCH detail export was an
        # invalid-response-only artifact, although the validated pointwise
        # run is retained in the earlier low-reasoning output directory and
        # is the source used by the previous McNemar analysis.  Use that
        # source only for these four affected cells; leave all other strata
        # on their canonical detail files.
        if (
            not values
            and source.get("Model") == "GPT-OSS-20B"
            and source.get("Paradigm") == "Pointwise"
            and attack in {"doh", "dch"}
        ):
            source_name = Path(source["Source"]).name
            if source_name.startswith("result_"):
                fallback_name = "detail_" + source_name[7:].replace(".jsonl", ".json")
                fallback = ROOT / "LLM_prompt_attack/outputs/gptoss_pointwise_low_reasoning" / fallback_name
                if fallback.is_file():
                    fallback_values = source_outcomes(source, fallback, missing_as_failure, deduplicate_repeats)
                    if fallback_values:
                        detail, values = fallback, fallback_values
        condition = normalize_condition(source.get("Prompt", ""), attack)
        group = (source["Model"], source["Dataset"], source["Paradigm"], attack, condition)
        candidate = (len(values), detail.stat().st_mtime_ns, source, detail, values)
        if group not in selected or candidate[:2] > selected[group][:2]:
            selected[group] = candidate

    outcomes = defaultdict(dict)
    sources = defaultdict(dict)
    for (model, dataset, paradigm, attack, condition), (_, _, source, detail, values) in selected.items():
        stratum = (model, dataset, paradigm, attack)
        for key, value in values.items():
            outcomes[stratum].setdefault(key, {})[condition] = bool(value)
        sources[stratum][condition] = detail.as_posix()
    return outcomes, sources, missing


def collapse_to_query(outcomes: dict, dataset: str) -> dict:
    mapping = official_query_map(dataset)
    collapsed = defaultdict(dict)
    for key, conditions in outcomes.items():
        query = key[0]
        query_id = mapping.get(query, query)
        for condition, value in conditions.items():
            collapsed[query_id][condition] = collapsed[query_id].get(condition, False) or value
    return collapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "Results/attack_outcomes.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "Results/defense_by_attack_tests.csv")
    parser.add_argument("--unit", choices=("query-passage", "query"), default="query-passage")
    parser.add_argument("--missing-as-failure", action="store_true")
    parser.add_argument("--deduplicate-repeats", action="store_true")
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    outcomes, source_paths, missing = query_passage_outcomes(
        rows, args.missing_as_failure, args.deduplicate_repeats
    )

    result_rows = []
    q_rows = []
    pair_rows = []
    for stratum, units in sorted(outcomes.items()):
        model, dataset, paradigm, attack = stratum
        if args.unit == "query":
            units = collapse_to_query(units, dataset)
        conditions = sorted({condition for values in units.values() for condition in values})
        if len(conditions) < 2:
            continue
        paired = [
            [int(units[key].get(condition, False)) for condition in conditions]
            for key in sorted(units)
            if args.missing_as_failure or all(condition in units[key] for condition in conditions)
        ]
        if not paired:
            continue
        rates = [sum(row[i] for row in paired) / len(paired) for i in range(len(conditions))]
        base = {
            "model": model,
            "dataset": dataset,
            "paradigm": paradigm,
            "attack": ATTACK_LABELS[attack],
            "conditions": " | ".join(conditions),
            "paired_units": len(paired),
            "condition_rates": " | ".join(f"{r:.6f}" for r in rates),
            "statistic": "",
            "p_value": "",
            "p_value_holm": "",
            "significant_holm_0.05": "",
            "condition_a": "",
            "condition_b": "",
            "discordant_a_success_b_failure": "",
            "discordant_a_failure_b_success": "",
            "source_paths": json.dumps(source_paths.get(stratum, {}), sort_keys=True),
            "unit": args.unit,
            "missing_as_failure": args.missing_as_failure,
        }
        if len(conditions) >= 3:
            statistic, p_value = cochran_q(paired)
            row = dict(base)
            row.update({"test": "cochran_q", "statistic": statistic, "p_value": p_value})
            q_rows.append(row)

        for left, right in itertools.combinations(range(len(conditions)), 2):
            b = sum(row[left] and not row[right] for row in paired)
            c = sum(not row[left] and row[right] for row in paired)
            row = dict(base)
            row.update({
                "test": "mcnemar",
                "condition_a": conditions[left],
                "condition_b": conditions[right],
                "statistic": "",
                "p_value": exact_mcnemar_p(b, c),
                "discordant_a_success_b_failure": b,
                "discordant_a_failure_b_success": c,
            })
            pair_rows.append(row)

    if q_rows:
        for row, adjusted in zip(q_rows, holm([float(r["p_value"]) for r in q_rows])):
            row["p_value_holm"] = adjusted
            row["significant_holm_0.05"] = adjusted < 0.05
    if pair_rows:
        # Correct pairwise comparisons within each attack/model/dataset/paradigm
        # stratum, as recommended after a significant Q test.
        groups = defaultdict(list)
        for index, row in enumerate(pair_rows):
            groups[(row["model"], row["dataset"], row["paradigm"], row["attack"])].append(index)
        for indices in groups.values():
            adjusted = holm([float(pair_rows[i]["p_value"]) for i in indices])
            for i, value in zip(indices, adjusted):
                pair_rows[i]["p_value_holm"] = value
                pair_rows[i]["significant_holm_0.05"] = value < 0.05

    output_rows = q_rows + pair_rows
    fields = [
        "test", "model", "dataset", "paradigm", "attack", "conditions", "paired_units",
        "condition_rates", "condition_a", "condition_b", "discordant_a_success_b_failure",
        "discordant_a_failure_b_success", "statistic", "p_value", "p_value_holm",
        "significant_holm_0.05", "unit", "missing_as_failure", "source_paths",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({
        "output": str(args.output),
        "rows_written": len(output_rows),
        "cochran_q_tests": len(q_rows),
        "mcnemar_tests": len(pair_rows),
        "missing_or_unreadable_sources": len(missing),
        "unit": args.unit,
        "missing_as_failure": args.missing_as_failure,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
