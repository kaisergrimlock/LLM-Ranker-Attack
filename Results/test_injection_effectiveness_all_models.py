#!/usr/bin/env python3
"""Cochran's Q and paired McNemar tests for injection methods across models.

The unit is one query.  Only non-ablation, default-prompt attack runs are
used, so this compares injection methods rather than mixing attack type with
defense type.  A query is successful for a method if at least one complete
query/passage-set record for that method is successful.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
import re
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

METHODS = ("dch", "doh", "keyword injection", "query injection")
METHOD_LABELS = {
    "dch": "DCH",
    "doh": "DOH",
    "keyword injection": "Keyword injection",
    "query injection": "Query injection",
}


def chi_square3_sf(value: float) -> float:
    """Survival function for chi-square with 3 degrees of freedom."""
    x = max(0.0, value / 2.0)
    return math.erfc(math.sqrt(x)) + 2.0 * math.sqrt(x / math.pi) * math.exp(-x)


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def cochran_q(matrix: list[list[int]]) -> tuple[float, float]:
    n = len(matrix)
    k = len(METHODS)
    if n == 0:
        return 0.0, 1.0
    column_totals = [sum(row[j] for row in matrix) for j in range(k)]
    row_totals = [sum(row) for row in matrix]
    total = sum(column_totals)
    denominator = k * total - sum(value * value for value in row_totals)
    if denominator == 0:
        return 0.0, 1.0
    statistic = (k - 1) * (k * sum(value * value for value in column_totals) - total * total) / denominator
    return statistic, chi_square3_sf(statistic)


def is_ablation(source: str) -> bool:
    lowered = source.casefold()
    return "ablation" in lowered or "thinking_" in lowered or "reasoning_" in lowered


def record_key(record: dict, paradigm: str):
    if paradigm == "Pointwise":
        pids = (str(record.get("doc_id", "")),)
    elif paradigm == "Pairwise":
        pids = tuple(sorted(str(record.get(key, "")) for key in ("doc1_id", "doc2_id")))
    else:
        pids = tuple(sorted(str(pid) for pid in (record.get("doc_ids") or [])))
    query = " ".join(str(record.get("query", "")).split()).casefold()
    return query, pids


def inferred_attack(source: str) -> str | None:
    """Infer the attack encoded by a result filename for source validation."""
    name = Path(source).name.casefold()
    if "query-injection" in name or "query_injection" in name or "_qi_" in name:
        return "query injection"
    if "key-injection" in name or "key_injection" in name:
        return "keyword injection"
    if re.search(r"(?:^|[_-])(?:dch|sd)(?:[_-]|$)", name):
        return "dch"
    if re.search(r"(?:^|[_-])(?:doh|so)(?:[_-]|$)", name):
        return "doh"
    return None


def source_outcomes(source: dict, detail: Path, missing_as_failure: bool = False,
                    deduplicate_repeats: bool = False) -> dict[tuple[str, tuple[str, ...]], bool]:
    payload = json.loads(detail.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("results", []))
    grouped = defaultdict(dict)
    occurrence_counts = defaultdict(int)
    seen_phase_keys = set()
    for record in payload:
        if record.get("phase") in {"original", "attacked"}:
            phase = record["phase"]
            base_key = record_key(record, source["Paradigm"])
            if deduplicate_repeats:
                occurrence = 0
                if (base_key, phase) in seen_phase_keys:
                    continue
                seen_phase_keys.add((base_key, phase))
            else:
                occurrence = occurrence_counts[(phase, base_key)]
                occurrence_counts[(phase, base_key)] += 1
            grouped[(base_key, occurrence)][phase] = record
    outcomes = {}
    for (base_key, occurrence), phases in grouped.items():
        if "original" not in phases:
            continue
        if "attacked" not in phases:
            if missing_as_failure:
                outcomes[(*base_key, occurrence)] = False
            continue
        original, attacked = phases["original"], phases["attacked"]
        # Query-level repairs may preserve an invalid model response as an
        # explicit record.  Keep it in the denominator and count it as an
        # attack failure rather than silently discarding it.
        if attacked.get("invalid_as_failure"):
            outcomes[(*base_key, occurrence)] = False
            continue
        target = attack_target_label(original, attacked, source["Paradigm"])
        if source["Paradigm"] == "Listwise":
            # Match update_attack_outcomes.py: listwise success means the
            # attacked passage reaches rank 1, not merely that it moves up.
            ranking = ranking_from_record(attacked)
            value = (
                "true" if target and ranking and ranking[0] == target
                else "false" if target and ranking and target in ranking
                else ""
            )
        else:
            value = success_value(original, attacked, source["Paradigm"], target)
        if value in {"true", "false"}:
            outcomes[(*base_key, occurrence)] = value == "true"
        elif missing_as_failure:
            outcomes[(*base_key, occurrence)] = False
    return outcomes


def official_query_map(dataset: str) -> dict[str, str]:
    year = dataset.rsplit("-", 1)[-1]
    path = ROOT / "ir_datasets" / "msmarco-passage" / f"trec-dl-{year}" / "queries.tsv"
    opener = open
    if not path.exists():
        path = Path(str(path) + ".gz")
        opener = gzip.open
    result = {}
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            qid, separator, text = line.rstrip("\n").partition("\t")
            if separator:
                result[" ".join(text.split()).casefold()] = str(qid)
    return result


def load_query_outcomes(rows: list[dict], unit: str = "query", missing_as_failure: bool = False,
                        deduplicate_repeats: bool = False):
    # Select the source with the largest number of complete paired outcomes
    # for each model/dataset/paradigm/method. This avoids an incomplete stale
    # source masking a repaired source.
    selected = {}
    missing_sources = []
    for source in rows:
        if source["Prompt"].strip().casefold() != "default":
            continue
        if is_ablation(source["Source"]):
            continue
        method = source["Attack"].strip().casefold()
        if method not in METHODS:
            continue
        source_attack = inferred_attack(source["Source"])
        if source_attack is not None and source_attack != method:
            # The CSV can contain stale/mislabeled duplicate rows. Never use a
            # detail file for a different attack than the row claims.
            missing_sources.append(source["Source"])
            continue
        detail = next((path for path in detail_candidates(source["Source"]) if path.is_file()), None)
        if detail is None:
            missing_sources.append(source["Source"])
            continue
        try:
            outcomes = source_outcomes(source, detail, missing_as_failure, deduplicate_repeats)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError):
            missing_sources.append(source["Source"])
            continue
        group = (source["Model"], source["Dataset"], source["Paradigm"], method)
        candidate = (len(outcomes), detail.stat().st_mtime_ns, source, detail, outcomes)
        old = selected.get(group)
        if old is None or candidate[:2] > old[:2]:
            selected[group] = candidate

    # Collapse outcomes to the selected analysis unit. Query-level analysis
    # uses official query IDs; query-passage uses the exact passage-set key.
    query_maps = {}
    datasets = {group[1] for group in selected}
    for dataset in datasets:
        try:
            query_maps[dataset] = official_query_map(dataset)
        except (FileNotFoundError, OSError):
            query_maps[dataset] = {}
    query_outcomes = defaultdict(dict)
    for (model, dataset, paradigm, method), (_, _, _, _, outcomes) in selected.items():
        grouped = defaultdict(list)
        if unit == "query-passage":
            for key, value in outcomes.items():
                query_outcomes[(model, dataset, paradigm, key)][method] = value
        else:
            for (query, _pids, _occurrence), value in outcomes.items():
                grouped[query_maps[dataset].get(query, query)].append(value)
            for query, values in grouped.items():
                query_outcomes[(model, dataset, paradigm, query)][method] = any(values)
    return query_outcomes, selected, missing_sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "Results/attack_outcomes.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "Results/injection_effectiveness_all_models.csv")
    parser.add_argument("--unit", choices=["query", "query-passage"], default="query",
                        help="Pair by query or by exact query-passage set.")
    parser.add_argument("--missing-as-failure", action="store_true",
                        help="Treat missing/invalid method outcomes as attack failures.")
    parser.add_argument("--deduplicate-repeats", action="store_true",
                        help="Keep only the first call for each query/passage set per source.")
    args = parser.parse_args()
    if args.output == ROOT / "Results/injection_effectiveness_all_models.csv":
        suffix = ""
        if args.unit == "query-passage":
            suffix += "_query_passage"
        if args.missing_as_failure:
            suffix += "_missing_as_failure"
        if args.deduplicate_repeats:
            suffix += "_deduplicated"
        if suffix:
            args.output = ROOT / f"Results/injection_effectiveness_all_models{suffix}.csv"

    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    outcomes, selected, missing = load_query_outcomes(
        rows, args.unit, args.missing_as_failure, args.deduplicate_repeats
    )

    strata = defaultdict(list)
    for key, methods in outcomes.items():
        model, dataset, paradigm = key[:3]
        if args.missing_as_failure or all(method in methods for method in METHODS):
            strata[(model, dataset, paradigm)].append(
                [int(methods.get(method, False)) for method in METHODS]
            )

    output_rows = []
    for (model, dataset, paradigm), matrix in sorted(strata.items()):
        statistic, p_value = cochran_q(matrix)
        rates = [sum(row[i] for row in matrix) / len(matrix) for i in range(len(METHODS))]
        output_rows.append({
            "test": "cochran_q", "model": model, "dataset": dataset,
            "paradigm": paradigm, "method_a": "|".join(METHOD_LABELS[m] for m in METHODS),
            "method_b": "", "paired_queries": len(matrix),
            "rate_a": "|".join(f"{rate:.6f}" for rate in rates), "rate_b": "",
            "discordant_a_success_b_failure": "", "discordant_a_failure_b_success": "",
            "statistic": statistic, "p_value": p_value,
            "p_value_holm": "", "significant_holm_0.05": "",
            "missing_as_failure": args.missing_as_failure,
        })
        pair_rows = []
        for left, right in itertools.combinations(range(len(METHODS)), 2):
            b = sum(row[left] and not row[right] for row in matrix)
            c = sum(not row[left] and row[right] for row in matrix)
            pair_rows.append({
                "test": "mcnemar", "model": model, "dataset": dataset,
                "paradigm": paradigm, "method_a": METHOD_LABELS[METHODS[left]],
                "method_b": METHOD_LABELS[METHODS[right]], "paired_queries": len(matrix),
                "rate_a": f"{sum(row[left] for row in matrix) / len(matrix):.6f}",
                "rate_b": f"{sum(row[right] for row in matrix) / len(matrix):.6f}",
                "discordant_a_success_b_failure": b,
                "discordant_a_failure_b_success": c, "statistic": "",
                "p_value": exact_mcnemar_p(b, c), "p_value_holm": "",
                "significant_holm_0.05": "",
                "missing_as_failure": args.missing_as_failure,
            })
        adjusted = holm([float(row["p_value"]) for row in pair_rows])
        for row, adjusted_p in zip(pair_rows, adjusted):
            row["p_value_holm"] = adjusted_p
            row["significant_holm_0.05"] = adjusted_p < 0.05
        output_rows.extend(pair_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["test", "model", "dataset", "paradigm", "method_a", "method_b",
              "paired_queries", "rate_a", "rate_b", "discordant_a_success_b_failure",
              "discordant_a_failure_b_success", "statistic", "p_value",
              "p_value_holm", "significant_holm_0.05", "missing_as_failure"]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({
        "selected_detail_sources": len(selected),
        "complete_paired_strata": len(strata),
        "tests_written": len(output_rows),
        "missing_or_unreadable_sources": len(missing),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
