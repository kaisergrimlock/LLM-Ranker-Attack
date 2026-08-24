#!/usr/bin/env python3
"""Evaluate clean and attacked TREC runs with graded nDCG@10."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_DATASET = "msmarco-passage/trec-dl-2019"
DEFAULT_CUTOFF = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate graded nDCG@10 for a clean TREC run and, optionally, "
            "compare it with an attacked run."
        )
    )
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--attacked-run", type=Path)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional destination for the complete machine-readable report.",
    )
    args = parser.parse_args()
    if args.cutoff < 1:
        parser.error("--cutoff must be at least 1")
    return args


def read_trec_run(path: Path) -> dict[str, list[str]]:
    """Read a six-column TREC run and return doc IDs ordered by declared rank."""
    rankings: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(
                    f"{path}:{line_number}: expected 6 TREC fields, got {len(fields)}"
                )
            qid, _, docid, rank_text, _, _ = fields
            try:
                rank = int(rank_text)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid rank {rank_text!r}"
                ) from exc
            if rank < 1:
                raise ValueError(f"{path}:{line_number}: rank must be positive")
            if docid in seen[qid]:
                raise ValueError(
                    f"{path}:{line_number}: duplicate document {docid!r} for query {qid}"
                )
            seen[qid].add(docid)
            rankings[qid].append((rank, docid))

    if not rankings:
        raise ValueError(f"TREC run is empty: {path}")

    ordered: dict[str, list[str]] = {}
    for qid, rows in rankings.items():
        rows.sort(key=lambda row: row[0])
        ranks = [rank for rank, _ in rows]
        if len(ranks) != len(set(ranks)):
            raise ValueError(f"{path}: duplicate ranks for query {qid}")
        ordered[qid] = [docid for _, docid in rows]
    return ordered


def load_qrels(dataset_name: str) -> dict[str, dict[str, int]]:
    try:
        import ir_datasets
    except ImportError as exc:
        raise RuntimeError("ir_datasets is required to load relevance judgments") from exc

    dataset = ir_datasets.load(dataset_name)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for record in dataset.qrels_iter():
        qid = str(record.query_id)
        docid = str(record.doc_id)
        relevance = int(record.relevance)
        if docid in qrels[qid] and qrels[qid][docid] != relevance:
            raise ValueError(
                f"Conflicting qrels for query {qid}, document {docid}: "
                f"{qrels[qid][docid]} and {relevance}"
            )
        qrels[qid][docid] = relevance
    if not qrels:
        raise RuntimeError(f"Dataset {dataset_name!r} did not provide qrels")
    return dict(qrels)


def dcg_at_k(docids: Iterable[str], relevance: Mapping[str, int], k: int) -> float:
    score = 0.0
    for rank, docid in enumerate(docids, start=1):
        if rank > k:
            break
        rel = max(0, int(relevance.get(docid, 0)))
        score += (2**rel - 1) / math.log2(rank + 1)
    return score


def ndcg_at_k(docids: Iterable[str], relevance: Mapping[str, int], k: int) -> float:
    ideal_relevances = sorted(
        (max(0, int(rel)) for rel in relevance.values()), reverse=True
    )[:k]
    ideal = sum(
        (2**rel - 1) / math.log2(rank + 1)
        for rank, rel in enumerate(ideal_relevances, start=1)
    )
    return dcg_at_k(docids, relevance, k) / ideal if ideal else 0.0


def evaluate_run(
    run: Mapping[str, list[str]],
    qrels: Mapping[str, Mapping[str, int]],
    k: int,
) -> dict:
    judged_qids = sorted(set(run).intersection(qrels), key=_qid_sort_key)
    per_query = {
        qid: ndcg_at_k(run[qid], qrels[qid], k) for qid in judged_qids
    }
    mean = sum(per_query.values()) / len(per_query) if per_query else 0.0
    return {
        "mean_ndcg": mean,
        "per_query": per_query,
        "run_query_count": len(run),
        "judged_query_count": len(judged_qids),
        "unjudged_run_query_count": len(set(run).difference(qrels)),
    }


def compare_zero_relevance_movement(
    clean: Mapping[str, list[str]],
    attacked: Mapping[str, list[str]],
    qrels: Mapping[str, Mapping[str, int]],
    k: int,
) -> dict[str, int]:
    counts = {"promoted": 0, "demoted": 0, "unchanged": 0}
    for qid in set(clean).intersection(attacked, qrels):
        clean_ranks = {docid: rank for rank, docid in enumerate(clean[qid][:k], 1)}
        attacked_ranks = {
            docid: rank for rank, docid in enumerate(attacked[qid][:k], 1)
        }
        for docid in set(clean_ranks).intersection(attacked_ranks):
            if qrels[qid].get(docid) != 0:
                continue
            if attacked_ranks[docid] < clean_ranks[docid]:
                counts["promoted"] += 1
            elif attacked_ranks[docid] > clean_ranks[docid]:
                counts["demoted"] += 1
            else:
                counts["unchanged"] += 1
    return counts


def _qid_sort_key(qid: str) -> tuple[int, int | str]:
    return (0, int(qid)) if qid.isdigit() else (1, qid)


def build_report(
    clean_run: Mapping[str, list[str]],
    qrels: Mapping[str, Mapping[str, int]],
    k: int,
    attacked_run: Mapping[str, list[str]] | None = None,
) -> dict:
    report = {"cutoff": k, "clean": evaluate_run(clean_run, qrels, k)}
    if attacked_run is not None:
        clean_qids = set(clean_run)
        attacked_qids = set(attacked_run)
        if clean_qids != attacked_qids:
            missing_from_attack = sorted(clean_qids - attacked_qids, key=_qid_sort_key)
            missing_from_clean = sorted(attacked_qids - clean_qids, key=_qid_sort_key)
            raise ValueError(
                "Clean and attacked runs must contain identical query IDs; "
                f"missing from attacked={missing_from_attack[:5]}, "
                f"missing from clean={missing_from_clean[:5]}"
            )
        attacked = evaluate_run(attacked_run, qrels, k)
        common_qids = sorted(
            set(report["clean"]["per_query"]).intersection(attacked["per_query"]),
            key=_qid_sort_key,
        )
        report["attacked"] = attacked
        report["comparison"] = {
            "mean_delta": attacked["mean_ndcg"] - report["clean"]["mean_ndcg"],
            "per_query_delta": {
                qid: attacked["per_query"][qid] - report["clean"]["per_query"][qid]
                for qid in common_qids
            },
            "zero_relevance_movement": compare_zero_relevance_movement(
                clean_run, attacked_run, qrels, k
            ),
        }
    return report


def print_report(report: Mapping, dataset_name: str) -> None:
    k = report["cutoff"]
    clean = report["clean"]
    print(f"Dataset: {dataset_name}")
    print(
        f"Clean nDCG@{k}: {clean['mean_ndcg']:.4f} "
        f"({clean['judged_query_count']} judged queries; "
        f"{clean['unjudged_run_query_count']} unjudged run queries ignored)"
    )
    if "attacked" not in report:
        return

    attacked = report["attacked"]
    comparison = report["comparison"]
    print(
        f"Attacked nDCG@{k}: {attacked['mean_ndcg']:.4f} "
        f"({attacked['judged_query_count']} judged queries; "
        f"{attacked['unjudged_run_query_count']} unjudged run queries ignored)"
    )
    print(f"Mean delta: {comparison['mean_delta']:+.4f}")
    print("Per-query delta:")
    for qid, delta in comparison["per_query_delta"].items():
        clean_score = clean["per_query"][qid]
        attacked_score = attacked["per_query"][qid]
        print(f"  {qid}: {clean_score:.4f} -> {attacked_score:.4f} ({delta:+.4f})")
    movement = comparison["zero_relevance_movement"]
    print(
        "Judged relevance-0 movement within the top "
        f"{k}: {movement['promoted']} promoted, {movement['demoted']} demoted, "
        f"{movement['unchanged']} unchanged"
    )


def main() -> int:
    args = parse_args()
    clean_run = read_trec_run(args.clean_run)
    attacked_run = read_trec_run(args.attacked_run) if args.attacked_run else None
    qrels = load_qrels(args.dataset)
    report = build_report(clean_run, qrels, args.cutoff, attacked_run)
    print_report(report, args.dataset)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"JSON report: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
