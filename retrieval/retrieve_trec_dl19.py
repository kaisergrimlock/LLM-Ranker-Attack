#!/usr/bin/env python3
"""Retrieve a depth-1000 BM25 run for the TREC DL 2019 passage topics."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "LLM_re_ranker"
    / "run.msmarco-v1-passage.bm25-default.dl19.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve the top passages for every TREC DL 2019 passage query "
            "and write a six-column TREC run."
        )
    )
    parser.add_argument(
        "--index",
        default="msmarco-v1-passage",
        help="Pyserini prebuilt index name or path to a local Lucene index",
    )
    parser.add_argument(
        "--topics",
        default="dl19-passage",
        help="Pyserini topic identifier when --topic-source=pyserini",
    )
    parser.add_argument(
        "--topic-source",
        choices=("ir-datasets", "pyserini"),
        default="ir-datasets",
        help=(
            "Load queries from the local ir_datasets cache by default; "
            "the pyserini source may download topics from GitHub"
        ),
    )
    parser.add_argument(
        "--ir-dataset-name",
        default="msmarco-passage/trec-dl-2019",
        help="ir_datasets dataset used when --topic-source=ir-datasets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"TREC run destination (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--depth", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--run-tag", default="Pyserini-BM25")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output run",
    )
    parser.add_argument(
        "--allow-other-java",
        action="store_true",
        help="Skip this repository's Java 21 requirement check",
    )
    args = parser.parse_args()

    if args.depth < 1:
        parser.error("--depth must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_tag):
        parser.error("--run-tag may contain only letters, digits, '.', '_' and '-'")
    return args


def require_java_21() -> None:
    try:
        result = subprocess.run(
            ["java", "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Java was not found; install and select Java 21") from exc

    version_output = (result.stderr or result.stdout).strip()
    match = re.search(r'version "(?:1\.)?(\d+)', version_output)
    if result.returncode != 0 or match is None:
        raise RuntimeError(f"Could not determine the Java version: {version_output}")
    if int(match.group(1)) != 21:
        raise RuntimeError(
            f"Pyserini retrieval requires Java 21; active Java is {match.group(1)}"
        )
    print(version_output.splitlines()[0])


def topic_text(topic: Any) -> str:
    if isinstance(topic, str):
        return topic
    if isinstance(topic, Mapping):
        for field in ("title", "query", "text", "description"):
            value = topic.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError(f"Cannot extract query text from topic: {topic!r}")


def qid_sort_key(qid: str) -> tuple[int, int | str]:
    return (0, int(qid)) if qid.isdigit() else (1, qid)


def queries_from_records(records: Sequence[Any]) -> dict[str, str]:
    queries: dict[str, str] = {}
    for record in records:
        qid = str(record.query_id)
        text = str(record.text).strip()
        if not text:
            raise ValueError(f"Query {qid!r} has no text")
        if qid in queries:
            raise ValueError(f"Duplicate query ID: {qid}")
        queries[qid] = text
    if not queries:
        raise RuntimeError("No queries were loaded")
    return queries


def load_queries(args: argparse.Namespace) -> dict[str, str]:
    if args.topic_source == "ir-datasets":
        try:
            import ir_datasets
        except ImportError as exc:
            raise RuntimeError(
                "ir_datasets is required for the local topic source"
            ) from exc
        print(f"Loading local queries from ir_datasets: {args.ir_dataset_name}")
        dataset = ir_datasets.load(args.ir_dataset_name)
        return queries_from_records(list(dataset.queries_iter()))

    from pyserini.search import get_topics

    print(f"Loading Pyserini topics: {args.topics}")
    raw_topics = get_topics(args.topics)
    if not raw_topics:
        raise RuntimeError(f"No topics found for {args.topics!r}")
    return {
        str(qid): topic_text(topic) for qid, topic in raw_topics.items()
    }


def write_trec_run(
    output: Path,
    qids: Sequence[str],
    results: Mapping[str, Sequence[Any]],
    *,
    depth: int,
    run_tag: str,
    overwrite: bool,
) -> int:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite")

    short_queries = {
        qid: len(results.get(qid, ()))
        for qid in qids
        if len(results.get(qid, ())) < depth
    }
    if short_queries:
        examples = ", ".join(
            f"{qid}={count}" for qid, count in list(short_queries.items())[:5]
        )
        raise RuntimeError(
            f"Expected {depth} hits for every query; short result sets: {examples}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    lines_written = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as stream:
            tmp_path = Path(stream.name)
            for qid in sorted(qids, key=qid_sort_key):
                hits = results[qid]
                seen_docids: set[str] = set()
                for rank, hit in enumerate(hits[:depth], start=1):
                    docid = str(hit.docid)
                    if docid in seen_docids:
                        raise RuntimeError(f"Duplicate docid {docid!r} for query {qid}")
                    seen_docids.add(docid)
                    stream.write(
                        f"{qid} Q0 {docid} {rank} {float(hit.score):.8f} {run_tag}\n"
                    )
                    lines_written += 1
        os.replace(tmp_path, output)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    return lines_written


def main() -> int:
    args = parse_args()
    if not args.allow_other_java:
        require_java_21()

    try:
        from pyserini.search.lucene import LuceneSearcher
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Pyserini or one of its runtime dependencies is missing: {exc.name}"
        ) from exc

    query_by_qid = load_queries(args)

    index_path = Path(args.index).expanduser()
    if index_path.exists():
        print(f"Loading local Lucene index: {index_path.resolve()}")
        searcher = LuceneSearcher(str(index_path.resolve()))
    else:
        print(f"Loading Pyserini prebuilt index: {args.index}")
        searcher = LuceneSearcher.from_prebuilt_index(args.index)

    searcher.set_bm25(args.k1, args.b)
    qids = sorted(query_by_qid, key=qid_sort_key)
    queries = [query_by_qid[qid] for qid in qids]

    print(
        f"Retrieving {args.depth} passages for {len(qids)} queries "
        f"with BM25(k1={args.k1}, b={args.b}) using {args.threads} threads"
    )
    results = searcher.batch_search(
        queries,
        qids,
        k=args.depth,
        threads=args.threads,
    )
    lines_written = write_trec_run(
        args.output,
        qids,
        results,
        depth=args.depth,
        run_tag=args.run_tag,
        overwrite=args.overwrite,
    )

    expected = len(qids) * args.depth
    if lines_written != expected:
        raise RuntimeError(f"Wrote {lines_written} rows; expected {expected}")
    print(f"Wrote {lines_written} rows to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
