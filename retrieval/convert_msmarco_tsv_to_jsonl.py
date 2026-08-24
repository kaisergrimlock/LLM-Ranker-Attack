#!/usr/bin/env python3
"""Convert MS MARCO V1 passage TSV into sharded Anserini JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_EXPECTED_DOCUMENTS = 8_841_823


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--documents-per-shard", type=int, default=250_000)
    parser.add_argument(
        "--expected-documents",
        type=int,
        default=DEFAULT_EXPECTED_DOCUMENTS,
        help="Expected input rows; pass 0 to disable the count check",
    )
    args = parser.parse_args()
    if args.documents_per_shard < 1:
        parser.error("--documents-per-shard must be positive")
    if args.expected_documents < 0:
        parser.error("--expected-documents cannot be negative")
    return args


def convert_collection(
    input_path: Path,
    output_dir: Path,
    *,
    documents_per_shard: int,
    expected_documents: int,
) -> tuple[int, int]:
    if not input_path.is_file():
        raise FileNotFoundError(f"MS MARCO collection not found: {input_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; "
            "use a new directory or remove a confirmed partial conversion"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    stream = None
    tmp_path: Path | None = None
    shard_path: Path | None = None
    document_count = 0
    shard_count = 0
    try:
        with input_path.open("r", encoding="utf-8", errors="strict") as source:
            for line_number, line in enumerate(source, start=1):
                line = line.rstrip("\r\n")
                try:
                    docid, contents = line.split("\t", 1)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid collection row {line_number}: expected docid<TAB>text"
                    ) from exc
                if not docid:
                    raise ValueError(f"Empty docid on collection row {line_number}")

                if document_count % documents_per_shard == 0:
                    if stream is not None:
                        stream.close()
                        os.replace(tmp_path, shard_path)
                    shard_path = output_dir / f"docs-{shard_count:05d}.jsonl"
                    tmp_path = shard_path.with_suffix(".jsonl.tmp")
                    stream = tmp_path.open("w", encoding="utf-8", newline="\n")
                    shard_count += 1

                json.dump(
                    {"id": docid, "contents": contents},
                    stream,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                stream.write("\n")
                document_count += 1

        if stream is not None:
            stream.close()
            stream = None
            os.replace(tmp_path, shard_path)

        if expected_documents and document_count != expected_documents:
            raise RuntimeError(
                f"Converted {document_count} documents; expected {expected_documents}"
            )
    except Exception:
        if stream is not None:
            stream.close()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise

    marker = output_dir.with_name(output_dir.name + ".complete")
    marker.write_text(
        json.dumps(
            {
                "input": str(input_path.resolve()),
                "documents": document_count,
                "shards": shard_count,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return document_count, shard_count


def main() -> int:
    args = parse_args()
    documents, shards = convert_collection(
        args.input,
        args.output_dir,
        documents_per_shard=args.documents_per_shard,
        expected_documents=args.expected_documents,
    )
    print(f"Converted {documents} documents into {shards} JSONL shards")
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
