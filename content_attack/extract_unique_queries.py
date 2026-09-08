"""Extract distinct query texts from the cached TREC DL 2019 and 2020 topics.

Run ``python content_attack/extract_unique_queries.py`` from the repository root.
Includes all topics, whether or not they have relevance judgments. Deduplication
uses exact query text, preserving case, whitespace, query IDs, and source years.
Only the Python standard library is required.
"""

import argparse
import csv
import gzip
from pathlib import Path


def _extract(data_dir):
    queries = {}
    counts = {}
    for year in (2019, 2020):
        path = data_dir / f"trec-dl-{year}" / "queries.tsv"
        if not path.is_file():
            path = path.with_suffix(".tsv.gz")
        opener = gzip.open if path.suffix == ".gz" else open
        counts[year] = 0
        with opener(path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                line = line.rstrip("\r\n")
                if not line:
                    continue
                fields = line.split("\t", 1)
                if len(fields) != 2 or not fields[0] or not fields[1]:
                    raise ValueError(f"Invalid query at {path}:{line_number}")
                query_id, query = fields
                record = queries.setdefault(query, {"query_ids": [], "years": []})
                if query_id not in record["query_ids"]:
                    record["query_ids"].append(query_id)
                if year not in record["years"]:
                    record["years"].append(year)
                counts[year] += 1
    return queries, counts


def _main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=folder.parent / "ir_datasets" / "msmarco-passage",
        help="Directory containing trec-dl-2019 and trec-dl-2020 caches.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=folder / "unique_queries.tsv",
        help="Destination TSV (default: beside this script).",
    )
    args = parser.parse_args()
    queries, counts = _extract(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=["query", "query_ids", "years"], delimiter="\t"
        )
        writer.writeheader()
        for query, record in queries.items():
            writer.writerow(
                {
                    "query": query,
                    "query_ids": ",".join(record["query_ids"]),
                    "years": ",".join(str(year) for year in record["years"]),
                }
            )
    print(f"Read {counts[2019]} queries from 2019 and {counts[2020]} from 2020.")
    print(f"Wrote {len(queries)} unique query texts to {args.output}")


if __name__ == "__main__":
    _main()
