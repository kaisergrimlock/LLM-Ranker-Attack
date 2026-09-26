"""Extract the newest raw attack artifacts selected by the outcome updater.

The output contains one canonical JSONL result file per
dataset/model/paradigm/attack/prompt key, plus any adjacent detailed JSON
file and a manifest describing the selection.  This keeps statistical tests
independent of duplicate or superseded runs in ``LLM_prompt_attack/outputs``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import update_attack_outcomes as outcomes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "Results" / "newest_attack_results"


def _copy_artifact(source: Path, output_dir: Path) -> Path | None:
    """Copy a result artifact while preserving its relative output path."""
    if not source.is_file():
        return None
    relative = source.relative_to(outcomes.OUTPUT_DIR)
    destination = output_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _selected_from_csv(source_contains: str | None) -> dict[tuple[str, ...], dict]:
    """Load the updater's already-selected newest rows without rescanning files."""
    selected: dict[tuple[str, ...], dict] = {}
    if not outcomes.OUTCOME_CSV.exists():
        return selected
    with outcomes.OUTCOME_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = str(row.get("Source", ""))
            if not source or (source_contains and source_contains not in source):
                continue
            key = tuple(str(row.get(field, "")) for field in outcomes.KEY_FIELDS)
            selected[key] = row
    return selected


def main(
    output_dir: Path = DEFAULT_OUTPUT,
    source_contains: str | None = None,
    rescan: bool = False,
) -> int:
    if rescan:
        selected, scanned = outcomes._scan_results()
        if source_contains:
            selected = {
                key: row
                for key, row in selected.items()
                if source_contains in str(row["Source"])
            }
    else:
        selected = _selected_from_csv(source_contains)
        scanned = len(selected)
        if not selected:
            raise SystemExit(
                "No rows loaded from Results/attack_outcomes.csv; "
                "run update_attack_outcomes.py first or pass --rescan."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    fields = [
        *outcomes.KEY_FIELDS,
        "Date",
        "Requested",
        "Valid attacked",
        "Source",
        "Copied result",
        "Copied detail",
        "Line",
    ]
    copied_results: set[Path] = set()
    rows: list[dict[str, object]] = []

    for key, row in sorted(selected.items()):
        source = PROJECT_ROOT / row["Source"]
        copied_result = _copy_artifact(source, output_dir)
        if copied_result is not None:
            copied_results.add(source)

        detail_name = source.stem
        if detail_name.startswith("result_"):
            detail_name = "detail_" + detail_name.removeprefix("result_")
        detail_source = source.with_name(detail_name + ".json")
        copied_detail = ""
        if detail_source.exists():
            copied_detail_path = _copy_artifact(detail_source, output_dir)
            if copied_detail_path is not None:
                copied_detail = str(copied_detail_path.relative_to(PROJECT_ROOT))

        rows.append(
            {
                **dict(zip(outcomes.KEY_FIELDS, key)),
                "Date": row["Date"],
                "Requested": row["Requested"],
                "Valid attacked": row["Valid attacked"],
                "Source": row["Source"],
                "Copied result": (
                    str(copied_result.relative_to(PROJECT_ROOT))
                    if copied_result is not None
                    else ""
                ),
                "Copied detail": copied_detail,
                "Line": row["Line"],
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scanned {scanned} compatible result records.")
    print(f"Selected {len(rows)} newest outcome keys.")
    print(f"Copied {len(copied_results)} unique result files.")
    print(f"Unavailable locally: {sum(not row['Copied result'] for row in rows)}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-contains")
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Rescan raw JSONL files instead of using attack_outcomes.csv.",
    )
    args = parser.parse_args()
    raise SystemExit(main(args.output_dir, args.source_contains, args.rescan))
