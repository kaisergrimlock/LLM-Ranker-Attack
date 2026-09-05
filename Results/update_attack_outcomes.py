"""Build portable discarded, successful, and failed attack outcomes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "LLM_prompt_attack" / "outputs"
OUTCOME_CSV = PROJECT_ROOT / "Results" / "attack_outcomes.csv"

DATASET_LABELS = {
    "msmarco-passage/trec-dl-2019": "TREC-DL-2019",
    "msmarco-passage/trec-dl-2020": "TREC-DL-2020",
}
SCHEME_LABELS = {
    "pairwise": "Pairwise",
    "setwise": "Setwise",
    "listwise": "Listwise",
}
ATTACK_LABELS = {"so": "DOH", "sd": "DCH", "qi": "Query injection"}
MODEL_ALIASES = {
    "qwen/qwen3-4b": "Qwen3-4B",
    "qwen/qwen3-8b": "Qwen3-8B",
    "qwen/qwen3-32b": "Qwen3-32B",
    "qwen.qwen3-32b-v1:0": "Qwen3-32B",
    "openai/gpt-oss-20b": "GPT-OSS-20B",
    "openai.gpt-oss-20b-1:0": "GPT-OSS-20B",
    "meta.llama3-8b-instruct-v1:0": "Llama-3-8B",
    "meta.llama3-70b-instruct-v1:0": "Llama3 70B",
    "meta.llama3-3-70b-instruct-v1:0": "Llama3 70B",
}
FIELDNAMES = (
    "Dataset",
    "Model",
    "Paradigm",
    "Attack",
    "Prompt",
    "Requested",
    "Valid attacked",
    "Discarded",
    "Attack success",
    "Valid attack failure",
    "Discarded (%)",
    "Attack success (%)",
    "Valid attack failure (%)",
    "Date",
    "Source",
    "Line",
)
KEY_FIELDS = ("Dataset", "Model", "Paradigm", "Attack", "Prompt")


def _display_model(model_name: str) -> str:
    normalized = model_name.strip().lower()
    return MODEL_ALIASES.get(normalized, model_name.rsplit("/", maxsplit=1)[-1])


def _success_count(record: dict[str, Any]) -> int:
    scheme = record["ranking_scheme"]
    if scheme == "pairwise":
        return int(record.get("flipped_count", 0))
    if scheme == "setwise":
        return int(record.get("attack_success_count", 0))
    return int(record.get("attack_top_position_count", 0))


def _percentage(numerator: int, denominator: int) -> float:
    return 100 * numerator / denominator if denominator else 0.0


def _key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def _priority(row: dict[str, Any]) -> tuple[int, int, str, str, int]:
    return (
        int(row["Requested"]),
        int(row["Valid attacked"]),
        str(row["Date"]),
        str(row["Source"]),
        int(row["Line"]),
    )


def _outcome_from_record(
    record: dict[str, Any], source: Path, line_number: int
) -> dict[str, Any] | None:
    scheme = record.get("ranking_scheme")
    attack = record.get("attack_type")
    dataset = record.get("dataset_name")
    if (
        scheme not in SCHEME_LABELS
        or attack not in ATTACK_LABELS
        or dataset not in DATASET_LABELS
        or record.get("attack_position") != "back"
    ):
        return None

    requested = int(
        record.get("original_total_rankings", record.get("total_queries", 0))
    )
    valid_attacked = int(
        record.get("attacked_valid_rankings", record.get("total_queries", 0))
    )
    success = _success_count(record)
    if requested <= 0 or valid_attacked < 0 or success < 0 or success > valid_attacked:
        return None

    discarded = requested - valid_attacked
    failure = valid_attacked - success
    prompt_mode = record.get("prompt_mode", "standard")
    prompt = "Defense" if prompt_mode == "defense" else "Default"
    return {
        "Dataset": DATASET_LABELS[dataset],
        "Model": _display_model(str(record.get("model_name", "Unknown"))),
        "Paradigm": SCHEME_LABELS[scheme],
        "Attack": ATTACK_LABELS[attack],
        "Prompt": prompt,
        "Requested": requested,
        "Valid attacked": valid_attacked,
        "Discarded": discarded,
        "Attack success": success,
        "Valid attack failure": failure,
        "Discarded (%)": _percentage(discarded, requested),
        "Attack success (%)": _percentage(success, requested),
        "Valid attack failure (%)": _percentage(failure, requested),
        "Date": str(record.get("date", "")),
        "Source": source.relative_to(PROJECT_ROOT).as_posix(),
        "Line": line_number,
    }


def _existing_rows() -> dict[tuple[str, ...], dict[str, Any]]:
    if not OUTCOME_CSV.exists():
        return {}
    with OUTCOME_CSV.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        loaded: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in rows:
            try:
                row["Model"] = _display_model(row["Model"])
                for field in (
                    "Requested",
                    "Valid attacked",
                    "Discarded",
                    "Attack success",
                    "Valid attack failure",
                    "Line",
                ):
                    row[field] = int(row[field])
                for field in (
                    "Discarded (%)",
                    "Attack success (%)",
                    "Valid attack failure (%)",
                ):
                    row[field] = float(row[field])
                loaded[_key(row)] = row
            except (KeyError, TypeError, ValueError):
                continue
    return loaded


def _scan_results() -> tuple[dict[tuple[str, ...], dict[str, Any]], int]:
    selected: dict[tuple[str, ...], dict[str, Any]] = {}
    scanned = 0
    # Experiment metadata identifies results, regardless of filename or subfolder.
    for path in sorted(OUTPUT_DIR.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                candidate = _outcome_from_record(record, path, line_number)
                if candidate is None:
                    continue
                scanned += 1
                key = _key(candidate)
                if key not in selected or _priority(candidate) > _priority(
                    selected[key]
                ):
                    selected[key] = candidate
    return selected, scanned


def _write_rows(rows: dict[tuple[str, ...], dict[str, Any]]) -> None:
    ordered = sorted(
        rows.values(),
        key=lambda row: (
            row["Attack"],
            row["Dataset"],
            row["Paradigm"],
            row["Model"],
            row["Prompt"],
        ),
    )
    with OUTCOME_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in ordered:
            formatted = row.copy()
            for field in (
                "Discarded (%)",
                "Attack success (%)",
                "Valid attack failure (%)",
            ):
                formatted[field] = f"{float(formatted[field]):.6f}"
            writer.writerow(formatted)


def main() -> None:
    """Merge this host's completed runs into the portable outcome CSV.

    Returns
    -------
    None
        Writes ``Results/attack_outcomes.csv``. Existing rows for runs not
        available on this host are retained, allowing local and server runs to
        be combined through Git.
    """
    rows = _existing_rows()
    candidates, scanned = _scan_results()
    updated = 0
    for key, candidate in candidates.items():
        if key not in rows or _priority(candidate) > _priority(rows[key]):
            rows[key] = candidate
            updated += 1
    _write_rows(rows)
    print(f"Scanned {scanned} compatible result records.")
    print(f"Updated {updated} outcome rows; retained {len(rows)} total rows.")
    print(f"Wrote {OUTCOME_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
