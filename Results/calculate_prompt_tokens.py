"""Calculate tokenizer-specific input-token statistics for ranking prompts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "LLM_prompt_attack" / "outputs"
DEFAULT_OUTPUT = PROJECT_ROOT / "Results" / "prompt_token_counts.csv"

ATTACK_LABELS = {"so": "DOH", "sd": "DCH", "qi": "Query injection"}
PARADIGM_LABELS = {
    "pairwise": "Pairwise",
    "setwise": "Setwise",
    "listwise": "Listwise",
}
PROMPT_LABELS = {"standard": "Default", "defense": "Defense"}

# Bedrock IDs and local served names used by this repository share these
# Hugging Face tokenizers. Llama 3 8B and 70B use the same vocabulary.
MODEL_TOKENIZERS = {
    "qwen.qwen3-4b-v1:0": "Qwen/Qwen3-4B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen/qwen3-4b": "Qwen/Qwen3-4B",
    "qwen.qwen3-8b-v1:0": "Qwen/Qwen3-8B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen/qwen3-8b": "Qwen/Qwen3-8B",
    "qwen.qwen3-32b-v1:0": "Qwen/Qwen3-32B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen/qwen3-32b": "Qwen/Qwen3-32B",
    "openai.gpt-oss-20b-1:0": "openai/gpt-oss-20b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    "meta.llama3-8b-instruct-v1:0": "NousResearch/Meta-Llama-3-8B-Instruct",
    "llama-3-8b": "NousResearch/Meta-Llama-3-8B-Instruct",
    "meta.llama3-70b-instruct-v1:0": "NousResearch/Meta-Llama-3-8B-Instruct",
    "llama3 70b": "NousResearch/Meta-Llama-3-8B-Instruct",
    "llama-3-70b": "NousResearch/Meta-Llama-3-8B-Instruct",
}

MODEL_LABELS = {
    "qwen.qwen3-4b-v1:0": "Qwen3-4B",
    "qwen3-4b": "Qwen3-4B",
    "qwen/qwen3-4b": "Qwen3-4B",
    "qwen.qwen3-8b-v1:0": "Qwen3-8B",
    "qwen3-8b": "Qwen3-8B",
    "qwen/qwen3-8b": "Qwen3-8B",
    "qwen.qwen3-32b-v1:0": "Qwen3-32B",
    "qwen3-32b": "Qwen3-32B",
    "qwen/qwen3-32b": "Qwen3-32B",
    "openai.gpt-oss-20b-1:0": "GPT-OSS-20B",
    "gpt-oss-20b": "GPT-OSS-20B",
    "openai/gpt-oss-20b": "GPT-OSS-20B",
    "meta.llama3-8b-instruct-v1:0": "Llama-3-8B",
    "llama-3-8b": "Llama-3-8B",
    "meta.llama3-70b-instruct-v1:0": "Llama3 70B",
    "llama3 70b": "Llama3 70B",
    "llama-3-70b": "Llama3 70B",
}

FIELDNAMES = (
    "Dataset",
    "Model",
    "Tokenizer",
    "Paradigm",
    "Attack",
    "Position",
    "Prompt",
    "Phase",
    "Prompts counted",
    "Prompts missing text",
    "Total tokens",
    "Mean tokens per prompt",
    "Median tokens per prompt",
    "P95 tokens per prompt",
    "Minimum tokens",
    "Maximum tokens",
    "Mean delta from clean",
    "Date",
    "Source",
)

KEY_FIELDS = (
    "Dataset",
    "Model",
    "Paradigm",
    "Attack",
    "Position",
    "Prompt",
    "Phase",
)


@dataclass(frozen=True)
class _Run:
    metadata: dict[str, Any]
    result_path: Path
    detail_path: Path

    @property
    def key(self) -> tuple[str, ...]:
        return (
            str(self.metadata.get("dataset_name", "")),
            str(self.metadata.get("model_name", "")),
            str(self.metadata.get("ranking_scheme", "")),
            str(self.metadata.get("attack_type", "")),
            str(self.metadata.get("attack_position", "")),
            str(self.metadata.get("prompt_mode", "standard")),
        )

    @property
    def priority(self) -> tuple[int, int, str, str]:
        requested = int(
            self.metadata.get(
                "original_total_rankings", self.metadata.get("total_queries", 0)
            )
        )
        valid = int(
            self.metadata.get(
                "attacked_valid_rankings", self.metadata.get("total_queries", 0)
            )
        )
        return (
            requested,
            valid,
            str(self.metadata.get("date", "")),
            str(self.result_path),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count raw input tokens in completed clean and attacked ranking prompts."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing matching result_*.jsonl and detail_*.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Portable aggregate CSV to update.",
    )
    parser.add_argument(
        "--tokenizer",
        action="append",
        default=[],
        metavar="MODEL=HF_TOKENIZER",
        help="Override or add a model-to-tokenizer mapping; repeat as needed.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Only process this model label or model ID; repeat as needed.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Only process this dataset ID; repeat as needed.",
    )
    parser.add_argument(
        "--position",
        choices=("back", "front", "all"),
        default="back",
        help="Attack position to include. Defaults to the reported back-position runs.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only tokenizers already present in the Hugging Face cache.",
    )
    parser.add_argument(
        "--max-prompts-per-phase",
        type=int,
        default=None,
        help="Optional cap for a quick diagnostic; omit to count every prompt.",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Replace the output CSV instead of retaining rows from another machine.",
    )
    return parser.parse_args()


def _parse_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        model, separator, tokenizer = value.partition("=")
        if not separator or not model.strip() or not tokenizer.strip():
            raise ValueError(
                f"Invalid --tokenizer value {value!r}; expected MODEL=HF_TOKENIZER."
            )
        overrides[model.strip().lower()] = tokenizer.strip()
    return overrides


def _detail_path(result_path: Path) -> Path:
    suffix = result_path.name.removeprefix("result_").removesuffix(".jsonl")
    return result_path.with_name(f"detail_{suffix}.json")


def _selected_runs(results_dir: Path) -> list[_Run]:
    selected: dict[tuple[str, ...], _Run] = {}
    for result_path in sorted(results_dir.rglob("result_*.jsonl")):
        detail_path = _detail_path(result_path)
        if not detail_path.exists():
            continue
        with result_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    metadata = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(metadata, dict):
                    continue
                run = _Run(metadata, result_path, detail_path)
                if run.key not in selected or run.priority > selected[run.key].priority:
                    selected[run.key] = run
    return list(selected.values())


def _display_model(model_name: str) -> str:
    normalized = model_name.strip().lower()
    return MODEL_LABELS.get(normalized, model_name.rsplit("/", maxsplit=1)[-1])


def _dataset_label(dataset_name: str) -> str:
    return dataset_name.replace("msmarco-passage/trec-dl-", "TREC-DL-")


def _nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _source_label(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _local_tokenizer_path(tokenizer_name: str) -> str:
    supplied_path = Path(tokenizer_name).expanduser()
    if supplied_path.exists():
        return str(supplied_path)

    cache_roots = []
    if os.environ.get("HF_HUB_CACHE"):
        cache_roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        cache_roots.append(Path(os.environ["HF_HOME"]) / "hub")
    if os.environ.get("XDG_CACHE_HOME"):
        cache_roots.append(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "hub")
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    repository = "models--" + tokenizer_name.replace("/", "--")
    snapshots = []
    for cache_root in cache_roots:
        snapshots.extend((cache_root / repository / "snapshots").glob("*"))
    usable = [path for path in snapshots if (path / "tokenizer_config.json").exists()]
    if not usable:
        return tokenizer_name
    return str(max(usable, key=lambda path: path.stat().st_mtime))


def _count_run_prompts(
    run: _Run,
    tokenizer: Any,
    tokenizer_name: str,
    max_prompts_per_phase: int | None = None,
) -> list[dict[str, Any]]:
    with run.detail_path.open(encoding="utf-8") as handle:
        details = json.load(handle)
    if not isinstance(details, list):
        return []

    token_counts: dict[str, list[int]] = defaultdict(list)
    missing_counts: dict[str, int] = defaultdict(int)
    seen_counts: dict[str, int] = defaultdict(int)
    for detail in details:
        if not isinstance(detail, dict):
            continue
        phase = str(detail.get("phase", "")).lower()
        if phase not in {"original", "attacked"}:
            continue
        if (
            max_prompts_per_phase is not None
            and seen_counts[phase] >= max_prompts_per_phase
        ):
            continue
        seen_counts[phase] += 1
        prompt = detail.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            missing_counts[phase] += 1
            continue
        token_counts[phase].append(
            len(tokenizer.encode(prompt, add_special_tokens=False))
        )

    means = {
        phase: statistics.fmean(counts)
        for phase, counts in token_counts.items()
        if counts
    }
    clean_mean = means.get("original")
    metadata = run.metadata
    rows = []
    for phase in ("original", "attacked"):
        counts = token_counts.get(phase, [])
        missing = missing_counts.get(phase, 0)
        if not counts and not missing:
            continue
        mean = means.get(phase)
        rows.append(
            {
                "Dataset": _dataset_label(str(metadata.get("dataset_name", ""))),
                "Model": _display_model(str(metadata.get("model_name", "Unknown"))),
                "Tokenizer": tokenizer_name,
                "Paradigm": PARADIGM_LABELS.get(
                    str(metadata.get("ranking_scheme", "")),
                    str(metadata.get("ranking_scheme", "")),
                ),
                "Attack": ATTACK_LABELS.get(
                    str(metadata.get("attack_type", "")),
                    str(metadata.get("attack_type", "")),
                ),
                "Position": str(metadata.get("attack_position", "")),
                "Prompt": PROMPT_LABELS.get(
                    str(metadata.get("prompt_mode", "standard")),
                    str(metadata.get("prompt_mode", "standard")),
                ),
                "Phase": "Clean" if phase == "original" else "Attacked",
                "Prompts counted": len(counts),
                "Prompts missing text": missing,
                "Total tokens": sum(counts) if counts else "",
                "Mean tokens per prompt": f"{mean:.2f}" if mean is not None else "",
                "Median tokens per prompt": (
                    f"{statistics.median(counts):.2f}" if counts else ""
                ),
                "P95 tokens per prompt": _nearest_rank(counts, 0.95) if counts else "",
                "Minimum tokens": min(counts) if counts else "",
                "Maximum tokens": max(counts) if counts else "",
                "Mean delta from clean": (
                    f"{mean - clean_mean:.2f}"
                    if mean is not None and clean_mean is not None
                    else ""
                ),
                "Date": str(metadata.get("date", "")),
                "Source": _source_label(run.detail_path),
            }
        )
    return rows


def _row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def _row_priority(row: dict[str, Any]) -> tuple[int, str, str]:
    return int(row["Prompts counted"]), str(row["Date"]), str(row["Source"])


def _existing_rows(output_path: Path) -> dict[tuple[str, ...], dict[str, Any]]:
    if not output_path.exists():
        return {}
    with output_path.open(newline="", encoding="utf-8") as handle:
        return {_row_key(row): row for row in csv.DictReader(handle)}


def _write_rows(output_path: Path, rows: dict[tuple[str, ...], dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows.values(),
        key=lambda row: (
            row["Dataset"],
            row["Model"],
            row["Paradigm"],
            row["Attack"],
            row["Prompt"],
            row["Phase"],
        ),
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered)


def main() -> int:
    """Calculate and persist per-run prompt-token summaries.

    Returns
    -------
    int
        Zero when at least one run was counted, otherwise one.
    """
    args = _parse_args()
    results_dir = args.results_dir.resolve()
    output_path = args.output.resolve()
    tokenizer_map = MODEL_TOKENIZERS | _parse_overrides(args.tokenizer)
    model_filters = {value.lower() for value in args.model}
    dataset_filters = set(args.dataset)

    rows = {} if args.no_merge else _existing_rows(output_path)
    tokenizers: dict[str, Any] = {}
    updated = 0
    counted_runs = 0
    skipped_models: set[str] = set()

    for run in _selected_runs(results_dir):
        metadata = run.metadata
        model_name = str(metadata.get("model_name", ""))
        display_model = _display_model(model_name)
        dataset_name = str(metadata.get("dataset_name", ""))
        position = str(metadata.get("attack_position", ""))
        if model_filters and not {
            model_name.lower(),
            display_model.lower(),
        }.intersection(model_filters):
            continue
        if dataset_filters and dataset_name not in dataset_filters:
            continue
        if args.position != "all" and position != args.position:
            continue

        tokenizer_name = tokenizer_map.get(model_name.lower())
        if tokenizer_name is None:
            skipped_models.add(model_name)
            continue
        if tokenizer_name not in tokenizers:
            try:
                tokenizer_source = (
                    _local_tokenizer_path(tokenizer_name)
                    if args.local_files_only
                    else tokenizer_name
                )
                tokenizers[tokenizer_name] = AutoTokenizer.from_pretrained(
                    tokenizer_source,
                    local_files_only=args.local_files_only,
                )
            except (OSError, ValueError) as error:
                print(
                    f"Warning: could not load {tokenizer_name}: {error}",
                    file=sys.stderr,
                )
                tokenizers[tokenizer_name] = None
        tokenizer = tokenizers[tokenizer_name]
        if tokenizer is None:
            continue

        run_rows = _count_run_prompts(
            run, tokenizer, tokenizer_name, args.max_prompts_per_phase
        )
        if run_rows:
            counted_runs += 1
        for row in run_rows:
            key = _row_key(row)
            if key not in rows or _row_priority(row) > _row_priority(rows[key]):
                rows[key] = row
                updated += 1

    _write_rows(output_path, rows)
    print(f"Updated {updated} prompt-token rows; retained {len(rows)} total rows.")
    print(f"Wrote {output_path}")
    if skipped_models:
        print(
            "Skipped models without tokenizer mappings: "
            + ", ".join(sorted(skipped_models)),
            file=sys.stderr,
        )
        print("Add mappings with --tokenizer 'MODEL=HF_TOKENIZER'.", file=sys.stderr)
    return 0 if counted_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
