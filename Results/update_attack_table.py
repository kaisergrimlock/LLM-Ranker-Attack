"""Build the TREC-DL attack comparison table from completed experiment results."""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "Results"
EXPERIMENT_OUTPUT_DIR = PROJECT_ROOT / "LLM_prompt_attack" / "outputs"
ATTACK_POSITION = "back"

DATASET_ORDER = (
    "msmarco-passage/trec-dl-2019",
    "msmarco-passage/trec-dl-2020",
)
DATASET_LABELS = {
    "msmarco-passage/trec-dl-2019": "TREC-DL-2019",
    "msmarco-passage/trec-dl-2020": "TREC-DL-2020",
}
MODEL_ORDER = (
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "Gemma-3-12B",
    "Gemma-3-27B",
    "Llama-3-8B",
    "Llama-3.3-70B",
    "GPT-4.1-mini",
    "GPT-OSS-20B",
)
MODEL_ALIASES = {
    "qwen/qwen3-1.7b": "Qwen3-1.7B",
    "qwen/qwen3-8b": "Qwen3-8B",
    "qwen/qwen3-14b": "Qwen3-14B",
    "qwen/qwen3-32b": "Qwen3-32B",
    "qwen.qwen3-32b-v1:0": "Qwen3-32B",
    "google/gemma-3-12b-it": "Gemma-3-12B",
    "google/gemma-3-27b-it": "Gemma-3-27B",
    "meta.llama3-8b-instruct-v1:0": "Llama-3-8B",
    "meta.llama3-3-70b-instruct-v1:0": "Llama-3.3-70B",
    "gpt-4.1-mini": "GPT-4.1-mini",
    "openai/gpt-oss-20b": "GPT-OSS-20B",
    "openai.gpt-oss-20b-1:0": "GPT-OSS-20B",
}
SCHEMES = ("pairwise", "setwise", "listwise")
SCHEME_LABELS = {
    "pairwise": "Pairwise Flipped %",
    "setwise": "Setwise Attack Success %",
    "listwise": "Listwise Attack Top Position %",
}
ATTACKS = ("DOH", "DCH")
ATTACK_LABELS = {"so": "DOH", "sd": "DCH"}
SOURCES = ("Orig", "Ours", "Defense")
PROMPT_MODE_BY_SOURCE = {
    "Ours": "standard",
    "Defense": "defense",
}
DATASET_BY_LABEL = {label: dataset for dataset, label in DATASET_LABELS.items()}
TABLE_METRIC_PATTERN = re.compile(
    r"^(?P<rate>\d+(?:\.\d+)?)% \((?P<numerator>\d+)/(?P<denominator>\d+)\)$"
)


def _display_model(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized]
    return model_name.rsplit("/", maxsplit=1)[-1]


def _extract_metric(record: dict) -> tuple[float, int, int] | None:
    scheme = record.get("ranking_scheme")
    total = int(record.get("total_queries", 0))
    if scheme == "pairwise":
        return (
            float(record["flipped_percentage"]),
            int(record["flipped_count"]),
            total,
        )
    if scheme == "setwise":
        return (
            float(record["attack_success_rate"]),
            int(record["attack_success_count"]),
            total,
        )
    if scheme == "listwise":
        invalid = int(record.get("invalid_ranking_count", 0))
        return (
            float(record["attack_top_position_rate"]),
            int(record["attack_top_position_count"]),
            total - invalid,
        )
    return None


def _load_results() -> dict[tuple[str, str, str, str, str], dict]:
    selected: dict[tuple[str, str, str, str, str], dict] = {}
    for path in sorted(EXPERIMENT_OUTPUT_DIR.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as result_file:
            for line_number, line in enumerate(result_file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                scheme = record.get("ranking_scheme")
                attack = ATTACK_LABELS.get(record.get("attack_type"))
                prompt_mode = record.get("prompt_mode", "standard")
                if (
                    scheme not in SCHEMES
                    or attack is None
                    or record.get("attack_position") != ATTACK_POSITION
                    or prompt_mode not in PROMPT_MODE_BY_SOURCE.values()
                ):
                    continue
                metric = _extract_metric(record)
                if metric is None:
                    continue
                rate, numerator, denominator = metric
                dataset = record["dataset_name"]
                model = _display_model(record["model_name"])
                key = (dataset, model, scheme, attack, prompt_mode)
                candidate = {
                    "rate": rate,
                    "numerator": numerator,
                    "denominator": denominator,
                    "date": record.get("date", ""),
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "line": line_number,
                }
                current = selected.get(key)
                candidate_priority = (
                    candidate["denominator"],
                    candidate["date"],
                    candidate["path"],
                    candidate["line"],
                )
                current_priority = (
                    (
                        current["denominator"],
                        current["date"],
                        current["path"],
                        current["line"],
                    )
                    if current
                    else None
                )
                if current_priority is None or candidate_priority > current_priority:
                    selected[key] = candidate
    return selected


def _load_existing_table_results() -> dict[tuple[str, str, str, str, str], dict]:
    """Load prior table values as a fallback when raw JSONL is unavailable."""
    selected: dict[tuple[str, str, str, str, str], dict] = {}
    table_path = RESULTS_DIR / "attack_table.csv"
    if not table_path.exists():
        return selected

    with table_path.open(encoding="utf-8", newline="") as table_file:
        for line_number, row in enumerate(csv.DictReader(table_file), start=2):
            dataset = DATASET_BY_LABEL.get(row.get("Dataset", ""))
            model = row.get("Model", "")
            if not dataset or not model or model == "Mean Â± Std":
                continue
            for scheme in SCHEMES:
                for attack in ATTACKS:
                    for source, prompt_mode in PROMPT_MODE_BY_SOURCE.items():
                        column = f"{SCHEME_LABELS[scheme]} | {attack} | {source}"
                        match = TABLE_METRIC_PATTERN.match(row.get(column) or "")
                        if match is None:
                            continue
                        key = (dataset, model, scheme, attack, prompt_mode)
                        selected[key] = {
                            "rate": float(match["rate"]),
                            "numerator": int(match["numerator"]),
                            "denominator": int(match["denominator"]),
                            "date": "",
                            "path": str(table_path.relative_to(PROJECT_ROOT)),
                            "line": line_number,
                        }
    return selected


def _datasets_and_models(
    selected: dict[tuple[str, str, str, str, str], dict],
) -> tuple[list[str], dict[str, list[str]]]:
    discovered_datasets = {key[0] for key in selected}
    datasets = list(DATASET_ORDER)
    datasets.extend(sorted(discovered_datasets.difference(datasets)))

    models_by_dataset = {}
    for dataset in datasets:
        discovered_models = {key[1] for key in selected if key[0] == dataset}
        models = list(MODEL_ORDER)
        models.extend(sorted(discovered_models.difference(models)))
        models_by_dataset[dataset] = models
    return datasets, models_by_dataset


def _value(
    selected: dict[tuple[str, str, str, str, str], dict],
    dataset: str,
    model: str,
    scheme: str,
    attack: str,
    source: str,
) -> dict | None:
    if source == "Orig":
        return None
    prompt_mode = PROMPT_MODE_BY_SOURCE[source]
    return selected.get((dataset, model, scheme, attack, prompt_mode))


def _plain_value(metric: dict | None) -> str:
    if metric is None:
        return ""
    return f"{metric['rate']:.2f}% ({metric['numerator']}/{metric['denominator']})"


def _html_value(metric: dict | None) -> str:
    if metric is None:
        return ""
    return (
        f"{metric['rate']:.2f}%<br>"
        f"<span>({metric['numerator']}/{metric['denominator']})</span>"
    )


def _write_csv(
    selected: dict[tuple[str, str, str, str, str], dict],
    datasets: list[str],
    models_by_dataset: dict[str, list[str]],
) -> None:
    columns = ["Dataset", "Model"]
    columns.extend(
        f"{SCHEME_LABELS[scheme]} | {attack} | {source}"
        for scheme in SCHEMES
        for attack in ATTACKS
        for source in SOURCES
    )
    with (RESULTS_DIR / "attack_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        for dataset in datasets:
            for model in models_by_dataset[dataset]:
                row = [DATASET_LABELS.get(dataset, dataset), model]
                row.extend(
                    _plain_value(
                        _value(selected, dataset, model, scheme, attack, source)
                    )
                    for scheme in SCHEMES
                    for attack in ATTACKS
                    for source in SOURCES
                )
                writer.writerow(row)
            writer.writerow([DATASET_LABELS.get(dataset, dataset), "Mean ± Std"])


def _write_markdown(
    selected: dict[tuple[str, str, str, str, str], dict],
    datasets: list[str],
    models_by_dataset: dict[str, list[str]],
) -> None:
    headers = ["Dataset", "Model"]
    headers.extend(
        f"{SCHEME_LABELS[scheme]}<br>{attack} {source}"
        for scheme in SCHEMES
        for attack in ATTACKS
        for source in SOURCES
    )
    lines = [
        "# TREC-DL attack reproduction table",
        "",
        f"Only `{ATTACK_POSITION}`-position runs are included. "
        "`Ours` uses the standard evaluator prompt; `Defense` uses the defense "
        "evaluator prompt. Blank cells have no recorded result.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for dataset in datasets:
        label = DATASET_LABELS.get(dataset, dataset)
        for model in models_by_dataset[dataset]:
            row = [label, model]
            row.extend(
                _plain_value(
                    _value(selected, dataset, model, scheme, attack, source)
                ).replace(" ", "<br>", 1)
                for scheme in SCHEMES
                for attack in ATTACKS
                for source in SOURCES
            )
            lines.append("| " + " | ".join(row) + " |")
        metric_column_count = len(SCHEMES) * len(ATTACKS) * len(SOURCES)
        lines.append(
            "| "
            + " | ".join([label, "**Mean ± Std**", *([""] * metric_column_count)])
            + " |"
        )
    (RESULTS_DIR / "attack_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_html(
    selected: dict[tuple[str, str, str, str, str], dict],
    datasets: list[str],
    models_by_dataset: dict[str, list[str]],
) -> None:
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>TREC-DL attack reproduction table</title>",
        "<style>",
        "body{font-family:Georgia,serif;margin:24px;color:#111}",
        "table{border-collapse:collapse;width:100%;font-size:14px}",
        "th,td{padding:5px 7px;text-align:center;vertical-align:middle}",
        "thead{border-top:2px solid #222;border-bottom:2px solid #222}",
        "thead tr:not(:last-child) th{border-bottom:1px solid #777}",
        ".model{text-align:left;white-space:nowrap}",
        ".dataset{writing-mode:vertical-rl;transform:rotate(180deg);font-weight:bold}",
        ".group-start{border-left:1px solid #555}",
        ".dataset-start{border-top:2px solid #777}",
        ".mean{border-top:1px solid #777;border-bottom:2px solid #777;"
        "font-weight:bold}",
        ".ours:not(:empty){background:#dceeff}",
        ".defense:not(:empty){background:#dff3df}",
        "td span{white-space:nowrap}",
        "caption{caption-side:top;text-align:left;font-weight:bold;margin-bottom:10px}",
        "</style></head><body>",
        "<table>",
        "<caption>Back-position attacks; Ours uses the standard prompt and Defense "
        "uses the defense prompt. Blank cells have no recorded result.</caption>",
        "<thead>",
        '<tr><th rowspan="3"></th><th rowspan="3">Model</th>',
    ]
    for scheme in SCHEMES:
        parts.append(
            f'<th class="group-start" colspan="{len(ATTACKS) * len(SOURCES)}">'
            f"{html.escape(SCHEME_LABELS[scheme])}</th>"
        )
    parts.append("</tr><tr>")
    for _scheme in SCHEMES:
        for attack in ATTACKS:
            parts.append(
                f'<th class="group-start" colspan="{len(SOURCES)}">'
                f"{html.escape(attack)}</th>"
            )
    parts.append("</tr><tr>")
    for _scheme in SCHEMES:
        for _attack in ATTACKS:
            for source in SOURCES:
                class_name = ' class="group-start"' if source == "Orig" else ""
                parts.append(f"<th{class_name}>{html.escape(source)}</th>")
    parts.append("</tr></thead><tbody>")

    for dataset_index, dataset in enumerate(datasets):
        models = models_by_dataset[dataset]
        dataset_class = "dataset-start" if dataset_index else ""
        for model_index, model in enumerate(models):
            row_class = dataset_class if model_index == 0 else ""
            parts.append(f'<tr class="{row_class}">')
            if model_index == 0:
                label = html.escape(DATASET_LABELS.get(dataset, dataset))
                parts.append(
                    f'<th class="dataset" rowspan="{len(models) + 1}">{label}</th>'
                )
            parts.append(f'<td class="model">{html.escape(model)}</td>')
            for scheme in SCHEMES:
                for attack in ATTACKS:
                    for source in SOURCES:
                        metric = _value(
                            selected, dataset, model, scheme, attack, source
                        )
                        if source == "Orig":
                            classes = ["group-start"]
                        elif source == "Ours":
                            classes = ["ours"]
                        else:
                            classes = ["defense"]
                        class_names = " ".join(classes)
                        parts.append(
                            f'<td class="{class_names}">{_html_value(metric)}</td>'
                        )
            parts.append("</tr>")
        parts.append('<tr class="mean"><td class="model">Mean ± Std</td>')
        metric_column_count = len(SCHEMES) * len(ATTACKS) * len(SOURCES)
        parts.extend("<td></td>" for _ in range(metric_column_count))
        parts.append("</tr>")
    parts.extend(("</tbody></table>", "</body></html>"))
    (RESULTS_DIR / "attack_table.html").write_text(
        "\n".join(parts) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Update all attack-table formats from completed JSONL runs.

    Returns
    -------
    None
        The CSV, Markdown, and HTML tables are written under ``Results``.
    """
    selected = _load_existing_table_results()
    for key, metric in _load_results().items():
        current = selected.get(key)
        if current is None or metric["denominator"] >= current["denominator"]:
            selected[key] = metric
    datasets, models_by_dataset = _datasets_and_models(selected)
    _write_csv(selected, datasets, models_by_dataset)
    _write_markdown(selected, datasets, models_by_dataset)
    _write_html(selected, datasets, models_by_dataset)
    print(f"Updated attack table with {len(selected)} recorded result(s).")


if __name__ == "__main__":
    main()
