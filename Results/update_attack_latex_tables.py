#!/usr/bin/env python3
"""Regenerate the failure-aware attack LaTeX tables from attack_outcomes.csv.

The outcomes CSV is the authoritative aggregate for the full evaluation.  A
failed model call remains in ``Requested`` and therefore contributes a zero
to the attack-success numerator.  Reasoning/thinking ablation outputs are not
present in the outcomes CSV and are consequently excluded here.

This deliberately does not update ``ndcg_table.tex``: that table is generated
from TREC run files by ``build_ndcg_latex_table.py``, not from attack outcomes.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"
DEFAULT_INPUT = RESULTS / "attack_outcomes.csv"

DATASETS = ("TREC-DL-2019", "TREC-DL-2020")
DATASET_KEYS = {"TREC-DL-2019": "TREC-DL-2019", "TREC-DL-2020": "TREC-DL-2020"}
PARADIGMS = ("Pairwise", "Setwise", "Listwise", "Pointwise")
MODELS = ("Qwen3-4B", "Qwen3-32B", "GPT-OSS-20B", "Llama-3-8B", "Llama3-70B")
ATTACKS = {
    "DOH": {
        "filename": "attack_table_failure_aware_doh.tex",
        "label": "DOH",
        "defense": "Defense",
    },
    "DCH": {
        "filename": "attack_table_failure_aware_dch.tex",
        "label": "DCH",
        "defense": "Defense",
    },
    "Keyword injection": {
        "filename": "attack_table_failure_aware_keyword-injection.tex",
        "label": "Keyword injection",
        "defense": "Defense",
    },
    "Query injection": {
        "filename": "attack_table_failure_aware_query-injection.tex",
        "label": "Query injection",
        "defense": "Defense QI",
    },
}

ALIASES = {
    "Llama3 70B": "Llama3-70B",
    "Llama3-70B": "Llama3-70B",
    "Llama-3.3-70B": "Llama3-70B",
    "Qwen3-4b": "Qwen3-4B",
}


def latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def read_rows(path: Path) -> dict[tuple[str, str, str, str, str], dict[str, int]]:
    selected: dict[tuple[str, str, str, str, str], dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            dataset = DATASET_KEYS.get(row.get("Dataset", "").strip())
            model = ALIASES.get(row.get("Model", "").strip(), row.get("Model", "").strip())
            paradigm = row.get("Paradigm", "").strip()
            attack = row.get("Attack", "").strip()
            prompt = row.get("Prompt", "").strip()
            # These labels refer to the same standard defense condition in
            # the full-table results.
            if prompt == "Defense Qwen":
                prompt = "Defense"
            if dataset not in DATASETS or model not in MODELS:
                continue
            if paradigm not in PARADIGMS or attack not in ATTACKS:
                continue
            if prompt not in {"Default", "Defense", "Defense QI"}:
                continue
            try:
                requested = int(row["Requested"])
                success = int(row["Attack success"])
            except (KeyError, TypeError, ValueError):
                continue
            if requested <= 0 or success < 0 or success > requested:
                continue
            key = (dataset, model, paradigm, attack, prompt)
            # attack_outcomes.csv is already deduplicated. Keep the last row
            # if an older hand-merged file contains a duplicate key.
            selected[key] = {"requested": requested, "success": success}
    return selected


def metric(rows, dataset, model, paradigm, attack, prompt):
    value = rows.get((dataset, model, paradigm, attack, prompt))
    if value is None:
        return "--"
    requested = value["requested"]
    success = value["success"]
    rate = 100.0 * success / requested
    return rf"\makecell[c]{{{rate:.2f}\%\\[-1pt]\scriptsize({success}/{requested})}}"


def mean_cell(rows, dataset, paradigm, attack, prompt):
    values = []
    for model in MODELS:
        value = rows.get((dataset, model, paradigm, attack, prompt))
        if value is not None:
            values.append(100.0 * value["success"] / value["requested"])
    if not values:
        return "--"
    mean = sum(values) / len(values)
    std = math.sqrt(sum((item - mean) ** 2 for item in values) / len(values))
    return rf"{mean:.2f}\% $\pm$ {std:.2f}"


def render(rows, attack: str) -> str:
    config = ATTACKS[attack]
    defense = config["defense"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{attack} attack success rates with failed calls included in the denominator. Parentheses report successes / requested calls.}}",
        rf"\label{{tab:attack-failure-aware-{attack.lower().replace(' ', '-')}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll*{8}{c}}",
        r"\toprule",
        r" & \textbf{Model} & \multicolumn{2}{c}{\textbf{Pairwise}} & \multicolumn{2}{c}{\textbf{Setwise}} & \multicolumn{2}{c}{\textbf{Listwise}} & \multicolumn{2}{c}{\textbf{Pointwise}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}",
        rf" & & Default & {defense} & Default & {defense} & Default & {defense} & Default & {defense} \\",
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASETS):
        for model_index, model in enumerate(MODELS):
            prefix = (
                rf"\multirow{{6}}{{*}}{{\rotatebox{{90}}{{{dataset}}}}} & "
                if model_index == 0
                else " & "
            )
            cells = []
            for paradigm in PARADIGMS:
                cells.append(metric(rows, dataset, model, paradigm, attack, "Default"))
                cells.append(metric(rows, dataset, model, paradigm, attack, defense))
            lines.append(prefix + latex_escape(model) + " & " + " & ".join(cells) + r" \\")
        means = []
        for paradigm in PARADIGMS:
            means.append(mean_cell(rows, dataset, paradigm, attack, "Default"))
            means.append(mean_cell(rows, dataset, paradigm, attack, defense))
        lines.append(r"\midrule")
        lines.append(r" & \textbf{Mean $\pm$ Std} & " + " & ".join(means) + r" \\")
        if dataset_index == 0:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def write_table(path: Path, content: str, backup: bool) -> None:
    if backup and path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS,
        help="Directory for generated tables; defaults to Results.",
    )
    parser.add_argument(
        "--attack",
        action="append",
        choices=tuple(ATTACKS),
        help="Update only this attack; repeat for multiple attacks. Default: all four.",
    )
    parser.add_argument("--backup", action="store_true", help="Save existing .tex files as .tex.bak")
    parser.add_argument("--dry-run", action="store_true", help="Report targets without writing files")
    args = parser.parse_args()

    rows = read_rows(args.input.resolve())
    attacks = args.attack or list(ATTACKS)
    print(f"Read {len(rows)} outcome rows from {args.input}")
    print("Ablation outputs and NDCG are excluded by design.")
    for attack in attacks:
        target = args.output_dir / ATTACKS[attack]["filename"]
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            write_table(target, render(rows, attack), args.backup)
        print(f"{'Would write' if args.dry_run else 'Wrote'} {target}")
        if attack == "Keyword injection":
            alias = args.output_dir / "keyword_injection_table.tex"
            if not args.dry_run:
                write_table(alias, render(rows, attack), args.backup)
            print(f"{'Would write' if args.dry_run else 'Wrote'} {alias} (keyword-table alias)")


if __name__ == "__main__":
    main()
