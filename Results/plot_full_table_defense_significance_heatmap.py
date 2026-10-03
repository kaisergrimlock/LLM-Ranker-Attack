#!/usr/bin/env python3
"""Seaborn heatmap of paired defence effects across ranking paradigms.

Input is the non-ablation paired-test output produced by
test_defense_by_attack.py.  Cells show signed -log10(Holm-adjusted McNemar
p-value):

  negative = defence reduced attack success (robustness improved)
  positive = defence increased attack success
  magnitude = strength of evidence

The heatmap is therefore about statistical differences, not raw relevance or
aggregate ASR.  One figure is produced per defence condition and dataset.
"""

from __future__ import annotations

import argparse
import csv
import math
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Results/defense_by_attack_tests.csv"
DEFAULT_OUTPUT = ROOT / "Results/full_table_defense_significance_heatmap.png"
PARADIGMS = ("Pairwise", "Setwise", "Listwise", "Pointwise")
PARADIGM_SHORT = {"Pairwise": "Pair", "Setwise": "Set", "Listwise": "List", "Pointwise": "Point"}
ATTACK_CONDITIONS = {"DOH": "Defense", "DCH": "Defense", "Keyword injection": "Defense QI", "Query injection": "Defense QI"}
ATTACKS = ("DOH", "DCH", "Keyword injection", "Query injection")
MODELS = ("Qwen3-4B", "Qwen3-32B", "GPT-OSS-20B", "Llama-3-8B", "Llama3 70B")


def signed_log_p(row: dict) -> float:
    p = max(float(row["p_value_holm"]), 1e-300)
    b = int(row["discordant_a_success_b_failure"] or 0)
    c = int(row["discordant_a_failure_b_success"] or 0)
    # b > c means Default succeeded where Defense failed more often: defense helps.
    direction = -1.0 if b > c else 1.0 if c > b else 0.0
    return direction * min(300.0, -math.log10(p))


def effect_color(value: float):
    """Return the heatmap colour used by the SVG/PDF renderers."""
    from matplotlib.colors import to_rgba

    strength = min(0.88, 0.18 + abs(value) / 300 * 0.70)
    rgb = "#2e8fd3" if value < 0 else "#e89d2a"
    return to_rgba(rgb, strength)


def cell_label(value: float) -> str:
    return ("↓" if value < 0 else "↑" if value > 0 else "=") + (
        "*" if abs(value) > -math.log10(0.05) else ""
    )


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_colour(value: float | None) -> tuple[float, float, float]:
    if value is None:
        return (0.933, 0.933, 0.933)
    alpha = min(0.88, 0.18 + abs(value) / 300 * 0.70)
    base = (46 / 255, 143 / 255, 211 / 255) if value < 0 else (232 / 255, 157 / 255, 42 / 255)
    return tuple(1 - (1 - channel) * alpha for channel in base)


def _write_vector_pdf(output: Path, width: float, height: float, commands: list[str]) -> None:
    stream = "\n".join(commands).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    pdf.extend("".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode())
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(pdf)


def _pdf_text(commands: list[str], x: float, y_top: float, text: str, size: float, width: float, height: float, *, align: str = "left", bold: bool = False, colour: tuple[float, float, float] = (0.13, 0.13, 0.13)) -> None:
    # Helvetica metrics are sufficient for these compact labels.
    estimated = len(text) * size * (0.58 if not bold else 0.62)
    tx = x - estimated / 2 if align == "center" else x - estimated if align == "right" else x
    font = "/F1"
    commands.append(f"{colour[0]:.3f} {colour[1]:.3f} {colour[2]:.3f} rg BT {font} {size:.2f} Tf 1 0 0 1 {tx:.2f} {height - y_top - size:.2f} Tm ({_pdf_escape(text)}) Tj ET")


def _pdf_vertical_text(commands: list[str], x: float, y_top: float, text: str, size: float, width: float, height: float, *, bold: bool = False, colour: tuple[float, float, float] = (0.13, 0.13, 0.13)) -> None:
    """Draw text rotated 90 degrees counter-clockwise in the vector PDF."""
    font = "/F1"
    estimated = len(text) * size * (0.58 if not bold else 0.62)
    commands.append(
        f"{colour[0]:.3f} {colour[1]:.3f} {colour[2]:.3f} rg BT {font} {size:.2f} Tf "
        f"0 1 -1 0 {x:.2f} {height - y_top - estimated / 2:.2f} Tm ({_pdf_escape(text)}) Tj ET"
    )


def _pdf_rect(commands: list[str], x: float, y_top: float, w: float, h: float, colour: tuple[float, float, float], width: float, height: float) -> None:
    commands.append(f"{colour[0]:.3f} {colour[1]:.3f} {colour[2]:.3f} rg {x:.2f} {height - y_top - h:.2f} {w:.2f} {h:.2f} re f")


def _pdf_arrow(commands: list[str], x: float, y_top: float, direction: str, width: float, height: float, size: float = 10) -> None:
    """Draw a vector up/down arrow, matching the SVG labels."""
    cx = x
    top = y_top
    bottom = y_top + size
    commands.append("0.13 0.13 0.13 RG 1.1 w")
    if direction == "down":
        shaft_top, shaft_bottom = top + 1, bottom - 3
        commands.append(f"{cx:.2f} {height - shaft_top:.2f} m {cx:.2f} {height - shaft_bottom:.2f} l S")
        commands.append(f"{cx - 3:.2f} {height - (bottom - 5):.2f} m {cx:.2f} {height - (bottom - 2):.2f} l {cx + 3:.2f} {height - (bottom - 5):.2f} l S")
    else:
        shaft_top, shaft_bottom = top + 3, bottom - 1
        commands.append(f"{cx:.2f} {height - shaft_bottom:.2f} m {cx:.2f} {height - shaft_top:.2f} l S")
        commands.append(f"{cx - 3:.2f} {height - (top + 5):.2f} m {cx:.2f} {height - (top + 2):.2f} l {cx + 3:.2f} {height - (top + 5):.2f} l S")


def load_rows(path: Path, condition: str, attacks: list[str]) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [
        row for row in rows
        if row.get("test") == "mcnemar"
        and row.get("condition_a") == "Default"
        and (condition == "all" or row.get("condition_b") == condition)
        and (not attacks or row.get("attack") in attacks)
    ]


def write_svg(rows: list[dict], output: Path, condition: str, attack: str) -> None:
    models = [m for m in MODELS if any(r["model"] == m for r in rows)]
    values = {(r["dataset"], r["model"], r["paradigm"]): signed_log_p(r) for r in rows if r["attack"] == attack}
    width, height = 820, 215
    left, top, cell_w, cell_h = 125, 30, 50, 24
    panel_gap = 30
    panel_w = left + 4 * cell_w + 20
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:16px;font-weight:bold}.head{font-size:10px;font-weight:bold}.label{font-size:10px}.cell{font-size:14px;font-weight:bold}.legend{font-size:9px;fill:#555}</style>',
    ]
    for panel, dataset in enumerate(("TREC-DL-2019", "TREC-DL-2020")):
        x0 = panel * (panel_w + panel_gap)
        parts.append(f'<text x="{x0 + left + 2 * cell_w}" y="16" text-anchor="middle" class="head">{dataset}</text>')
        for col, paradigm in enumerate(PARADIGMS):
            parts.append(f'<text x="{x0 + left + col * cell_w + cell_w/2}" y="27" text-anchor="middle" class="head">{PARADIGM_SHORT[paradigm]}</text>')
        for row_i, model in enumerate(models):
            y = top + 25 + row_i * cell_h
            parts.append(f'<text x="{x0 + left - 8}" y="{y + 20}" text-anchor="end" class="label">{escape(model)}</text>')
            for col, paradigm in enumerate(PARADIGMS):
                value = values.get((dataset, model, paradigm))
                x = x0 + left + col * cell_w
                if value is None:
                    fill, label = "#eeeeee", ""
                else:
                    strength = min(0.88, 0.18 + abs(value) / 300 * 0.70)
                    fill = f'rgba({"46,143,211" if value < 0 else "232,157,42"},{strength:.3f})'
                    label = ("↓" if value < 0 else "↑" if value > 0 else "=") + ("*" if abs(value) > -math.log10(0.05) else "")
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{fill}" stroke="#fff"/>')
                parts.append(f'<text x="{x + cell_w/2}" y="{y + 20}" text-anchor="middle" class="cell">{label}</text>')
    parts.insert(-1, f'<text x="{width/2}" y="{height-8}" text-anchor="middle" class="legend">↓ Defense reduces attack success; ↑ increases it; * Holm p &lt; 0.05</text>')
    parts.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_pdf(rows: list[dict], output: Path, condition: str, attack: str) -> None:
    models = [m for m in MODELS if any(r["model"] == m for r in rows)]
    values = {(r["dataset"], r["model"], r["paradigm"]): signed_log_p(r) for r in rows if r["attack"] == attack}
    width, height = 820, 215
    left, top, cell_w, cell_h = 125, 30, 50, 24
    panel_gap = 30
    panel_w = left + 4 * cell_w + 20
    commands: list[str] = ["1 1 1 rg 0 0 1 1 re f"]
    for panel, dataset in enumerate(("TREC-DL-2019", "TREC-DL-2020")):
        x0 = panel * (panel_w + panel_gap)
        _pdf_text(commands, x0 + left + 2 * cell_w, 10, dataset, 10, width, height, align="center", bold=True)
        for col, paradigm in enumerate(PARADIGMS):
            _pdf_text(commands, x0 + left + col * cell_w + cell_w / 2, 22, PARADIGM_SHORT[paradigm], 8, width, height, align="center", bold=True)
        for row_i, model in enumerate(models):
            y = top + 25 + row_i * cell_h
            _pdf_text(commands, x0 + left - 8, y + 7, model, 8, width, height, align="right")
            for col, paradigm in enumerate(PARADIGMS):
                value = values.get((dataset, model, paradigm))
                x = x0 + left + col * cell_w
                _pdf_rect(commands, x, y, cell_w - 2, cell_h - 2, _pdf_colour(value), width, height)
                if value is not None:
                    if value < 0:
                        _pdf_arrow(commands, x + cell_w / 2, y + 6, "down", width, height, size=11)
                    elif value > 0:
                        _pdf_arrow(commands, x + cell_w / 2, y + 6, "up", width, height, size=11)
                    else:
                        _pdf_text(commands, x + cell_w / 2, y + 3, "=", 10, width, height, align="center", bold=True)
                    if abs(value) > -math.log10(0.05):
                        _pdf_text(commands, x + cell_w / 2 + 7, y + 3, "*", 7, width, height, bold=True)
    _pdf_text(commands, width / 2, height - 15, "v Defense reduces attack success; ^ increases it; * Holm p < 0.05", 7, width, height, align="center", colour=(0.33, 0.33, 0.33))
    _write_vector_pdf(output, width, height, commands)


def write_combined_svg(rows: list[dict], output: Path, condition: str) -> None:
    models = [m for m in MODELS if any(r["model"] == m for r in rows)]
    attacks = list(ATTACKS)
    values = {
        (r["dataset"], r["attack"], r["model"], r["paradigm"]): signed_log_p(r)
        for r in rows
    }
    cell_w, cell_h, left, top = 34, 21, 82, 34
    panel_w, panel_h, gap_x, gap_y = 214, 164, 10, 18
    width, height = panel_w * 4 + gap_x * 3 + 92, panel_h * 2 + gap_y + 30
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.head{font-size:9px;font-weight:bold}.label{font-size:8px}.cell{font-size:12px;font-weight:bold}.legend{font-size:9px;fill:#555}</style>',
        '<defs><linearGradient id="evidence-gradient" x1="0" y1="1" x2="0" y2="0"><stop offset="0%" stop-color="#2e8fd3"/><stop offset="50%" stop-color="#f5f5f5"/><stop offset="100%" stop-color="#e89d2a"/></linearGradient></defs>',
    ]
    for year_i, dataset in enumerate(("TREC-DL-2019", "TREC-DL-2020")):
        for attack_i, attack in enumerate(attacks):
            x0 = attack_i * (panel_w + gap_x)
            y0 = year_i * (panel_h + gap_y)
            if attack_i == 0:
                year_label = "TREC-DL " + dataset.rsplit("-", 1)[-1]
                year_x, year_y = 12, y0 + panel_h / 2
                parts.append(f'<text x="{year_x}" y="{year_y}" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90 {year_x} {year_y})" class="head">{year_label}</text>')
            parts.append(f'<text x="{x0 + left + 2 * cell_w}" y="{y0 + 13}" text-anchor="middle" class="head">{escape(attack)}</text>')
            for col, paradigm in enumerate(PARADIGMS):
                x = x0 + left + col * cell_w
                parts.append(f'<text x="{x + cell_w/2}" y="{y0 + 29}" text-anchor="middle" class="head">{PARADIGM_SHORT[paradigm]}</text>')
            for row_i, model in enumerate(models):
                y = y0 + top + row_i * cell_h
                parts.append(f'<text x="{x0 + left - 5}" y="{y + 14}" text-anchor="end" class="label">{escape(model)}</text>')
                for col, paradigm in enumerate(PARADIGMS):
                    value = values.get((dataset, attack, model, paradigm))
                    x = x0 + left + col * cell_w
                    if value is None:
                        fill, label = "#eeeeee", ""
                    else:
                        strength = min(0.88, 0.18 + abs(value) / 300 * 0.70)
                        fill = f'rgba({"46,143,211" if value < 0 else "232,157,42"},{strength:.3f})'
                        label = ("↓" if value < 0 else "↑" if value > 0 else "=") + ("*" if abs(value) > -math.log10(0.05) else "")
                    parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{fill}" stroke="#fff"/>')
                    parts.append(f'<text x="{x + cell_w/2}" y="{y + 14}" text-anchor="middle" class="cell">{label}</text>')
    bar_x, bar_y, bar_h = width - 52, 18, 328
    parts.extend([
        f'<rect x="{bar_x}" y="{bar_y}" width="13" height="{bar_h}" fill="url(#evidence-gradient)" stroke="#bbb"/>',
        f'<text x="{bar_x + 19}" y="{bar_y + 4}" class="legend">+300</text>',
        f'<text x="{bar_x + 19}" y="{bar_y + bar_h/2 + 3}" class="legend">0</text>',
        f'<text x="{bar_x + 19}" y="{bar_y + bar_h + 3}" class="legend">−300</text>',
        f'<text x="{bar_x - 7}" y="{bar_y + bar_h/2}" text-anchor="middle" transform="rotate(-90 {bar_x - 7} {bar_y + bar_h/2})" class="legend">signed −log10(Holm p)</text>',
    ])
    parts.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_combined_pdf(rows: list[dict], output: Path, condition: str) -> None:
    models = [m for m in MODELS if any(r["model"] == m for r in rows)]
    values = {(r["dataset"], r["attack"], r["model"], r["paradigm"]): signed_log_p(r) for r in rows}
    cell_w, cell_h, left, top = 34, 21, 82, 34
    panel_w, panel_h, gap_x, gap_y = 214, 164, 10, 18
    width, height = panel_w * 4 + gap_x * 3 + 92, panel_h * 2 + gap_y + 30
    commands: list[str] = ["1 1 1 rg 0 0 1 1 re f"]
    for year_i, dataset in enumerate(("TREC-DL-2019", "TREC-DL-2020")):
        for attack_i, attack in enumerate(ATTACKS):
            x0 = attack_i * (panel_w + gap_x)
            y0 = year_i * (panel_h + gap_y)
            if attack_i == 0:
                year_label = "TREC-DL " + dataset.rsplit("-", 1)[-1]
                _pdf_vertical_text(commands, 18, y0 + panel_h / 2, year_label, 8, width, height, bold=True)
            _pdf_text(commands, x0 + left + 2 * cell_w, y0 + 7, attack, 8, width, height, align="center", bold=True)
            for col, paradigm in enumerate(PARADIGMS):
                x = x0 + left + col * cell_w
                _pdf_text(commands, x + cell_w / 2, y0 + 25, PARADIGM_SHORT[paradigm], 7, width, height, align="center", bold=True)
            for row_i, model in enumerate(models):
                y = y0 + top + row_i * cell_h
                _pdf_text(commands, x0 + left - 5, y + 4, model, 6.5, width, height, align="right")
                for col, paradigm in enumerate(PARADIGMS):
                    value = values.get((dataset, attack, model, paradigm))
                    x = x0 + left + col * cell_w
                    _pdf_rect(commands, x, y, cell_w - 2, cell_h - 2, _pdf_colour(value), width, height)
                    if value is not None:
                        if value < 0:
                            _pdf_arrow(commands, x + cell_w / 2, y + 4, "down", width, height, size=9)
                        elif value > 0:
                            _pdf_arrow(commands, x + cell_w / 2, y + 4, "up", width, height, size=9)
                        else:
                            _pdf_text(commands, x + cell_w / 2, y + 2, "=", 8, width, height, align="center", bold=True)
                        if abs(value) > -math.log10(0.05):
                            _pdf_text(commands, x + cell_w / 2 + 6, y + 3, "*", 6, width, height, bold=True)
    bar_x, bar_y, bar_h = width - 52, 18, 328
    for step in range(80):
        value = -300 + 600 * (step + 0.5) / 80
        _pdf_rect(commands, bar_x, bar_y + bar_h * (1 - (step + 1) / 80), 13, bar_h / 80 + 0.5, _pdf_colour(value), width, height)
    _pdf_text(commands, bar_x + 19, bar_y - 1, "+300", 7, width, height)
    _pdf_text(commands, bar_x + 19, bar_y + bar_h / 2 - 3, "0", 7, width, height)
    _pdf_text(commands, bar_x + 19, bar_y + bar_h - 3, "-300", 7, width, height)
    _pdf_text(commands, bar_x + 25, bar_y + bar_h / 2, "signed -log10(Holm p)", 7, width, height)
    _write_vector_pdf(output, width, height, commands)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--condition", default="Defense",
        help="Defense condition to compare with Default, e.g. Defense, Defense QI, Filter QI, or all",
    )
    parser.add_argument(
        "--attack",
        choices=ATTACKS,
        action="append",
        help="Restrict the matrix to one or more attacks. Default: all attacks.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--combined", action="store_true", help="Create one 2x4 figure: years by attack method.")
    args = parser.parse_args()

    attacks = args.attack or list(ATTACKS)
    rows = load_rows(args.input, "all" if args.combined else args.condition, attacks)
    if args.combined:
        rows = [row for row in rows if row.get("condition_b") == ATTACK_CONDITIONS.get(row.get("attack"))]
    if not rows:
        raise SystemExit(f"No Default-vs-{args.condition} McNemar rows found in {args.input}.")

    if args.format in {"pdf", "svg"}:
        if args.combined:
            output = args.output.with_suffix(f".{args.format}")
            if args.format == "svg":
                write_combined_svg(rows, output, args.condition)
            else:
                write_combined_pdf(rows, output, args.condition)
            print(f"Wrote {output}")
            return 0
        for attack in attacks:
            attack_rows = [row for row in rows if row.get("attack") == attack]
            suffix = attack.lower().replace(" ", "_")
            output = args.output.with_name(f"{args.output.stem}_{suffix}.{args.format}")
            if args.format == "svg":
                write_svg(attack_rows, output, args.condition, attack)
            else:
                write_pdf(attack_rows, output, args.condition, attack)
            print(f"Wrote {output}")
        return 0

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    frame = pd.DataFrame(rows)
    frame["value"] = frame.apply(signed_log_p, axis=1)
    frame["significance"] = frame["p_value_holm"].astype(float) < args.alpha
    frame["row"] = frame["attack"] + " | " + frame["model"]
    frame["label"] = frame.apply(
        lambda r: ("↓" if r["value"] < 0 else "↑" if r["value"] > 0 else "=")
        + ("*" if r["significance"] else ""), axis=1
    )

    conditions = [args.condition] if args.condition != "all" else sorted(frame["condition_b"].unique())
    datasets = sorted(frame["dataset"].unique())
    models = [m for m in MODELS if m in set(frame["model"])]
    models += sorted(set(frame["model"]) - set(models))
    attack_order = attacks
    row_order = [f"{attack} | {model}" for attack in attack_order for model in models]

    sns.set_theme(style="white", context="paper")
    for condition in conditions:
        subset_condition = frame[frame["condition_b"] == condition]
        fig, axes = plt.subplots(1, len(datasets), figsize=(7.0 * len(datasets), 10), squeeze=False)
        axes = axes[0]
        vmax = max(1.0, subset_condition["value"].abs().max())
        for axis, dataset in zip(axes, datasets):
            subset = subset_condition[subset_condition["dataset"] == dataset]
            values = subset.pivot_table(index="row", columns="paradigm", values="value", aggfunc="first")
            values = values.reindex(index=row_order, columns=PARADIGMS)
            labels = subset.pivot_table(index="row", columns="paradigm", values="label", aggfunc="first")
            labels = labels.reindex(index=row_order, columns=PARADIGMS).fillna("")
            sns.heatmap(
                values, ax=axis, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                annot=labels, fmt="", linewidths=0.4, linecolor="white",
                cbar=axis is axes[-1], cbar_kws={"label": "Signed −log10(Holm p)"},
            )
            axis.set_title(dataset)
            axis.set_xlabel("Ranking paradigm")
            axis.set_ylabel("Attack | Model")
            axis.tick_params(axis="y", labelrotation=0, labelsize=8)
            axis.tick_params(axis="x", labelrotation=35)
        fig.suptitle(f"Full-table paired defence effects: Default vs {condition}", y=1.01)
        fig.text(0.5, -0.01, "↓ defence reduces attack success; ↑ defence increases attack success; * Holm p < α", ha="center")
        fig.tight_layout()
        suffix = "all" if args.condition == "all" else condition.lower().replace(" ", "_")
        output = args.output.with_name(f"{args.output.stem}_{suffix}{args.output.suffix}")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {output}")


if __name__ == "__main__":
    raise SystemExit(main())
