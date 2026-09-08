"""Summarize query-injection evaluations with grade-3/grade-2 candidates."""

import csv
import html
import json

from update_attack_table import PROJECT_ROOT, RESULTS_DIR, _extract_metric


def main() -> None:
    """Write CSV, Markdown, and HTML summaries from close_attack results."""
    selected = {}
    for path in sorted((PROJECT_ROOT / "outputs" / "close_attack").rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("attack_type") != "qi":
                continue
            if record.get("evaluation_set") != "close_attack":
                raise ValueError(f"Missing close_attack provenance in {path}")
            metric = _extract_metric(record)
            if metric is None:
                continue
            key = tuple(
                record[field]
                for field in (
                    "dataset_name",
                    "model_name",
                    "ranking_scheme",
                    "prompt_mode",
                )
            )
            candidate = (record.get("date", ""), str(path), metric)
            if key not in selected or candidate[:2] > selected[key][:2]:
                selected[key] = candidate
    if not selected:
        raise SystemExit("No completed close_attack results found; no table written.")
    headers = [
        "Dataset",
        "Model",
        "Paradigm",
        "Prompt",
        "Success %",
        "Success",
        "Valid",
    ]
    rows = []
    for key, (_, _, (rate, success, valid)) in sorted(selected.items()):
        rows.append([*key, f"{rate:.2f}", str(success), str(valid)])
    with (RESULTS_DIR / "close_attack_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
    description = (
        "Query injection at the back of the model-selected target passage. "
        "Candidate grades: pairwise 3/2; setwise/listwise 3/2/2/2. "
        "Success uses each paradigm's existing valid-instance denominator. "
        "The latest completed run per dataset/model/paradigm/prompt is shown."
    )
    lines = ["# Close-attack results", "", description, ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    (RESULTS_DIR / "close_attack_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    table = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    heading = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
    (RESULTS_DIR / "close_attack_table.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>Close-attack results</title></head><body>"
        f"<h1>Close-attack results</h1><p>{html.escape(description)}</p>"
        f"<table><thead><tr>{heading}</tr></thead><tbody>{table}</tbody></table>"
        "</body></html>\n",
        encoding="utf-8",
    )
    print(f"Updated close_attack_table with {len(rows)} result(s).")


if __name__ == "__main__":
    main()
