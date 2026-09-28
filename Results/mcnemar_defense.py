"""McNemar tests comparing Default and Defense attack-success outcomes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mcnemar_all import _is_complete, _key, _metadata, exact_mcnemar_p, holm


def _success_records(path: Path):
    meta = _metadata(path)
    paradigm = meta["paradigm"]
    if paradigm == "listwise":
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    original = { _key(r, paradigm): r for r in payload if isinstance(r, dict) and r.get("phase") == "original" and _key(r, paradigm) is not None }
    attacked = { _key(r, paradigm): r for r in payload if isinstance(r, dict) and r.get("phase") == "attacked" and _key(r, paradigm) is not None }
    result = {}
    for key in original.keys() & attacked.keys():
        clean, attack = original[key], attacked[key]
        if paradigm in {"pointwise", "pairwise"}:
            result[key] = str(clean.get("label", "")).strip().upper() != str(attack.get("label", "")).strip().upper()
        elif paradigm == "setwise":
            target = str(attack.get("attack_label", "")).strip().upper()
            result[key] = bool(target) and target == str(attack.get("label", "")).strip().upper()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details-dir", type=Path, default=Path("LLM_prompt_attack/outputs"))
    parser.add_argument("--output", type=Path, default=Path("Results/mcnemar_defense.csv"))
    args = parser.parse_args()
    newest = {}
    for path in args.details_dir.rglob("detail_*.json"):
        meta = _metadata(path)
        if meta["paradigm"] == "listwise" or meta["prompt"] not in {"standard", "defense"} or not _is_complete(path):
            continue
        group = tuple(meta[field] for field in ("model", "dataset", "paradigm", "attack", "prompt"))
        old = newest.get(group)
        if old is None or (path.stat().st_mtime_ns, path.as_posix()) > (old.stat().st_mtime_ns, old.as_posix()):
            newest[group] = path
    rows = []
    groups = {}
    for group, path in newest.items():
        groups.setdefault(group[:4], {})[group[4]] = path
    for (model, dataset, paradigm, attack), sources in groups.items():
        if "standard" not in sources or "defense" not in sources:
            continue
        default, defense = _success_records(sources["standard"]), _success_records(sources["defense"])
        if default is None or defense is None:
            continue
        a = b = c = d = 0
        for key in default.keys() & defense.keys():
            left, right = default[key], defense[key]
            if left and right: a += 1
            elif left and not right: b += 1
            elif not left and right: c += 1
            else: d += 1
        if a + b + c + d == 0: continue
        rows.append({"model": model, "dataset": dataset, "paradigm": paradigm, "attack": attack, "paired": a+b+c+d, "both_success": a, "default_success_defense_failure": b, "default_failure_defense_success": c, "both_failure": d, "p_value_exact": exact_mcnemar_p(b, c), "direction": "defense helps" if b > c else "defense hurts" if c > b else "tie", "default_source": sources["standard"].as_posix(), "defense_source": sources["defense"].as_posix()})
    adjusted = holm([r["p_value_exact"] for r in rows])
    for row, p in zip(rows, adjusted): row["p_value_holm"] = p; row["significant_holm_0.05"] = p < 0.05
    fields = ["model", "dataset", "paradigm", "attack", "paired", "both_success", "default_success_defense_failure", "default_failure_defense_success", "both_failure", "p_value_exact", "p_value_holm", "significant_holm_0.05", "direction", "default_source", "defense_source"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
