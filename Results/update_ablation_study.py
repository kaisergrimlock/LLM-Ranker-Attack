"""Aggregate pointwise and all-paradigm ablation summaries."""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPT_DIR = ROOT / "LLM_prompt_attack/outputs/reasoning_ablation/gptoss_all_2019_1000"
QWEN_DIR = ROOT / "LLM_prompt_attack/outputs/thinking_ablation/qwen3_all_2019_1000"
DEFAULT_OUTPUT = ROOT / "Results/ablation_study.csv"
PATTERN = re.compile(r"(?P<model>GPT-OSS-20B|Qwen3-32B)_(?P<paradigm>pointwise|pairwise|setwise|listwise)_2019_(?P<attack>so|sd|qi)_(?P<prompt>standard|defense)_(?:(?P<reasoning>reasoning)_(?P<reasoning_level>low|medium|high)|(?P<thinking>thinking)_(?P<thinking_level>off|on))_n(?P<n>\d+)")
FIELDS = ["Setting", "Dataset", "Model", "Paradigm", "Attack", "Prompt", "Passages", "Requested", "Valid attacked", "Discarded", "Attack success", "Attack success (%)", "Date", "Source"]

def main(output: Path = DEFAULT_OUTPUT) -> int:
    rows = []
    for directory in (GPT_DIR, QWEN_DIR):
        for path in sorted(directory.glob("*.jsonl")):
            match = PATTERN.search(path.name)
            if not match: continue
            info = match.groupdict()
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not records: continue
            record = records[-1]
            requested = int(record.get("original_total_rankings", record.get("total_queries", 0)))
            valid = int(record.get("attacked_valid_rankings", record.get("total_queries", requested)))
            success = int(record.get("pointwise_flip_count", record.get("attack_success_count", record.get("flipped_count", record.get("attack_top_position_count", 0)))))
            percentage = record.get("pointwise_flip_percentage", record.get("attack_success_rate", 100 * success / requested if requested else 0))
            setting = info["reasoning_level"] or info["thinking_level"]
            rows.append({"Setting": setting, "Dataset": "TREC-DL-2019", "Model": info["model"], "Paradigm": info["paradigm"].capitalize(), "Attack": {"so": "DOH", "sd": "DCH", "qi": "QI"}[info["attack"]], "Prompt": "Defense" if info["prompt"] == "defense" else "Default", "Passages": info["n"], "Requested": requested, "Valid attacked": valid, "Discarded": requested - valid, "Attack success": success, "Attack success (%)": percentage, "Date": record.get("date", ""), "Source": path.relative_to(ROOT).as_posix()})
    rows.sort(key=lambda row: (row["Model"], row["Paradigm"], row["Setting"], row["Attack"], row["Prompt"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} ablation rows to {output}")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    raise SystemExit(main(parser.parse_args().output))
