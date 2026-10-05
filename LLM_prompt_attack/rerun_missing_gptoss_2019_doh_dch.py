#!/usr/bin/env python3
"""Resume the missing GPT-OSS-20B TREC-DL-2019 DOH/DCH full-table jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("PYTHON", "/research/remote/petabyte/users/s3891987/environments/qwen3-vllm/bin/python"))
RUN_ID = "gptoss_missing_full_2019_doh_dch"
OUTPUT_DIR = ROOT / "LLM_prompt_attack" / "outputs" / RUN_ID

JOBS = [
    ("setwise", "so", "standard"),
    ("setwise", "so", "defense"),
    ("setwise", "sd", "standard"),
    # The existing Setwise DCH Defense full-table result is retained.
    ("listwise", "so", "standard"),
    ("listwise", "so", "defense"),
    ("listwise", "sd", "standard"),
    ("listwise", "sd", "defense"),
]


def tag(paradigm: str, attack: str, prompt: str) -> str:
    return f"GPT-OSS-20B_trec-dl-2019_{paradigm}_{attack}_{prompt}"


def complete(checkpoint: Path) -> bool:
    try:
        return bool(json.loads(checkpoint.read_text()).get("complete"))
    except (OSError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Actually invoke Bedrock jobs")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pending = []
    for paradigm, attack, prompt in JOBS:
        name = tag(paradigm, attack, prompt)
        checkpoint = OUTPUT_DIR / f"{name}.checkpoint.json"
        result = OUTPUT_DIR / f"result_{name}.jsonl"
        if result.exists() and complete(checkpoint):
            print(f"Skipping completed: {name}")
        else:
            pending.append((paradigm, attack, prompt, name, checkpoint, result))

    print(json.dumps({"run_id": RUN_ID, "planned_jobs": len(JOBS), "pending_jobs": len(pending), "output_dir": str(OUTPUT_DIR)}, indent=2))
    if not args.execute:
        for job in pending:
            print("Would run:", job[3])
        print("Dry run only. Add --execute to invoke Bedrock.")
        return 0

    # Fail before spending calls if credentials are absent or expired.
    probe = subprocess.run(["aws", "sts", "get-caller-identity", "--region", args.region], capture_output=True, text=True)
    if probe.returncode != 0:
        print("AWS credential validation failed:", probe.stderr.strip(), file=sys.stderr)
        return 2

    failed = 0
    for index, (paradigm, attack, prompt, name, checkpoint, result) in enumerate(pending, 1):
        script = ROOT / "LLM_prompt_attack" / f"{paradigm}_ranking_attack_openai.py"
        detail = OUTPUT_DIR / f"detail_{name}.json"
        log = OUTPUT_DIR / f"run_{name}.log"
        command = [
            str(PYTHON), str(script),
            "--provider", "amazon-bedrock", "--aws_region", args.region,
            "--model_name", "openai.gpt-oss-20b-1:0",
            "--tokenizer_model", "openai/gpt-oss-20b",
            "--dataset_name", "msmarco-passage/trec-dl-2019",
            "--num_sets", "4096", "--set_size", "4", "--seed", "42",
            "--n_jobs", str(args.n_jobs),
            "--attack_type", attack, "--attack_position", "back",
            "--prompt_mode", prompt,
            "--result_json_path", str(result),
            "--detailed_results", str(detail),
            "--checkpoint_path", str(checkpoint),
            "--checkpoint_batch_size", "32",
        ]
        if checkpoint.exists():
            command.append("--resume")
        print(f"[{index}/{len(pending)}] Running {name}", flush=True)
        print("Command:", " ".join(command), flush=True)
        run_env = os.environ.copy()
        run_env["BEDROCK_MAX_TOKENS"] = str(args.max_tokens)
        with log.open("a", encoding="utf-8") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=run_env)
        if completed.returncode != 0:
            failed += 1
            print(f"FAILED: {name}; see {log}", file=sys.stderr)
            if completed.returncode in {130, 143}:
                break
        else:
            print(f"Finished: {name}", flush=True)

    if failed == 0:
        updater = [str(PYTHON), str(ROOT / "Results/update_attack_outcomes.py"), "--source-contains", RUN_ID]
        subprocess.run(updater, check=False)
    print(json.dumps({"completed_or_attempted": len(pending), "failed_jobs": failed, "output_dir": str(OUTPUT_DIR)}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
