#!/usr/bin/env bash
# Summarise resumable pointwise evaluations submitted with `batch < ...`.
# Run on the RMIT server from the repository root:
#   bash LLM_prompt_attack/check_pointwise_progress.sh
# Or pass a different output directory:
#   bash LLM_prompt_attack/check_pointwise_progress.sh LLM_prompt_attack/outputs/<run-id>

set -uo pipefail

PROJECT="${PROJECT:-$(pwd -P)}"
RUN_DIR="${1:-$PROJECT/LLM_prompt_attack/outputs/Qwen3-4B_pointwise_doh_dch}"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

if [ ! -d "$RUN_DIR" ]; then
    echo "Run directory not found: $RUN_DIR" >&2
    exit 1
fi

echo "Pointwise run directory: $RUN_DIR"
echo

echo "Local Qwen endpoint ($BASE_URL/models):"
if curl -fsS --max-time 5 "$BASE_URL/models" 2>/dev/null | \
    grep -q 'Qwen3-4B'; then
    echo "  READY (Qwen3-4B advertised)"
else
    echo "  NOT READY or not reachable"
fi
echo

echo "Active processes (your account):"
if ps -u "$USER" -o pid,etime,stat,%cpu,%mem,args | \
    grep -E '[v]llm|[p]ointwise_ranking_attack_openai' ; then
    :
else
    echo "  none"
fi
echo

echo "Queued batch jobs:"
if command -v atq >/dev/null 2>&1; then
    atq 2>/dev/null || true
else
    echo "  atq is unavailable on this server"
fi
echo

printf '%-9s %-24s %-9s %-10s %12s %12s %s\n' \
    'state' 'dataset' 'attack' 'prompt' 'clean' 'attacked' 'checkpoint'
printf '%s\n' '------------------------------------------------------------------------------------------------'

found=0
while IFS= read -r -d '' checkpoint; do
    found=1
    python - "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    state = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    print(f"CORRUPT   {'-':24} {'-':9} {'-':10} {'-':>12} {'-':>12} {path}")
    print(f"  {error}")
    raise SystemExit

fingerprint = state.get("fingerprint", {})
phases = state.get("phases", {})
target = fingerprint.get("num_passages", "?")
clean = f"{len(phases.get('clean', {}))}/{target}"
attacked = f"{len(phases.get('attacked', {}))}/{target}"
complete = bool(state.get("complete"))
state_label = "COMPLETE" if complete else "RESUMABLE"
dataset = fingerprint.get("dataset_name", "?").replace("msmarco-passage/", "")
attack = fingerprint.get("attack_type", "?")
prompt = fingerprint.get("prompt_mode", "?")
print(f"{state_label:<9} {dataset:<24} {attack:<9} {prompt:<10} {clean:>12} {attacked:>12} {path.name}")
PY
done < <(find "$RUN_DIR" -type f -name '*.checkpoint.json' -print0 | sort -z)

if [ "$found" -eq 0 ]; then
    echo "No pointwise checkpoints found yet."
fi

echo
echo "Recent failures (last 25 matching log lines):"
if grep -R -n -E 'Traceback|Error:|Exception|FAILED|ExpiredToken' \
    "$RUN_DIR"/run_*.log 2>/dev/null | tail -n 25; then
    :
else
    echo "  none found"
fi
