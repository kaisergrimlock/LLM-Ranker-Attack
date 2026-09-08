#!/usr/bin/env bash
# Direct SEG V100 entry point; GPU availability is checked by the shared job.
set -euo pipefail
exec bash "$(dirname "$0")/run_qwen3_4b_qi_defense.sbatch" "$@"
