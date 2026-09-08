#!/usr/bin/env bash
# Reuse the existing model server for evaluation on the direct SEG host.
set -euo pipefail
exec bash "$(dirname "$0")/evaluate_qwen3_4b_qi_defense.sh" "$@"
