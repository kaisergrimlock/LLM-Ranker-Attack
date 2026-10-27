#!/usr/bin/env bash
#SBATCH --job-name=autojev-rerank
#SBATCH --partition=SEG
#SBATCH --qos=hig_gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

source /opt/home/s3891987/miniforge3/etc/profile.d/conda.sh

ROOT=/opt/home/s3891987/projects
AUTOJEV_ROOT="$ROOT/autojev"
RERANK_ROOT="$ROOT/LLM-Ranker-Attack/LLM_re_ranker"
CHECKPOINT="$AUTOJEV_ROOT/checkpoints/selected"
RUN_PATH="${RUN_PATH:-$RERANK_ROOT/outputs/bm25-dl19-judged43-depth1000.txt}"
SAVE_PATH="${SAVE_PATH:-$RERANK_ROOT/outputs/autojev-dl19-smoke.txt}"
MAX_QUERIES="${MAX_QUERIES:-1}"
HITS="${HITS:-10}"
METHOD="${METHOD:-yes_no}"

mkdir -p "$RERANK_ROOT/slurm/logs" "$(dirname "$SAVE_PATH")"

conda activate autojev
cd "$AUTOJEV_ROOT"
AUTOJEV_CHECKPOINT="$CHECKPOINT" autojev-serve \
  > "$RERANK_ROOT/slurm/logs/autojev-${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for attempt in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    cat "$RERANK_ROOT/slurm/logs/autojev-${SLURM_JOB_ID}.log"
    exit 1
  fi
  sleep 2
done

curl -fsS http://127.0.0.1:8000/health

conda activate llm-ranker-attack
cd "$RERANK_ROOT"
PYTHONPATH=. python run_attack.py run \
  --provider autojev \
  --model_name_or_path autojev \
  --autojev_endpoint http://127.0.0.1:8000/v1/systemone \
  --run_path "$RUN_PATH" \
  --save_path "$SAVE_PATH" \
  --ir_dataset_name msmarco-passage/trec-dl-2019 \
  --hits "$HITS" \
  --max_queries "$MAX_QUERIES" \
  --query_length 32 \
  --passage_length 128 \
  pointwise --method "$METHOD"
