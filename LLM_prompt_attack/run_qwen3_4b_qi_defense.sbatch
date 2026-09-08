#!/usr/bin/env bash
#SBATCH --job-name=qwen3-4b-qi-defense
#SBATCH --partition=SCT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=qwen3-4b-qi-defense-%j.log

# Normal query injection: all paradigms, both TREC-DL years, defense_qi only.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch from the repository root.}"
RESEARCH_ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/$USER}"
ENV_FILE="${ENV_FILE:-$RESEARCH_ROOT/qwen3-ranking-env.sh}"
PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
source "$ENV_FILE"
cd "$PROJECT"
case "$(pwd -P)/" in
  "$RESEARCH_ROOT/"*) ;;
  *) echo "Repository must be under $RESEARCH_ROOT" >&2; exit 1 ;;
esac

PYTHON="${PYTHON:-python}"
NUM_SAMPLES="${NUM_SAMPLES:-4096}"
N_JOBS="${N_JOBS:-1}"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
RUN_DIR="LLM_prompt_attack/outputs/Qwen3-4B_qi_defense_qi_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd -P)"
SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Record hardware and reject GPUs unsupported by the installed vLLM stack.
hostname > "$RUN_DIR/environment.log"
nvidia-smi >> "$RUN_DIR/environment.log"
getconf GNU_LIBC_VERSION >> "$RUN_DIR/environment.log"
"$PYTHON" -m pip freeze > "$RUN_DIR/packages.txt"
git rev-parse HEAD > "$RUN_DIR/source_revision.txt"
cp LLM_prompt_attack/prompts.py "$RUN_DIR/prompts.py"
cat > "$RUN_DIR/check_gpu.py" <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
capability = torch.cuda.get_device_capability(0)
print(torch.cuda.get_device_name(0), capability)
assert capability >= (7, 0), "vLLM requires capability >= 7.0; P100 is unsupported"
PY
"$PYTHON" "$RUN_DIR/check_gpu.py" >> "$RUN_DIR/environment.log"
if curl -fsS --max-time 2 "$BASE_URL/models" >/dev/null 2>&1; then
  echo "Port $PORT already serves a model; set PORT to an unused port." >&2
  exit 1
fi

# The ranking client sends enable_thinking=false on local Qwen requests.
"$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name Qwen3-4B \
  --host 127.0.0.1 --port "$PORT" --dtype half \
  --max-model-len 8192 --max-num-seqs 2 --gpu-memory-utilization 0.90 \
  > "$RUN_DIR/vllm.log" 2>&1 &
SERVER_PID=$!
READY=0
for ((attempt=0; attempt<180; attempt++)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 60 "$RUN_DIR/vllm.log" >&2
    exit 1
  fi
  if curl -fsS --max-time 2 "$BASE_URL/models" > "$RUN_DIR/models.json"; then
    READY=1
    break
  fi
  sleep 5
done
if [ "$READY" -ne 1 ]; then
  echo "vLLM readiness timed out; see $RUN_DIR/vllm.log" >&2
  exit 1
fi

for DATASET in msmarco-passage/trec-dl-2019 msmarco-passage/trec-dl-2020; do
  for SCHEME in pairwise setwise listwise; do
    TAG="${DATASET##*/}_${SCHEME}_qi_defense_qi"
    if [ "$SCHEME" = pairwise ]; then
      SAMPLE_ARGS=(--pos_rel 3 --neg_rel 0 --num_pairs "$NUM_SAMPLES")
    else
      SAMPLE_ARGS=(--num_sets "$NUM_SAMPLES" --set_size 4)
    fi
    "$PYTHON" "LLM_prompt_attack/${SCHEME}_ranking_attack_openai.py" \
      --provider openai --base_url "$BASE_URL" \
      --model_name Qwen3-4B --tokenizer_model Qwen/Qwen3-4B \
      --dataset_name "$DATASET" --seed 42 --n_jobs "$N_JOBS" \
      --attack_type qi --attack_position back --prompt_mode defense_qi \
      "${SAMPLE_ARGS[@]}" \
      --result_json_path "$RUN_DIR/result_${TAG}.jsonl" \
      --detailed_results "$RUN_DIR/detail_${TAG}.json" \
      2>&1 | tee "$RUN_DIR/run_${TAG}.log"
    "$PYTHON" Results/update_attack_outcomes.py
  done
done
