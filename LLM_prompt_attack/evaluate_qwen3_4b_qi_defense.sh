#!/usr/bin/env bash
# Evaluate normal query injection against an already running Qwen3-4B server.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"
NUM_SAMPLES="${NUM_SAMPLES:-4096}"
N_JOBS="${N_JOBS:-1}"
OUT="LLM_prompt_attack/outputs/Qwen3-4B_qi_defense_qi_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
curl -fsS --max-time 10 "$BASE_URL/models" > "$OUT/models.json"
cp LLM_prompt_attack/prompts.py "$OUT/prompts.py"

for DATASET in msmarco-passage/trec-dl-2019 msmarco-passage/trec-dl-2020; do
  for SCHEME in pairwise setwise listwise; do
    TAG="${DATASET##*/}_${SCHEME}"
    if [ "$SCHEME" = pairwise ]; then
      SAMPLES=(--num_pairs "$NUM_SAMPLES" --pos_rel 3 --neg_rel 0)
    else
      SAMPLES=(--num_sets "$NUM_SAMPLES" --set_size 4)
    fi
    "$PYTHON" "LLM_prompt_attack/${SCHEME}_ranking_attack_openai.py" \
      --provider openai --base_url "$BASE_URL" \
      --model_name "$MODEL_NAME" --tokenizer_model Qwen/Qwen3-4B \
      --dataset_name "$DATASET" "${SAMPLES[@]}" --seed 42 --n_jobs "$N_JOBS" \
      --attack_type qi --attack_position back --prompt_mode defense_qi \
      --result_json_path "$OUT/result_${TAG}.jsonl" \
      --detailed_results "$OUT/detail_${TAG}.json" \
      2>&1 | tee "$OUT/run_${TAG}.log"
    "$PYTHON" Results/update_attack_outcomes.py
  done
done
