#!/usr/bin/env bash
# Run Qwen3-4B Filter QI with a reusable shared passage cache.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
NUM_SAMPLES="${NUM_SAMPLES:-4096}"
N_JOBS="${N_JOBS:-1}"
FILTER_MAX_TOKENS="${FILTER_MAX_TOKENS:-4000}"
CACHE_DIR="${FILTER_CACHE_DIR:-LLM_prompt_attack/outputs/filter_cache}"
OUT="LLM_prompt_attack/outputs/Qwen3-4B_filter_qi_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
for YEAR in 2019 2020; do
  for SCHEME in pairwise setwise listwise; do
    if [ "$SCHEME" = pairwise ]; then
      SAMPLES=(--num_pairs "$NUM_SAMPLES" --pos_rel 3 --neg_rel 0)
    else
      SAMPLES=(--num_sets "$NUM_SAMPLES" --set_size 4)
    fi
    "$PYTHON" "LLM_prompt_attack/${SCHEME}_ranking_attack_openai.py" \
      --provider openai --base_url "$BASE_URL" \
      --model_name Qwen3-4B --tokenizer_model Qwen/Qwen3-4B \
      --filter_model Qwen3-4B --filter_provider openai \
      --filter_base_url "$BASE_URL" --filter_max_tokens "$FILTER_MAX_TOKENS" \
      --filter_cache_dir "$CACHE_DIR" \
      --dataset_name "msmarco-passage/trec-dl-${YEAR}" \
      "${SAMPLES[@]}" --seed 42 --n_jobs "$N_JOBS" \
      --attack_type qi --attack_position back --prompt_mode filter_qi \
      --result_json_path "$OUT/result_${YEAR}_${SCHEME}.jsonl" \
      --detailed_results "$OUT/detail_${YEAR}_${SCHEME}.json" \
      2>&1 | tee "$OUT/run_${YEAR}_${SCHEME}.log"
    "$PYTHON" Results/update_attack_outcomes.py
  done
done
