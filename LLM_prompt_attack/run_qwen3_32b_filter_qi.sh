#!/usr/bin/env bash
# Qwen3-32B passage filtering followed by standard ranking, through Bedrock.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${PYTHON:-}" ]; then
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) PYTHON=".venv/Scripts/python.exe" ;;
    *) PYTHON="python" ;;
  esac
fi
NUM_SAMPLES="${NUM_SAMPLES:-4096}"
N_JOBS="${N_JOBS:-4}"
FILTER_MAX_TOKENS="${FILTER_MAX_TOKENS:-8192}"
AWS_REGION="${AWS_REGION:-ap-southeast-2}"
OUT="LLM_prompt_attack/outputs/Qwen3-32B_filter_qi_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
cp LLM_prompt_attack/filter_defense.py "$OUT/filter_defense.py"
cp LLM_prompt_attack/prompts.py "$OUT/prompts.py"

for YEAR in 2019 2020; do
  for SCHEME in pairwise setwise listwise; do
    TAG="Qwen3-32B_${YEAR}_${SCHEME}_qi_filter_qi"
    if [ "$SCHEME" = pairwise ]; then
      SAMPLE_ARGS=(--pos_rel 3 --neg_rel 0 --num_pairs "$NUM_SAMPLES")
    else
      SAMPLE_ARGS=(--num_sets "$NUM_SAMPLES" --set_size 4)
    fi
    "$PYTHON" "LLM_prompt_attack/${SCHEME}_ranking_attack_openai.py" \
      --provider amazon-bedrock --aws_region "$AWS_REGION" \
      --model_name qwen.qwen3-32b-v1:0 --tokenizer_model Qwen/Qwen3-32B \
      --filter_model qwen.qwen3-32b-v1:0 --filter_provider amazon-bedrock \
      --filter_aws_region "$AWS_REGION" --filter_max_tokens "$FILTER_MAX_TOKENS" \
      --dataset_name "msmarco-passage/trec-dl-${YEAR}" \
      "${SAMPLE_ARGS[@]}" --seed 42 --n_jobs "$N_JOBS" \
      --attack_type qi --attack_position back --prompt_mode filter_qi \
      --result_json_path "$OUT/result_${TAG}.jsonl" \
      --detailed_results "$OUT/detail_${TAG}.json" \
      2>&1 | tee "$OUT/run_${TAG}.log"
    "$PYTHON" Results/update_attack_outcomes.py
  done
done
echo "Finished. Results: $OUT; summary: Results/attack_outcomes.csv"
