#!/usr/bin/env bash
# Run TREC-DL 2020 listwise QI modes concurrently through Bedrock.
# The three modes use distinct result files. Filter QI uses a shared,
# content-keyed cache that is safe to reuse across later matching runs.

set -uo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
AWS_REGION="${AWS_REGION:-us-east-1}"
N_JOBS_PER_RUN="${N_JOBS_PER_RUN:-2}"
# Bedrock Llama 3 8B allows at most 2,048 output tokens per Converse request.
# Filtering returns one cleaned passage, so 1,024 is sufficient and portable.
FILTER_MAX_TOKENS="${FILTER_MAX_TOKENS:-1024}"
FILTER_CACHE_DIR="${FILTER_CACHE_DIR:-LLM_prompt_attack/outputs/filter_cache}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="LLM_prompt_attack/outputs/Llama3-8B_trec-dl-2020_listwise_qi_${TIMESTAMP}_$$"

mkdir -p "$OUTPUT_DIR" "$FILTER_CACHE_DIR"

run_mode() {
  local mode="$1"
  local filter_args=()

  if [ "$mode" = "filter_qi" ]; then
    filter_args=(
      --filter_model meta.llama3-8b-instruct-v1:0
      --filter_provider amazon-bedrock
      --filter_aws_region "$AWS_REGION"
      --filter_max_tokens "$FILTER_MAX_TOKENS"
      --filter_cache_dir "$FILTER_CACHE_DIR"
    )
  fi

  "$PYTHON" LLM_prompt_attack/listwise_ranking_attack_openai.py \
    --provider amazon-bedrock \
    --aws_region "$AWS_REGION" \
    --model_name meta.llama3-8b-instruct-v1:0 \
    --tokenizer_model NousResearch/Meta-Llama-3-8B-Instruct \
    --dataset_name msmarco-passage/trec-dl-2020 \
    --num_sets 4096 \
    --set_size 4 \
    --seed 42 \
    --attack_type qi \
    --attack_position back \
    --prompt_mode "$mode" \
    --n_jobs "$N_JOBS_PER_RUN" \
    "${filter_args[@]}" \
    --result_json_path "$OUTPUT_DIR/result_Llama3-8B_2020_listwise_qi_${mode}.jsonl" \
    --detailed_results "$OUTPUT_DIR/detail_Llama3-8B_2020_listwise_qi_${mode}.json" \
    2>&1 | tee "$OUTPUT_DIR/run_Llama3-8B_2020_listwise_qi_${mode}.log"
}

MODES=(standard defense_qi filter_qi)
PIDS=()
for mode in "${MODES[@]}"; do
  run_mode "$mode" &
  PIDS+=("$!")
done

FAILED=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "${MODES[$index]}_STATUS=0"
  else
    echo "${MODES[$index]}_STATUS=$?"
    FAILED=1
  fi
done

"$PYTHON" Results/update_attack_outcomes.py

if [ "$FAILED" -eq 0 ]; then
  echo "All three QI modes completed. Results: $OUTPUT_DIR"
else
  echo "At least one mode failed. Inspect: $OUTPUT_DIR/run_*.log"
fi
