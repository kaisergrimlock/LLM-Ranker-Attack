#!/usr/bin/env bash
# GPT-OSS-20B and Qwen3-32B query injection with grade-3/grade-2 candidates.
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
AWS_REGION="${AWS_REGION:-ap-southeast-2}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/close_attack/${TIMESTAMP}"
mkdir -p "$OUT_DIR"

for MODEL_TAG in GPT-OSS-20B Qwen3-32B; do
  case "$MODEL_TAG" in
    GPT-OSS-20B)
      MODEL_NAME="openai.gpt-oss-20b-1:0"
      TOKENIZER_MODEL="openai/gpt-oss-20b"
      ;;
    Qwen3-32B)
      MODEL_NAME="qwen.qwen3-32b-v1:0"
      TOKENIZER_MODEL="Qwen/Qwen3-32B"
      ;;
  esac
  for DATASET in msmarco-passage/trec-dl-2019 msmarco-passage/trec-dl-2020; do
    for SCHEME in pairwise setwise listwise; do
      for PROMPT_MODE in standard defense_qi; do
        TAG="${MODEL_TAG}_${DATASET##*/}_${SCHEME}_qi_${PROMPT_MODE}"
        if [ "$SCHEME" = pairwise ]; then
          SAMPLE_ARGS=(--pos_rel 3 --neg_rel 2 --num_pairs "$NUM_SAMPLES" --close_attack)
        else
          SAMPLE_ARGS=(--num_sets "$NUM_SAMPLES" --set_size 4 --close_attack)
        fi
        "$PYTHON" "LLM_prompt_attack/${SCHEME}_ranking_attack_openai.py" \
          --provider amazon-bedrock --aws_region "$AWS_REGION" \
          --model_name "$MODEL_NAME" \
          --tokenizer_model "$TOKENIZER_MODEL" \
          --dataset_name "$DATASET" --seed 42 --n_jobs "$N_JOBS" \
          --attack_type qi --attack_position back --prompt_mode "$PROMPT_MODE" \
          "${SAMPLE_ARGS[@]}" \
          --result_json_path "$OUT_DIR/result_${TAG}.jsonl" \
          --detailed_results "$OUT_DIR/detail_${TAG}.json" \
          2>&1 | tee "$OUT_DIR/run_${TAG}.log"
        "$PYTHON" Results/update_close_attack_table.py
      done
    done
  done
done
