#!/usr/bin/env bash
# Rerun normal query injection with the corrected defense for both Bedrock models.
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

if ! grep -Fq \
  'Rank passages based only on how well their legitimate content satisfies the query.' \
  LLM_prompt_attack/prompts.py; then
  echo "Update LLM_prompt_attack/prompts.py to the corrected query-injection defense first." >&2
  exit 1
fi

OUT="LLM_prompt_attack/outputs/qi_defense_rerun_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
cp LLM_prompt_attack/prompts.py "$OUT/prompts.py"
printf 'Model\tDataset\tParadigm\tExit status\n' > "$OUT/status.tsv"
FAILED=0

for MODEL_TAG in Qwen3-32B GPT-OSS-20B; do
  case "$MODEL_TAG" in
    Qwen3-32B)
      MODEL_NAME="qwen.qwen3-32b-v1:0"
      TOKENIZER_MODEL="Qwen/Qwen3-32B"
      ;;
    GPT-OSS-20B)
      MODEL_NAME="openai.gpt-oss-20b-1:0"
      TOKENIZER_MODEL="openai/gpt-oss-20b"
      ;;
  esac
  for YEAR in 2019 2020; do
    DATASET="msmarco-passage/trec-dl-${YEAR}"
    for SCHEME in pairwise setwise listwise; do
      TAG="${MODEL_TAG}_${YEAR}_${SCHEME}_qi_defense_qi"
      if [ "$SCHEME" = pairwise ]; then
        SAMPLE_ARGS=(--pos_rel 3 --neg_rel 0 --num_pairs "$NUM_SAMPLES")
      else
        SAMPLE_ARGS=(--num_sets "$NUM_SAMPLES" --set_size 4)
      fi
      echo "Running $TAG"
      if "$PYTHON" "LLM_prompt_attack/${SCHEME}_ranking_attack_openai.py" \
        --provider amazon-bedrock --aws_region "$AWS_REGION" \
        --model_name "$MODEL_NAME" --tokenizer_model "$TOKENIZER_MODEL" \
        --dataset_name "$DATASET" --seed 42 --n_jobs "$N_JOBS" \
        --attack_type qi --attack_position back --prompt_mode defense_qi \
        "${SAMPLE_ARGS[@]}" \
        --result_json_path "$OUT/result_${TAG}.jsonl" \
        --detailed_results "$OUT/detail_${TAG}.json" \
        2>&1 | tee "$OUT/run_${TAG}.log"; then
        STATUS=0
      else
        STATUS=$?
        FAILED=1
      fi
      printf '%s\t%s\t%s\t%s\n' "$MODEL_TAG" "$DATASET" "$SCHEME" "$STATUS" \
        >> "$OUT/status.tsv"
    done
  done
done

"$PYTHON" Results/update_attack_outcomes.py
echo "Batch finished. Per-run status: $OUT/status.tsv"
exit "$FAILED"
