#!/usr/bin/env bash
# Re-run the missing GPT-OSS-20B high-reasoning ablation evaluations.
# Missing combinations: pointwise/listwise x DOH/DCH x default/defense.
# Re-running this script resumes completed work from its checkpoint files.

set -uo pipefail

ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/$USER}"
PROJECT="${PROJECT:-$ROOT/LLM-Ranker-Attack}"
PYTHON="${PYTHON:-$ROOT/environments/llama3-8b/bin/python}"
REGION="${AWS_REGION:-ap-southeast-2}"
OUT="${RUN_DIR:-$PROJECT/LLM_prompt_attack/outputs/reasoning_ablation/gptoss_all_2019_1000}"
MODEL="openai.gpt-oss-20b-1:0"
DATASET="msmarco-passage/trec-dl-2019"
N="${NUM_PASSAGES:-1000}"
N_JOBS="${N_JOBS:-2}"

cd "$PROJECT" || exit 1
mkdir -p "$OUT"
export AWS_REGION="$REGION"
export GPT_OSS_REASONING_EFFORT=high
export BEDROCK_MAX_TOKENS="${GPT_OSS_BEDROCK_MAX_TOKENS:-${BEDROCK_MAX_TOKENS:-512}}"
unset QWEN_THINKING_MODE

FAILED=0
for PARADIGM in pointwise listwise; do
  for ATTACK in so sd; do
    for PROMPT in standard defense; do
      TAG="GPT-OSS-20B_${PARADIGM}_2019_${ATTACK}_${PROMPT}_high_n${N}"
      BASE="$OUT/$TAG"
      CHECKPOINT="$BASE.checkpoint.json"
      RESUME_ARGS=()
      [ -f "$CHECKPOINT" ] && RESUME_ARGS=(--resume)

      if [ -f "$CHECKPOINT" ] && grep -q '"complete": true' "$CHECKPOINT"; then
        echo "Skipping completed $TAG"
        continue
      fi

      case "$PARADIGM" in
        pointwise)
          SCRIPT=pointwise_ranking_attack_openai.py
          SIZE=(--num_passages "$N")
          ;;
        listwise)
          SCRIPT=listwise_ranking_attack_openai.py
          SIZE=(--num_sets "$N" --set_size 4)
          ;;
      esac

      echo "Running $TAG (region=$REGION)"
      set +u
      "$PYTHON" "LLM_prompt_attack/$SCRIPT" \
        --provider amazon-bedrock \
        --aws_region "$REGION" \
        --model_name "$MODEL" \
        --dataset_name "$DATASET" \
        "${SIZE[@]}" \
        --seed 42 \
        --n_jobs "$N_JOBS" \
        --attack_type "$ATTACK" \
        --attack_position back \
        --prompt_mode "$PROMPT" \
        --result_json_path "$BASE.jsonl" \
        --detailed_results "$BASE.json" \
        --checkpoint_path "$CHECKPOINT" \
        --checkpoint_batch_size 32 \
        "${RESUME_ARGS[@]}" \
        2>&1 | tee "$BASE.log"
      STATUS=${PIPESTATUS[0]}
      set -u
      [ "$STATUS" -eq 0 ] || FAILED=1
    done
  done
done

echo "Rebuilding ablation_study.csv"
"$PYTHON" Results/update_ablation_study.py || FAILED=1
exit "$FAILED"
