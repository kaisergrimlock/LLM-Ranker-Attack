#!/usr/bin/env bash
set -uo pipefail
ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/$USER}"
PROJECT="${PROJECT:-$ROOT/LLM-Ranker-Attack}"
PYTHON="${PYTHON:-$ROOT/environments/llama3-8b/bin/python}"
REGION="${AWS_REGION:-ap-southeast-2}"
OUT="${RUN_DIR:-LLM_prompt_attack/outputs/reasoning_ablation/gptoss_all_2019_1000}"
MODEL="openai.gpt-oss-20b-1:0"; DATA="msmarco-passage/trec-dl-2019"; N="${NUM_PASSAGES:-1000}"
cd "$PROJECT" || exit 1; mkdir -p "$OUT"
export AWS_REGION="$REGION"
export BEDROCK_MAX_TOKENS="${GPT_OSS_BEDROCK_MAX_TOKENS:-${BEDROCK_MAX_TOKENS:-512}}"
unset QWEN_THINKING_MODE
FAILED=0
for PARADIGM in pairwise setwise listwise; do
  for EFFORT in low medium high; do
    export GPT_OSS_REASONING_EFFORT="$EFFORT"
    for ATTACK in so sd; do
      for PROMPT in standard defense; do
        TAG="GPT-OSS-20B_${PARADIGM}_2019_${ATTACK}_${PROMPT}_${EFFORT}_n${N}"
        BASE="$OUT/$TAG"; CHECK="$BASE.checkpoint.json"; RESUME=()
        [ -f "$CHECK" ] && RESUME=(--resume)
        case "$PARADIGM" in
          pointwise) SCRIPT=pointwise_ranking_attack_openai.py; SIZE=(--num_passages "$N");;
          pairwise) SCRIPT=pairwise_ranking_attack_openai.py; SIZE=(--num_pairs "$N");;
          setwise|listwise) SCRIPT="${PARADIGM}_ranking_attack_openai.py"; SIZE=(--num_sets "$N");;
        esac
        if [ -f "$CHECK" ] && grep -q '"complete": true' "$CHECK"; then echo "Skipping completed $TAG"; continue; fi
        echo "Running $TAG (reasoning_effort=$GPT_OSS_REASONING_EFFORT, bedrock_max_tokens=$BEDROCK_MAX_TOKENS, checkpoint=$CHECK)"; set +u
        "$PYTHON" "LLM_prompt_attack/$SCRIPT" --provider amazon-bedrock --aws_region "$REGION" --model_name "$MODEL" --dataset_name "$DATA" "${SIZE[@]}" --seed 42 --n_jobs 2 --attack_type "$ATTACK" --attack_position back --prompt_mode "$PROMPT" --result_json_path "$BASE.jsonl" --detailed_results "$BASE.json" --checkpoint_path "$CHECK" --checkpoint_batch_size 32 "${RESUME[@]}" 2>&1 | tee "$BASE.log" || FAILED=1
        set -u
      done
    done
  done
done
exit "$FAILED"
