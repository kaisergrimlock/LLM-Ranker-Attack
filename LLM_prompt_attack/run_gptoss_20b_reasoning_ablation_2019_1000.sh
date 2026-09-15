#!/usr/bin/env bash
# GPT-OSS-20B pointwise reasoning-effort ablation on TREC-DL 2019 (1000 passages).
set -uo pipefail
RESEARCH_ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/$USER}"
PROJECT="${PROJECT:-$RESEARCH_ROOT/LLM-Ranker-Attack}"
PYTHON="${PYTHON:-$RESEARCH_ROOT/environments/llama3-8b/bin/python}"
AWS_REGION="${AWS_REGION:-ap-southeast-2}"; AWS_PROFILE="${AWS_PROFILE:-default}"
NUM_PASSAGES="${NUM_PASSAGES:-1000}"; N_JOBS="${N_JOBS:-2}"
RUN_ID="${RUN_ID:-gptoss-20b_reasoning_ablation_2019_1000}"
RUN_DIR="${RUN_DIR:-LLM_prompt_attack/outputs/reasoning_ablation/$RUN_ID}"
MODEL_NAME="openai.gpt-oss-20b-1:0"; DATASET="msmarco-passage/trec-dl-2019"
[ -x "$PYTHON" ] && [ -d "$PROJECT/LLM_prompt_attack" ] || { echo "Invalid Python/project path" >&2; exit 1; }
export AWS_PROFILE AWS_REGION BEDROCK_MAX_TOKENS="${BEDROCK_MAX_TOKENS:-512}"
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$RESEARCH_ROOT/ir_datasets}" HF_HOME="${HF_HOME:-$RESEARCH_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
aws sts get-caller-identity --region "$AWS_REGION" >/dev/null || { echo "AWS credentials unavailable" >&2; exit 1; }
cd "$PROJECT"; mkdir -p "$RUN_DIR"
[ -f "$RUN_DIR/status.tsv" ] || printf 'reasoning_effort\tdataset\tattack\tprompt\tstatus\n' > "$RUN_DIR/status.tsv"
cp LLM_prompt_attack/prompts.py "$RUN_DIR/prompts.py"
FAILED=0
for REASONING_EFFORT in low medium high; do
  export GPT_OSS_REASONING_EFFORT="$REASONING_EFFORT"
  for ATTACK in so sd; do
    for PROMPT_MODE in standard defense; do
      TAG="GPT-OSS-20B_trec-dl-2019_pointwise_${ATTACK}_${PROMPT_MODE}_reasoning_${REASONING_EFFORT}_n${NUM_PASSAGES}"
      RESULT_PATH="$RUN_DIR/result_${TAG}.jsonl"; DETAIL_PATH="$RUN_DIR/detail_${TAG}.json"; CHECKPOINT_PATH="$RUN_DIR/${TAG}.checkpoint.json"; RESUME_ARGS=()
      if [ -s "$RESULT_PATH" ] && [ -s "$DETAIL_PATH" ] && [ -s "$CHECKPOINT_PATH" ] && "$PYTHON" -c 'import json,sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text()).get("complete") is True)' "$CHECKPOINT_PATH" | grep -qx True; then
        echo "Skipping completed $TAG"; printf '%s\t%s\t%s\t%s\tskipped-complete\n' "$REASONING_EFFORT" "$DATASET" "$ATTACK" "$PROMPT_MODE" >> "$RUN_DIR/status.tsv"; continue
      fi
      [ -f "$CHECKPOINT_PATH" ] && RESUME_ARGS=(--resume); echo "Running $TAG"; set +u
      if "$PYTHON" LLM_prompt_attack/pointwise_ranking_attack_openai.py --provider amazon-bedrock --aws_region "$AWS_REGION" --model_name "$MODEL_NAME" --dataset_name "$DATASET" --num_passages "$NUM_PASSAGES" --seed 42 --n_jobs "$N_JOBS" --attack_type "$ATTACK" --attack_position back --prompt_mode "$PROMPT_MODE" --result_json_path "$RESULT_PATH" --detailed_results "$DETAIL_PATH" --checkpoint_path "$CHECKPOINT_PATH" --checkpoint_batch_size 32 "${RESUME_ARGS[@]}" 2>&1 | tee "$RUN_DIR/run_${TAG}.log"; then STATUS=0; else STATUS=$?; FAILED=1; fi
      set -u; printf '%s\t%s\t%s\t%s\t%s\n' "$REASONING_EFFORT" "$DATASET" "$ATTACK" "$PROMPT_MODE" "$STATUS" >> "$RUN_DIR/status.tsv"
    done
  done
done
echo "Finished. Results: $RUN_DIR"; exit "$FAILED"
