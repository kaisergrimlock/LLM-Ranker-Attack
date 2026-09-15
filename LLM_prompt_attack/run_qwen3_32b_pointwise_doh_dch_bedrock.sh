#!/usr/bin/env bash
# Run all Qwen3-32B Bedrock pointwise DOH/DCH evaluations for TREC-DL 2019/2020.
# Re-run the same command after an interruption; completed configurations are
# skipped and partial configurations resume from their checkpoints.

set -uo pipefail

RESEARCH_ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/$USER}"
PROJECT="${PROJECT:-$RESEARCH_ROOT/LLM-Ranker-Attack}"
PYTHON="${PYTHON:-$RESEARCH_ROOT/environments/llama3-8b/bin/python}"
AWS_REGION="${AWS_REGION:-ap-southeast-2}"
AWS_PROFILE="${AWS_PROFILE:-default}"
NUM_PASSAGES="${NUM_PASSAGES:-4096}"
N_JOBS="${N_JOBS:-2}"
RUN_ID="${RUN_ID:-pointwise_new}"

if [ ! -x "$PYTHON" ]; then
    echo "Python executable not found: $PYTHON" >&2
    echo "Set PYTHON to the active server environment's interpreter." >&2
    exit 1
fi
if [ ! -d "$PROJECT/LLM_prompt_attack" ]; then
    echo "Project directory is invalid: $PROJECT" >&2
    exit 1
fi

export AWS_PROFILE AWS_REGION
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$RESEARCH_ROOT/ir_datasets}"
export HF_HOME="${HF_HOME:-$RESEARCH_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export BEDROCK_MAX_TOKENS="${BEDROCK_MAX_TOKENS:-1024}"

if ! aws sts get-caller-identity --region "$AWS_REGION" >/dev/null; then
    echo "AWS credentials are unavailable. Refresh them, then rerun this script:" >&2
    echo "  aws sso login --profile $AWS_PROFILE --use-device-code" >&2
    exit 1
fi

cd "$PROJECT"
RUN_DIR="LLM_prompt_attack/outputs/$RUN_ID"
mkdir -p "$RUN_DIR"
if [ ! -f "$RUN_DIR/status.tsv" ]; then
    printf 'model\tdataset\tattack\tprompt\tstatus\n' > "$RUN_DIR/status.tsv"
fi
hostname > "$RUN_DIR/environment.log"
date --iso-8601=seconds >> "$RUN_DIR/environment.log"
aws sts get-caller-identity --region "$AWS_REGION" >> "$RUN_DIR/environment.log"
git rev-parse HEAD > "$RUN_DIR/source_revision.txt"
cp LLM_prompt_attack/prompts.py "$RUN_DIR/prompts.py"

FAILED=0
MODEL_TAG="Qwen3-32B"
MODEL_NAME="qwen.qwen3-32b-v1:0"
TOKENIZER_MODEL="Qwen/Qwen3-32B"

for YEAR in 2019 2020; do
    DATASET="msmarco-passage/trec-dl-$YEAR"
    for ATTACK in so sd; do
        for PROMPT_MODE in standard defense; do
            TAG="${MODEL_TAG}_trec-dl-${YEAR}_pointwise_${ATTACK}_${PROMPT_MODE}"
            RESULT_PATH="$RUN_DIR/result_${TAG}.jsonl"
            DETAIL_PATH="$RUN_DIR/detail_${TAG}.json"
            CHECKPOINT_PATH="$RUN_DIR/${TAG}.checkpoint.json"
            RESUME_ARGS=()

            if [ -s "$RESULT_PATH" ] && [ -s "$DETAIL_PATH" ] && \
                [ -f "$CHECKPOINT_PATH" ] && \
                "$PYTHON" -c 'import json, sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("complete") is True)' "$CHECKPOINT_PATH" | grep -qx True; then
                echo "Skipping completed $TAG"
                printf '%s\t%s\t%s\t%s\tskipped-complete\n' \
                    "$MODEL_TAG" "$DATASET" "$ATTACK" "$PROMPT_MODE" >> "$RUN_DIR/status.tsv"
                continue
            fi
            if [ -f "$CHECKPOINT_PATH" ]; then
                RESUME_ARGS=(--resume)
            fi

            echo "Running $TAG"
            set +u
            if "$PYTHON" LLM_prompt_attack/pointwise_ranking_attack_openai.py \
                --provider amazon-bedrock \
                --aws_region "$AWS_REGION" \
                --model_name "$MODEL_NAME" \
                --tokenizer_model "$TOKENIZER_MODEL" \
                --dataset_name "$DATASET" \
                --num_passages "$NUM_PASSAGES" \
                --seed 42 \
                --n_jobs "$N_JOBS" \
                --attack_type "$ATTACK" \
                --attack_position back \
                --prompt_mode "$PROMPT_MODE" \
                --result_json_path "$RESULT_PATH" \
                --detailed_results "$DETAIL_PATH" \
                --checkpoint_path "$CHECKPOINT_PATH" \
                --checkpoint_batch_size 32 \
                "${RESUME_ARGS[@]}" \
                2>&1 | tee "$RUN_DIR/run_${TAG}.log"; then
                STATUS=0
            else
                STATUS=$?
                FAILED=1
            fi
            set -u
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$MODEL_TAG" "$DATASET" "$ATTACK" "$PROMPT_MODE" "$STATUS" >> "$RUN_DIR/status.tsv"

            if [ "$STATUS" -ne 0 ] && grep -q 'ExpiredTokenException' "$RUN_DIR/run_${TAG}.log"; then
                echo "AWS credentials expired. Refresh them and rerun to resume." >&2
                exit "$STATUS"
            fi
        done
    done
done

echo "Finished. Per-run status: $RUN_DIR/status.tsv"
exit "$FAILED"
