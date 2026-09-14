#!/usr/bin/env bash
# Compare Qwen3-32B pointwise attack/defense effectiveness with thinking
# explicitly off versus on, using a smaller deterministic TREC-DL-2019 subset.
#
# Run on the server with:
#   bash LLM_prompt_attack/run_qwen3_32b_thinking_ablation_2019_1000.sh
#
# Results are intentionally written outside the main pointwise output folder:
#   LLM_prompt_attack/outputs/thinking_ablation/qwen3-32b_2019_1000

set -uo pipefail

RESEARCH_ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/$USER}"
PROJECT="${PROJECT:-$RESEARCH_ROOT/LLM-Ranker-Attack}"
PYTHON="${PYTHON:-$RESEARCH_ROOT/environments/qwen3-vllm/bin/python}"
AWS_REGION="${AWS_REGION:-us-west-2}"
AWS_PROFILE="${AWS_PROFILE:-default}"

MODEL_TAG="Qwen3-32B"
MODEL_NAME="qwen.qwen3-32b-v1:0"
TOKENIZER_MODEL="Qwen/Qwen3-32B"
DATASET="msmarco-passage/trec-dl-2019"
YEAR="2019"
NUM_PASSAGES="${NUM_PASSAGES:-1000}"
N_JOBS="${N_JOBS:-2}"
CHECKPOINT_BATCH_SIZE="${CHECKPOINT_BATCH_SIZE:-32}"
RUN_ID="${RUN_ID:-qwen3-32b_2019_1000}"
RUN_DIR="${RUN_DIR:-LLM_prompt_attack/outputs/thinking_ablation/$RUN_ID}"

if [ ! -x "$PYTHON" ]; then
    echo "Python executable not found: $PYTHON" >&2
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
export BEDROCK_MAX_TOKENS="${BEDROCK_MAX_TOKENS:-512}"

if ! aws sts get-caller-identity --region "$AWS_REGION" >/dev/null; then
    echo "AWS credentials are unavailable. Refresh them, then rerun this job:" >&2
    echo "  aws sso login --profile $AWS_PROFILE --use-device-code" >&2
    exit 1
fi

cd "$PROJECT"
mkdir -p "$RUN_DIR"
if [ ! -f "$RUN_DIR/status.tsv" ]; then
    printf 'thinking_mode\tmodel\tdataset\tattack\tprompt\tstatus\n' > "$RUN_DIR/status.tsv"
fi

{
    echo "started=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "project=$PROJECT"
    echo "python=$PYTHON"
    echo "aws_region=$AWS_REGION"
    echo "dataset=$DATASET"
    echo "num_passages=$NUM_PASSAGES"
    echo "n_jobs=$N_JOBS"
    echo "run_dir=$RUN_DIR"
    echo "source_revision=$(git rev-parse HEAD)"
    aws sts get-caller-identity --region "$AWS_REGION"
} > "$RUN_DIR/environment.log"
cp LLM_prompt_attack/prompts.py "$RUN_DIR/prompts.py"

FAILED=0
for THINKING_MODE in off on; do
    export QWEN_THINKING_MODE="$THINKING_MODE"
    for ATTACK in so sd; do
        for PROMPT_MODE in standard defense; do
            TAG="${MODEL_TAG}_trec-dl-${YEAR}_pointwise_${ATTACK}_${PROMPT_MODE}_thinking_${THINKING_MODE}_n${NUM_PASSAGES}"
            RESULT_PATH="$RUN_DIR/result_${TAG}.jsonl"
            DETAIL_PATH="$RUN_DIR/detail_${TAG}.json"
            CHECKPOINT_PATH="$RUN_DIR/${TAG}.checkpoint.json"
            RESUME_ARGS=()

            if [ -s "$RESULT_PATH" ] && [ -s "$DETAIL_PATH" ] && [ -s "$CHECKPOINT_PATH" ] && \
                "$PYTHON" -c 'import json, sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("complete") is True)' "$CHECKPOINT_PATH" | grep -qx True; then
                echo "Skipping completed $TAG"
                printf '%s\t%s\t%s\t%s\t%s\tskipped-complete\n' \
                    "$THINKING_MODE" "$MODEL_TAG" "$DATASET" "$ATTACK" "$PROMPT_MODE" \
                    >> "$RUN_DIR/status.tsv"
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
                --thinking_mode "$THINKING_MODE" \
                --result_json_path "$RESULT_PATH" \
                --detailed_results "$DETAIL_PATH" \
                --checkpoint_path "$CHECKPOINT_PATH" \
                --checkpoint_batch_size "$CHECKPOINT_BATCH_SIZE" \
                "${RESUME_ARGS[@]}" \
                2>&1 | tee "$RUN_DIR/run_${TAG}.log"; then
                STATUS=0
            else
                STATUS=$?
                FAILED=1
            fi
            set -u

            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$THINKING_MODE" "$MODEL_TAG" "$DATASET" "$ATTACK" "$PROMPT_MODE" "$STATUS" \
                >> "$RUN_DIR/status.tsv"

            if [ "$STATUS" -ne 0 ] && grep -q 'ExpiredTokenException' "$RUN_DIR/run_${TAG}.log"; then
                echo "AWS credentials expired. Refresh them and rerun this job to resume." >&2
                exit "$STATUS"
            fi
        done
    done
done

echo "Finished. Results directory: $RUN_DIR"
echo "Per-run status: $RUN_DIR/status.tsv"
exit "$FAILED"
