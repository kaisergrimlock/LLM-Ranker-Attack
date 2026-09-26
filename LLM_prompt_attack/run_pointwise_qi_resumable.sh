#!/usr/bin/env bash
# Run pointwise query-injection evaluations for TREC-DL 2019/2020.
# Completed configurations are skipped and interrupted configurations resume
# from their checkpoints when this script is run again.

set -uo pipefail

RESEARCH_ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/$USER}"
PROJECT="${PROJECT:-$RESEARCH_ROOT/LLM-Ranker-Attack}"
PYTHON="${PYTHON:-$RESEARCH_ROOT/environments/qwen3-vllm/bin/python}"
AWS_REGION="${AWS_REGION:-ap-southeast-2}"
AWS_PROFILE="${AWS_PROFILE:-default}"
NUM_PASSAGES="${NUM_PASSAGES:-4096}"
N_JOBS="${N_JOBS:-1}"
RUN_ID="${RUN_ID:-pointwise_query_injection_resumable}"

if [ ! -x "$PYTHON" ]; then
    echo "Python executable not found: $PYTHON" >&2
    echo "Set PYTHON to the active Bedrock environment's interpreter." >&2
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
export BEDROCK_MAX_TOKENS="${BEDROCK_MAX_TOKENS:-128}"

if ! aws sts get-caller-identity --region "$AWS_REGION" >/dev/null; then
    echo "AWS credentials are unavailable. Refresh them, then rerun this script." >&2
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
# Override with a space-separated subset, e.g. MODEL_TAGS="Qwen3-4B Qwen3-32B".
MODEL_TAGS="${MODEL_TAGS:-Qwen3-4B Qwen3-32B GPT-OSS-20B Llama-3-8B Llama3-70B}"

for MODEL_TAG in $MODEL_TAGS; do
    case "$MODEL_TAG" in
        Qwen3-4B)
            MODEL_NAME="qwen.qwen3-4b-v1:0"
            TOKENIZER_MODEL="Qwen/Qwen3-4B"
            ;;
        Qwen3-32B)
            MODEL_NAME="qwen.qwen3-32b-v1:0"
            TOKENIZER_MODEL="Qwen/Qwen3-32B"
            ;;
        GPT-OSS-20B)
            MODEL_NAME="openai.gpt-oss-20b-1:0"
            TOKENIZER_MODEL="openai/gpt-oss-20b"
            ;;
        Llama-3-8B)
            MODEL_NAME="meta.llama3-8b-instruct-v1:0"
            TOKENIZER_MODEL="NousResearch/Meta-Llama-3-8B-Instruct"
            ;;
        Llama3-70B)
            MODEL_NAME="meta.llama3-70b-instruct-v1:0"
            TOKENIZER_MODEL="NousResearch/Meta-Llama-3-70B-Instruct"
            ;;
        *)
            echo "Unknown MODEL_TAG=$MODEL_TAG; skipping" >&2
            continue
            ;;
    esac

    for YEAR in 2019 2020; do
        DATASET="msmarco-passage/trec-dl-$YEAR"
        for PROMPT_MODE in standard defense_qi; do
            TAG="${MODEL_TAG}_trec-dl-${YEAR}_pointwise_qi_${PROMPT_MODE}"
            RESULT_PATH="$RUN_DIR/result_${TAG}.jsonl"
            DETAIL_PATH="$RUN_DIR/detail_${TAG}.json"
            CHECKPOINT_PATH="$RUN_DIR/${TAG}.checkpoint.json"
            RESUME_ARGS=()

            if [ -s "$RESULT_PATH" ] && [ -s "$DETAIL_PATH" ] && [ -f "$CHECKPOINT_PATH" ] && \
                "$PYTHON" -c 'import json,sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("complete") is True)' "$CHECKPOINT_PATH" | grep -qx True; then
                echo "Skipping completed $TAG"
                printf '%s\t%s\tqi\t%s\tskipped-complete\n' "$MODEL_TAG" "$DATASET" "$PROMPT_MODE" >> "$RUN_DIR/status.tsv"
                continue
            fi
            if [ -f "$CHECKPOINT_PATH" ]; then
                RESUME_ARGS=(--resume)
            fi

            echo "Running $TAG (region=$AWS_REGION)"
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
                --attack_type qi \
                --attack_position back \
                --prompt_mode "$PROMPT_MODE" \
                --result_json_path "$RESULT_PATH" \
                --detailed_results "$DETAIL_PATH" \
                --checkpoint_path "$CHECKPOINT_PATH" \
                --checkpoint_batch_size 32 \
                "${RESUME_ARGS[@]}" 2>&1 | tee "$RUN_DIR/run_${TAG}.log"; then
                STATUS=0
            else
                STATUS=$?
                FAILED=1
            fi
            set -u
            printf '%s\t%s\tqi\t%s\t%s\n' "$MODEL_TAG" "$DATASET" "$PROMPT_MODE" "$STATUS" >> "$RUN_DIR/status.tsv"
            "$PYTHON" Results/pointwise/update_pointwise.py --input-dir "$RUN_DIR"

            if [ "$STATUS" -ne 0 ] && grep -q 'ExpiredTokenException' "$RUN_DIR/run_${TAG}.log"; then
                echo "AWS credentials expired. Refresh them and rerun this script to resume." >&2
                exit "$STATUS"
            fi
        done
    done
done

echo "Finished. Per-run status: $RUN_DIR/status.tsv"
exit "$FAILED"
