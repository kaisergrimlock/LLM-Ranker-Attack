#!/usr/bin/env bash
# Pairwise keyword-injection evaluation on Bedrock for TREC-DL 2019/2020.
# Runs GPT-OSS-20B and Qwen3-32B in the primary region, then retries the
# previously skipped Llama3-70B in an alternate region. Qwen3-4B is not
# listed by Bedrock and remains skipped.
# Re-run with the same RUN_ID to skip completed conditions and resume checkpoints.

set -o pipefail

RESEARCH_ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/${USER:-$(id -un)}}"
PROJECT="${PROJECT:-$RESEARCH_ROOT/LLM-Ranker-Attack}"
PYTHON="${PYTHON:-python}"
AWS_REGION="${AWS_REGION:-ap-southeast-2}"
SKIPPED_AWS_REGION="${SKIPPED_AWS_REGION:-us-east-1}"
NUM_PAIRS="${NUM_PAIRS:-4096}"
N_JOBS="${N_JOBS:-2}"
RUN_ID="${RUN_ID:-keyword_injection_bedrock_pairwise_except_llama3_8b}"
KEYWORDS_PATH="${KEYWORDS_PATH:-content_attack/unique_queries.tsv}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON. Activate the .conda-bedrock environment or set PYTHON." >&2
    exit 1
fi
if ! command -v aws >/dev/null 2>&1; then
    echo "AWS CLI is required for the credential preflight; check that aws is installed." >&2
    exit 1
fi
if [ ! -d "$PROJECT/LLM_prompt_attack" ]; then
    echo "Project directory not found: $PROJECT" >&2
    exit 1
fi

cd "$PROJECT"
if [ ! -f "$KEYWORDS_PATH" ]; then
    echo "Keyword mapping file not found: $KEYWORDS_PATH" >&2
    exit 1
fi

export AWS_REGION
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$RESEARCH_ROOT/ir_datasets}"
export HF_HOME="${HF_HOME:-$RESEARCH_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

if ! aws --version; then
    echo "Unable to run AWS CLI. Check the server's AWS installation." >&2
    exit 1
fi
if ! aws sts get-caller-identity --region "$AWS_REGION" >/dev/null; then
    echo "AWS credentials are unavailable for region $AWS_REGION." >&2
    echo "Refresh/configure credentials, verify with aws sts get-caller-identity, then rerun." >&2
    exit 1
fi
if ! aws sts get-caller-identity --region "$SKIPPED_AWS_REGION" >/dev/null; then
    echo "AWS credentials are unavailable for alternate region $SKIPPED_AWS_REGION." >&2
    exit 1
fi

RUN_DIR="LLM_prompt_attack/outputs/$RUN_ID"
mkdir -p "$RUN_DIR"
if [ ! -f "$RUN_DIR/status.tsv" ]; then
    printf 'model\tdataset\tstatus\n' > "$RUN_DIR/status.tsv"
fi

KEYWORDS_SHA="$(sha256sum "$KEYWORDS_PATH" | awk '{print $1}')"
SOURCE_REVISION="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
CONFIG_PATH="$RUN_DIR/run_config.tsv"
CONFIG_TMP="$RUN_DIR/run_config.tsv.tmp"
{
    printf 'source_revision\t%s\n' "$SOURCE_REVISION"
    printf 'aws_region\t%s\n' "$AWS_REGION"
    printf 'skipped_aws_region\t%s\n' "$SKIPPED_AWS_REGION"
    printf 'num_pairs\t%s\n' "$NUM_PAIRS"
    printf 'seed\t42\n'
    printf 'n_jobs\t%s\n' "$N_JOBS"
    printf 'attack\tkey_injection\trandom\tstandard\n'
    printf 'keywords_sha256\t%s\n' "$KEYWORDS_SHA"
    printf 'models\tGPT-OSS-20B,Qwen3-32B,Llama3-70B\n'
    printf 'datasets\ttrec-dl-2019,trec-dl-2020\n'
} > "$CONFIG_TMP"
if [ -f "$CONFIG_PATH" ]; then
    # Source revision and concurrency are operational metadata, not checkpoint
    # identity; allow code updates and a changed N_JOBS when resuming.
    if ! diff -u \
        <(grep -Ev '^(source_revision|n_jobs|models|skipped_aws_region)[[:space:]]' "$CONFIG_PATH") \
        <(grep -Ev '^(source_revision|n_jobs|models|skipped_aws_region)[[:space:]]' "$CONFIG_TMP") \
        >/dev/null; then
        rm -f "$CONFIG_TMP"
        echo "Run settings differ from $CONFIG_PATH." >&2
        echo "Use the original settings to resume, or choose a new RUN_ID for a fresh evaluation." >&2
        exit 1
    fi
    rm -f "$CONFIG_TMP"
else
    mv "$CONFIG_TMP" "$CONFIG_PATH"
fi

hostname > "$RUN_DIR/environment.log"
date --iso-8601=seconds >> "$RUN_DIR/environment.log"
aws --version >> "$RUN_DIR/environment.log" 2>&1
printf 'AWS_REGION=%s\n' "$AWS_REGION" >> "$RUN_DIR/environment.log"
printf 'SKIPPED_AWS_REGION=%s\n' "$SKIPPED_AWS_REGION" >> "$RUN_DIR/environment.log"
printf 'RUN_ID=%s\n' "$RUN_ID" >> "$RUN_DIR/environment.log"
git rev-parse HEAD > "$RUN_DIR/source_revision.txt" 2>/dev/null || true
cp LLM_prompt_attack/prompts.py "$RUN_DIR/prompts.py"
cp "$KEYWORDS_PATH" "$RUN_DIR/unique_queries.tsv"

FAILED=0
for MODEL_TAG in GPT-OSS-20B Qwen3-32B Llama3-70B; do
    MODEL_REGION="$AWS_REGION"
    case "$MODEL_TAG" in
        GPT-OSS-20B)
            MODEL_NAME="openai.gpt-oss-20b-1:0"
            TOKENIZER_MODEL="openai/gpt-oss-20b"
            ;;
        Llama3-70B)
            MODEL_NAME="meta.llama3-70b-instruct-v1:0"
            TOKENIZER_MODEL="NousResearch/Meta-Llama-3-70B-Instruct"
            MODEL_REGION="$SKIPPED_AWS_REGION"
            ;;
        Qwen3-32B)
            MODEL_NAME="qwen.qwen3-32b-v1:0"
            TOKENIZER_MODEL="Qwen/Qwen3-32B"
            ;;
    esac

    for YEAR in 2019 2020; do
        DATASET="msmarco-passage/trec-dl-${YEAR}"
        TAG="${MODEL_TAG}_trec-dl-${YEAR}_pairwise_key_injection"
        RESULT_PATH="$RUN_DIR/result_${TAG}.jsonl"
        DETAIL_PATH="$RUN_DIR/detail_${TAG}.json"
        CHECKPOINT_PATH="$RUN_DIR/${TAG}.checkpoint.json"
        LOG_PATH="$RUN_DIR/run_${TAG}.log"

        if [ -s "$RESULT_PATH" ] && [ -f "$CHECKPOINT_PATH" ] && \
            "$PYTHON" -c 'import json,sys; print(str(json.load(open(sys.argv[1], encoding="utf-8")).get("complete") is True).lower())' \
                "$CHECKPOINT_PATH" | grep -qx true; then
            echo "Skipping completed $TAG"
            printf '%s\t%s\tskipped-complete\n' "$MODEL_TAG" "$DATASET" >> "$RUN_DIR/status.tsv"
            continue
        fi

        RESUME_ARGS=()
        if [ -f "$CHECKPOINT_PATH" ]; then
            RESUME_ARGS=(--resume)
        fi

        echo "Running $TAG (region=$MODEL_REGION, run_id=$RUN_ID)"
        printf '%s\t%s\trunning\n' "$MODEL_TAG" "$DATASET" >> "$RUN_DIR/status.tsv"
        if "$PYTHON" LLM_prompt_attack/pairwise_ranking_attack_openai.py \
            --provider amazon-bedrock \
            --aws_region "$MODEL_REGION" \
            --model_name "$MODEL_NAME" \
            --tokenizer_model "$TOKENIZER_MODEL" \
            --dataset_name "$DATASET" \
            --pos_rel 3 \
            --neg_rel 0 \
            --num_pairs "$NUM_PAIRS" \
            --seed 42 \
            --attack_type key_injection \
            --attack_position random \
            --prompt_mode standard \
            --keywords_path "$KEYWORDS_PATH" \
            --n_jobs "$N_JOBS" \
            --checkpoint_batch_size 32 \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --result_json_path "$RESULT_PATH" \
            --detailed_results "$DETAIL_PATH" \
            "${RESUME_ARGS[@]}" \
            2>&1 | tee "$LOG_PATH"; then
            STATUS=complete
        else
            STATUS=failed
            FAILED=1
        fi
        printf '%s\t%s\t%s\n' "$MODEL_TAG" "$DATASET" "$STATUS" >> "$RUN_DIR/status.tsv"
    done
done

echo "Importing only this run's results into Results/attack_outcomes.csv"
if ! "$PYTHON" Results/update_attack_outcomes.py --source-contains "$RUN_ID"; then
    FAILED=1
fi

echo "Finished. Output and status: $RUN_DIR"
exit "$FAILED"
