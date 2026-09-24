#!/usr/bin/env bash
# Resumable pairwise keyword-injection evaluation with the marker-aware
# Defense prompt on TREC-DL 2019/2020.
# Models: GPT-OSS-20B, Qwen3-32B, and Bedrock Llama-3-8B.

set -o pipefail

RESEARCH_ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/${USER:-$(id -un)}}"
PROJECT="${PROJECT:-$RESEARCH_ROOT/LLM-Ranker-Attack}"
PYTHON="${PYTHON:-python}"
AWS_REGION="${AWS_REGION:-ap-southeast-2}"
LLAMA8_AWS_REGION="${LLAMA8_AWS_REGION:-us-east-1}"
NUM_PAIRS="${NUM_PAIRS:-4096}"
N_JOBS="${N_JOBS:-1}"
RUN_ID="${RUN_ID:-keyword_injection_bedrock_defense_with_llama3_8b}"
KEYWORDS_PATH="${KEYWORDS_PATH:-content_attack/unique_queries.tsv}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi
if ! command -v aws >/dev/null 2>&1; then
    echo "AWS CLI is required." >&2
    exit 1
fi
if [ ! -d "$PROJECT/LLM_prompt_attack" ]; then
    echo "Project directory not found: $PROJECT" >&2
    exit 1
fi

cd "$PROJECT"
[ -f "$KEYWORDS_PATH" ] || { echo "Keyword mapping file not found: $KEYWORDS_PATH" >&2; exit 1; }
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$RESEARCH_ROOT/ir_datasets}"
export HF_HOME="${HF_HOME:-$RESEARCH_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

RUN_DIR="LLM_prompt_attack/outputs/$RUN_ID"
mkdir -p "$RUN_DIR"
[ -f "$RUN_DIR/status.tsv" ] || printf 'model\tdataset\tregion\tstatus\n' > "$RUN_DIR/status.tsv"

KEYWORDS_SHA="$(sha256sum "$KEYWORDS_PATH" | awk '{print $1}')"
CONFIG_PATH="$RUN_DIR/run_config.tsv"
CONFIG_TMP="$RUN_DIR/run_config.tsv.tmp"
{
    printf 'aws_region\t%s\n' "$AWS_REGION"
    printf 'llama8_aws_region\t%s\n' "$LLAMA8_AWS_REGION"
    printf 'num_pairs\t%s\n' "$NUM_PAIRS"
    printf 'seed\t42\n'
    printf 'n_jobs\t%s\n' "$N_JOBS"
    printf 'attack\tkey_injection\trandom\tdefense\n'
    printf 'keywords_sha256\t%s\n' "$KEYWORDS_SHA"
    printf 'models\tGPT-OSS-20B,Qwen3-32B,Llama-3-8B\n'
    printf 'datasets\ttrec-dl-2019,trec-dl-2020\n'
} > "$CONFIG_TMP"
if [ -f "$CONFIG_PATH" ]; then
    if ! diff -u \
        <(grep -Ev '^(n_jobs|models)[[:space:]]' "$CONFIG_PATH") \
        <(grep -Ev '^(n_jobs|models)[[:space:]]' "$CONFIG_TMP") \
        >/dev/null; then
        rm -f "$CONFIG_TMP"
        echo "Run settings differ from $CONFIG_PATH; use a new RUN_ID for a fresh run." >&2
        exit 1
    fi
    rm -f "$CONFIG_TMP"
else
    mv "$CONFIG_TMP" "$CONFIG_PATH"
fi

aws --version | tee "$RUN_DIR/aws_version.txt"
hostname > "$RUN_DIR/environment.log"
date --iso-8601=seconds >> "$RUN_DIR/environment.log"
printf 'AWS_REGION=%s\nLLAMA8_AWS_REGION=%s\nRUN_ID=%s\n' \
    "$AWS_REGION" "$LLAMA8_AWS_REGION" "$RUN_ID" >> "$RUN_DIR/environment.log"
git rev-parse HEAD > "$RUN_DIR/source_revision.txt" 2>/dev/null || true
cp LLM_prompt_attack/prompts.py "$RUN_DIR/prompts.py"
cp "$KEYWORDS_PATH" "$RUN_DIR/unique_queries.tsv"

FAILED=0
for MODEL_SPEC in \
    'GPT-OSS-20B|openai.gpt-oss-20b-1:0|ap-southeast-2|openai/gpt-oss-20b' \
    'Qwen3-32B|qwen.qwen3-32b-v1:0|ap-southeast-2|Qwen/Qwen3-32B' \
    'Llama-3-8B|meta.llama3-8b-instruct-v1:0|us-east-1|NousResearch/Meta-Llama-3-8B-Instruct'; do
    IFS='|' read -r MODEL_TAG MODEL_NAME DEFAULT_REGION TOKENIZER_MODEL <<< "$MODEL_SPEC"
    MODEL_REGION="$DEFAULT_REGION"
    [ "$MODEL_TAG" = "Llama-3-8B" ] && MODEL_REGION="$LLAMA8_AWS_REGION"

    if ! aws sts get-caller-identity --region "$MODEL_REGION" >/dev/null; then
        echo "AWS credentials unavailable for $MODEL_TAG in $MODEL_REGION" >&2
        FAILED=1
        continue
    fi

    for YEAR in 2019 2020; do
        DATASET="msmarco-passage/trec-dl-${YEAR}"
        TAG="${MODEL_TAG}_trec-dl-${YEAR}_pairwise_key_injection_defense"
        RESULT_PATH="$RUN_DIR/result_${TAG}.jsonl"
        DETAIL_PATH="$RUN_DIR/detail_${TAG}.json"
        CHECKPOINT_PATH="$RUN_DIR/${TAG}.checkpoint.json"
        LOG_PATH="$RUN_DIR/run_${TAG}.log"

        if [ -s "$RESULT_PATH" ] && [ -f "$CHECKPOINT_PATH" ] && \
            "$PYTHON" -c 'import json,sys; print(str(json.load(open(sys.argv[1], encoding="utf-8")).get("complete") is True).lower())' \
            "$CHECKPOINT_PATH" | grep -qx true; then
            echo "Skipping completed $TAG"
            printf '%s\t%s\t%s\tskipped-complete\n' "$MODEL_TAG" "$DATASET" "$MODEL_REGION" >> "$RUN_DIR/status.tsv"
            continue
        fi

        RESUME_ARGS=()
        [ -f "$CHECKPOINT_PATH" ] && RESUME_ARGS=(--resume)
        echo "Running $TAG (region=$MODEL_REGION, run_id=$RUN_ID)"
        if "$PYTHON" LLM_prompt_attack/pairwise_ranking_attack_openai.py \
            --provider amazon-bedrock \
            --aws_region "$MODEL_REGION" \
            --model_name "$MODEL_NAME" \
            --tokenizer_model "$TOKENIZER_MODEL" \
            --dataset_name "$DATASET" \
            --pos_rel 3 --neg_rel 0 --num_pairs "$NUM_PAIRS" --seed 42 \
            --attack_type key_injection --attack_position random \
            --prompt_mode defense \
            --keywords_path "$KEYWORDS_PATH" \
            --n_jobs "$N_JOBS" --checkpoint_batch_size 32 \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --result_json_path "$RESULT_PATH" \
            --detailed_results "$DETAIL_PATH" \
            "${RESUME_ARGS[@]}" 2>&1 | tee "$LOG_PATH"; then
            STATUS=complete
        else
            STATUS=failed
            FAILED=1
        fi
        printf '%s\t%s\t%s\t%s\n' "$MODEL_TAG" "$DATASET" "$MODEL_REGION" "$STATUS" >> "$RUN_DIR/status.tsv"
    done
done

echo "Importing only this run's results"
"$PYTHON" Results/update_attack_outcomes.py --source-contains "$RUN_ID" || FAILED=1
echo "Finished. Output directory: $RUN_DIR"
exit "$FAILED"
