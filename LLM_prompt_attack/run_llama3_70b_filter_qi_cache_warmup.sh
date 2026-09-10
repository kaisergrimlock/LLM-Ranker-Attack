#!/usr/bin/env bash
# Prewarm the Filter QI passage cache for Llama 3 70B before evaluation.
# Run directly on the server after AWS credentials are available:
#   ./LLM_prompt_attack/run_llama3_70b_filter_qi_cache_warmup.sh

set -euo pipefail

RESEARCH_ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/$USER}"
PROJECT="${PROJECT:-$RESEARCH_ROOT/LLM-Ranker-Attack}"
PYTHON="${PYTHON:-$RESEARCH_ROOT/environments/qwen3-vllm/bin/python}"
AWS_REGION="${AWS_REGION:-us-west-2}"
NUM_SAMPLES="${NUM_SAMPLES:-4096}"
N_JOBS="${N_JOBS:-2}"
FILTER_MAX_TOKENS="${FILTER_MAX_TOKENS:-8192}"
CACHE_DIR="${FILTER_CACHE_DIR:-LLM_prompt_attack/outputs/filter_cache}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"

if [ ! -x "$PYTHON" ]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi
if [ ! -d "$PROJECT/LLM_prompt_attack" ]; then
    echo "Project directory is invalid: $PROJECT" >&2
    exit 1
fi

export AWS_REGION
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$RESEARCH_ROOT/ir_datasets}"
export HF_HOME="${HF_HOME:-$RESEARCH_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export BEDROCK_MAX_TOKENS="${BEDROCK_MAX_TOKENS:-1024}"

if ! aws sts get-caller-identity --region "$AWS_REGION" >/dev/null; then
    echo "AWS credentials are unavailable." >&2
    exit 1
fi

cd "$PROJECT"
RUN_DIR="LLM_prompt_attack/outputs/Llama3-70B_filter_qi_cache_warmup_${RUN_ID}"
mkdir -p "$RUN_DIR" "$CACHE_DIR/filtered"
hostname > "$RUN_DIR/environment.log"
date --iso-8601=seconds >> "$RUN_DIR/environment.log"
printf 'dataset\tparadigm\texit_status\n' > "$RUN_DIR/status.tsv"
cp LLM_prompt_attack/filter_defense.py "$RUN_DIR/filter_defense.py"
cp LLM_prompt_attack/prompts.py "$RUN_DIR/prompts.py"

for DATASET in msmarco-passage/trec-dl-2019 msmarco-passage/trec-dl-2020; do
    for SCHEME in pairwise setwise listwise; do
        if [ "$SCHEME" = pairwise ]; then
            SAMPLE_ARGS=(--pos_rel 3 --neg_rel 0 --num_pairs "$NUM_SAMPLES")
        else
            SAMPLE_ARGS=(--num_sets "$NUM_SAMPLES" --set_size 4)
        fi
        TAG="${DATASET##*/}_${SCHEME}_qi_filter_cache_warmup"

        echo "Prewarming $TAG"
        if "$PYTHON" "LLM_prompt_attack/${SCHEME}_ranking_attack_openai.py" \
            --provider amazon-bedrock --aws_region "$AWS_REGION" \
            --model_name meta.llama3-70b-instruct-v1:0 \
            --tokenizer_model NousResearch/Meta-Llama-3-70B-Instruct \
            --filter_model meta.llama3-70b-instruct-v1:0 \
            --filter_provider amazon-bedrock --filter_aws_region "$AWS_REGION" \
            --filter_max_tokens "$FILTER_MAX_TOKENS" \
            --filter_cache_dir "$CACHE_DIR" --filter_cache_mode read-write \
            --filter_cache_only \
            --dataset_name "$DATASET" --seed 42 --n_jobs "$N_JOBS" \
            --attack_type qi --attack_position back --prompt_mode filter_qi \
            "${SAMPLE_ARGS[@]}" \
            --result_json_path "$RUN_DIR/result_Llama3-70B_${TAG}.jsonl" \
            --detailed_results "$RUN_DIR/detail_Llama3-70B_${TAG}.json" \
            2>&1 | tee "$RUN_DIR/run_Llama3-70B_${TAG}.log"; then
            STATUS=0
        else
            STATUS=$?
            printf '%s\t%s\t%s\n' "$DATASET" "$SCHEME" "$STATUS" >> "$RUN_DIR/status.tsv"
            exit "$STATUS"
        fi
        printf '%s\t%s\t%s\n' "$DATASET" "$SCHEME" "$STATUS" >> "$RUN_DIR/status.tsv"
    done
done

FILTERED_COUNT="$(find "$CACHE_DIR/filtered" -maxdepth 1 -name '*.json' -type f 2>/dev/null | wc -l)"
echo "Filter cache warm-up finished. Cached passages: $FILTERED_COUNT"
echo "Status: $RUN_DIR/status.tsv"
