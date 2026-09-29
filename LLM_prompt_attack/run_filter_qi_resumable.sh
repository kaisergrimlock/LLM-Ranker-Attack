#!/usr/bin/env bash
# Resumable Filter-QI evaluation for pairwise, setwise, and listwise ranking.
set -uo pipefail

RESEARCH_ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/$USER}"
PROJECT="${PROJECT:-$RESEARCH_ROOT/LLM-Ranker-Attack}"
PYTHON="${PYTHON:-$RESEARCH_ROOT/environments/qwen3-vllm/bin/python}"
PROVIDER="${PROVIDER:-amazon-bedrock}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
AWS_REGION="${AWS_REGION:-us-east-1}"
NUM_ITEMS="${NUM_ITEMS:-4096}"
N_JOBS="${N_JOBS:-1}"
FILTER_MAX_TOKENS="${FILTER_MAX_TOKENS:-4096}"
FILTER_CACHE_DIR="${FILTER_CACHE_DIR:-$PROJECT/LLM_prompt_attack/outputs/filter_cache}"
RUN_ID="${RUN_ID:-filter_qi_resumable}"
MODEL_TAGS="${MODEL_TAGS:-Qwen3-32B}"
PARADIGMS="${PARADIGMS:-pairwise setwise listwise}"

cd "$PROJECT" || exit 1
[ -x "$PYTHON" ] || { echo "Python executable not found: $PYTHON" >&2; exit 1; }
RUN_DIR="LLM_prompt_attack/outputs/$RUN_ID"
mkdir -p "$RUN_DIR" "$FILTER_CACHE_DIR"
[ -f "$RUN_DIR/status.tsv" ] || printf 'model\tdataset\tparadigm\tprompt\tstatus\n' > "$RUN_DIR/status.tsv"

if [ "$PROVIDER" = amazon-bedrock ] && ! aws sts get-caller-identity --region "$AWS_REGION" >/dev/null; then
    echo "AWS credentials unavailable in $AWS_REGION" >&2; exit 1
fi

FAILED=0
for MODEL_TAG in $MODEL_TAGS; do
    case "$MODEL_TAG" in
        Qwen3-32B) MODEL_NAME="qwen.qwen3-32b-v1:0"; TOKENIZER_MODEL="Qwen/Qwen3-32B";;
        Qwen3-4B) MODEL_NAME="Qwen3-4B"; TOKENIZER_MODEL="Qwen/Qwen3-4B";;
        Llama-3-8B) MODEL_NAME="meta.llama3-8b-instruct-v1:0"; TOKENIZER_MODEL="NousResearch/Meta-Llama-3-8B-Instruct";;
        Llama3-70B) MODEL_NAME="meta.llama3-70b-instruct-v1:0"; TOKENIZER_MODEL="NousResearch/Meta-Llama-3-70B-Instruct";;
        GPT-OSS-20B) MODEL_NAME="openai.gpt-oss-20b-1:0"; TOKENIZER_MODEL="openai/gpt-oss-20b";;
        *) echo "Skipping unknown model $MODEL_TAG" >&2; continue;;
    esac
    for YEAR in 2019 2020; do
        DATASET="msmarco-passage/trec-dl-$YEAR"
        for PARADIGM in $PARADIGMS; do
            case "$PARADIGM" in
                pairwise) SCRIPT=pairwise_ranking_attack_openai.py; SAMPLE_ARGS=(--pos_rel 3 --neg_rel 0 --num_pairs "$NUM_ITEMS");;
                setwise|listwise) SCRIPT="${PARADIGM}_ranking_attack_openai.py"; SAMPLE_ARGS=(--num_sets "$NUM_ITEMS" --set_size 4);;
                *) echo "Skipping unsupported paradigm $PARADIGM" >&2; continue;;
            esac
            TAG="${MODEL_TAG}_trec-dl-${YEAR}_${PARADIGM}_qi_filter_qi"
            RESULT="$RUN_DIR/result_${TAG}.jsonl"; DETAIL="$RUN_DIR/detail_${TAG}.json"; CHECKPOINT="$RUN_DIR/${TAG}.checkpoint.json"
            if [ -s "$RESULT" ] && [ -s "$DETAIL" ] && [ -f "$CHECKPOINT" ] && "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8")).get("complete") is True)' "$CHECKPOINT" | grep -qx True; then
                echo "Skipping completed $TAG"; continue
            fi
            RESUME_ARGS=(); [ -f "$CHECKPOINT" ] && RESUME_ARGS=(--resume)
            PROVIDER_ARGS=(--provider "$PROVIDER" --base_url "$BASE_URL")
            FILTER_ARGS=(--filter_model "$MODEL_NAME" --filter_provider "$PROVIDER" --filter_base_url "$BASE_URL" --filter_aws_region "$AWS_REGION" --filter_max_tokens "$FILTER_MAX_TOKENS" --filter_cache_dir "$FILTER_CACHE_DIR")
            [ "$PROVIDER" = amazon-bedrock ] && PROVIDER_ARGS+=(--aws_region "$AWS_REGION")
            echo "Running $TAG (provider=$PROVIDER region=$AWS_REGION)"
            set +u
            if "$PYTHON" "LLM_prompt_attack/$SCRIPT" "${PROVIDER_ARGS[@]}" "${FILTER_ARGS[@]}" \
                --model_name "$MODEL_NAME" --tokenizer_model "$TOKENIZER_MODEL" --dataset_name "$DATASET" \
                "${SAMPLE_ARGS[@]}" --seed 42 --n_jobs "$N_JOBS" --attack_type qi --attack_position back --prompt_mode filter_qi \
                --result_json_path "$RESULT" --detailed_results "$DETAIL" --checkpoint_path "$CHECKPOINT" --checkpoint_batch_size 32 "${RESUME_ARGS[@]}" \
                2>&1 | tee "$RUN_DIR/run_${TAG}.log"; then STATUS=complete; else STATUS=failed; FAILED=1; fi
            set -u
            printf '%s\t%s\t%s\tfilter_qi\t%s\n' "$MODEL_TAG" "$DATASET" "$PARADIGM" "$STATUS" >> "$RUN_DIR/status.tsv"
        done
    done
done
"$PYTHON" Results/update_attack_outcomes.py --source-contains "$RUN_ID" || FAILED=1
echo "Finished. Status: $RUN_DIR/status.tsv"
exit "$FAILED"
