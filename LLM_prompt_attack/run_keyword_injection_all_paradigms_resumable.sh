/bin/bash: warning: setlocale: LC_ALL: cannot change locale (C.UTF-8)
/bin/bash: warning: setlocale: LC_ALL: cannot change locale (C.UTF-8)
#!/usr/bin/env bash
# Resumable Bedrock keyword-injection evaluation for the paradigms missing from
# the existing pairwise table.  By default this runs Pointwise, Setwise, and
# Listwise for both datasets and both standard/defense prompts.

set -uo pipefail

RESEARCH_ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/$USER}"
PROJECT="${PROJECT:-$RESEARCH_ROOT/LLM-Ranker-Attack}"
PYTHON="${PYTHON:-$RESEARCH_ROOT/environments/qwen3-vllm/bin/python}"
AWS_REGION="${AWS_REGION:-ap-southeast-2}"
PROVIDER="${PROVIDER:-amazon-bedrock}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
LLAMA8_AWS_REGION="${LLAMA8_AWS_REGION:-us-east-1}"
LLAMA70_AWS_REGION="${LLAMA70_AWS_REGION:-us-east-1}"
NUM_ITEMS="${NUM_ITEMS:-4096}"
N_JOBS="${N_JOBS:-1}"
RUN_ID="${RUN_ID:-keyword_injection_all_paradigms_resumable}"
KEYWORDS_PATH="${KEYWORDS_PATH:-content_attack/unique_queries.tsv}"
MODEL_TAGS="${MODEL_TAGS:-GPT-OSS-20B Qwen3-32B Llama-3-8B Llama3-70B}"
PARADIGMS="${PARADIGMS:-pointwise setwise listwise}"

if [ ! -x "$PYTHON" ]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi
if [ ! -d "$PROJECT/LLM_prompt_attack" ]; then
    echo "Project directory is invalid: $PROJECT" >&2
    exit 1
fi
cd "$PROJECT" || exit 1
[ -f "$KEYWORDS_PATH" ] || { echo "Missing $KEYWORDS_PATH" >&2; exit 1; }

export AWS_REGION
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$RESEARCH_ROOT/ir_datasets}"
export HF_HOME="${HF_HOME:-$RESEARCH_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export BEDROCK_MAX_TOKENS="${BEDROCK_MAX_TOKENS:-512}"

if [ "$PROVIDER" = "amazon-bedrock" ] && ! aws sts get-caller-identity --region "$AWS_REGION" >/dev/null; then
    echo "AWS credentials unavailable in $AWS_REGION" >&2
    exit 1
fi

RUN_DIR="LLM_prompt_attack/outputs/$RUN_ID"
mkdir -p "$RUN_DIR"
[ -f "$RUN_DIR/status.tsv" ] || printf 'model\tdataset\tparadigm\tprompt\tstatus\n' > "$RUN_DIR/status.tsv"
cp "$KEYWORDS_PATH" "$RUN_DIR/unique_queries.tsv"
FAILED=0

for MODEL_TAG in $MODEL_TAGS; do
    case "$MODEL_TAG" in
        GPT-OSS-20B) MODEL_NAME="openai.gpt-oss-20b-1:0"; TOKENIZER_MODEL="openai/gpt-oss-20b"; MODEL_REGION="$AWS_REGION";;
        Qwen3-32B) MODEL_NAME="qwen.qwen3-32b-v1:0"; TOKENIZER_MODEL="Qwen/Qwen3-32B"; MODEL_REGION="$AWS_REGION";;
        Qwen3-4B) MODEL_NAME="Qwen3-4B"; TOKENIZER_MODEL="Qwen/Qwen3-4B"; MODEL_REGION="";;
        Llama-3-8B) MODEL_NAME="meta.llama3-8b-instruct-v1:0"; TOKENIZER_MODEL="NousResearch/Meta-Llama-3-8B-Instruct"; MODEL_REGION="$LLAMA8_AWS_REGION";;
        Llama3-70B) MODEL_NAME="meta.llama3-70b-instruct-v1:0"; TOKENIZER_MODEL="NousResearch/Meta-Llama-3-70B-Instruct"; MODEL_REGION="$LLAMA70_AWS_REGION";;
        *) echo "Skipping unknown model $MODEL_TAG" >&2; continue;;
    esac
    if [ "$PROVIDER" = "amazon-bedrock" ] && ! aws sts get-caller-identity --region "$MODEL_REGION" >/dev/null; then
        echo "Skipping $MODEL_TAG: credentials unavailable in $MODEL_REGION" >&2
        FAILED=1
        continue
    fi

    for YEAR in 2019 2020; do
        DATASET="msmarco-passage/trec-dl-$YEAR"
        for PARADIGM in $PARADIGMS; do
            case "$PARADIGM" in
                pointwise) SCRIPT=pointwise_ranking_attack_openai.py; SAMPLE_ARGS=(--num_passages "$NUM_ITEMS");;
                setwise|listwise) SCRIPT="${PARADIGM}_ranking_attack_openai.py"; SAMPLE_ARGS=(--num_sets "$NUM_ITEMS" --set_size 4);;
                pairwise) SCRIPT=pairwise_ranking_attack_openai.py; SAMPLE_ARGS=(--pos_rel 3 --neg_rel 0 --num_pairs "$NUM_ITEMS");;
                *) echo "Skipping unsupported paradigm $PARADIGM" >&2; continue;;
            esac
            for PROMPT_MODE in standard defense; do
                TAG="${MODEL_TAG}_trec-dl-${YEAR}_${PARADIGM}_key_injection_${PROMPT_MODE}"
                RESULT_PATH="$RUN_DIR/result_${TAG}.jsonl"
                DETAIL_PATH="$RUN_DIR/detail_${TAG}.json"
                CHECKPOINT_PATH="$RUN_DIR/${TAG}.checkpoint.json"
                if [ -s "$RESULT_PATH" ] && [ -f "$CHECKPOINT_PATH" ] && \
                    "$PYTHON" -c 'import json,sys; print(str(json.load(open(sys.argv[1], encoding="utf-8")).get("complete") is True).lower())' "$CHECKPOINT_PATH" | grep -qx true; then
                    echo "Skipping completed $TAG"
                    continue
                fi
                RESUME_ARGS=(); [ -f "$CHECKPOINT_PATH" ] && RESUME_ARGS=(--resume)
                echo "Running $TAG (region=$MODEL_REGION)"
                set +u
                PROVIDER_ARGS=(--provider "$PROVIDER" --base_url "$BASE_URL")
                if [ "$PROVIDER" = "amazon-bedrock" ]; then
                    PROVIDER_ARGS+=(--aws_region "$MODEL_REGION")
                fi
                if "$PYTHON" "LLM_prompt_attack/$SCRIPT" \
                    "${PROVIDER_ARGS[@]}" \
                    --model_name "$MODEL_NAME" --tokenizer_model "$TOKENIZER_MODEL" \
                    --dataset_name "$DATASET" "${SAMPLE_ARGS[@]}" --seed 42 --n_jobs "$N_JOBS" \
                    --attack_type key_injection --attack_position random \
                    --prompt_mode "$PROMPT_MODE" --keywords_path "$KEYWORDS_PATH" \
                    --result_json_path "$RESULT_PATH" --detailed_results "$DETAIL_PATH" \
                    --checkpoint_path "$CHECKPOINT_PATH" --checkpoint_batch_size 32 \
                    "${RESUME_ARGS[@]}" 2>&1 | tee "$RUN_DIR/run_${TAG}.log"; then
                    STATUS=complete
                else
                    STATUS=failed
                    FAILED=1
                fi
                set -u
                printf '%s\t%s\t%s\t%s\t%s\n' "$MODEL_TAG" "$DATASET" "$PARADIGM" "$PROMPT_MODE" "$STATUS" >> "$RUN_DIR/status.tsv"
            done
        done
    done
done

"$PYTHON" Results/update_attack_outcomes.py --source-contains "$RUN_ID" || FAILED=1
echo "Finished. Status: $RUN_DIR/status.tsv"
exit "$FAILED"
