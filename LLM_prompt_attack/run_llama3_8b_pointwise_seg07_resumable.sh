#!/usr/bin/env bash
# Run pointwise Llama-3-8B evaluations against the SEG07 local Transformers
# server. The server must already be listening on loopback; this script does
# not start or stop it. Re-running resumes incomplete conditions from their
# checkpoints and skips completed JSONL outputs.

set -uo pipefail

RESEARCH_ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/$USER}"
PROJECT="${PROJECT:-$RESEARCH_ROOT/LLM-Ranker-Attack}"
PYTHON="${PYTHON:-$RESEARCH_ROOT/environments/llama3-8b/bin/python}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}/v1}"
MODEL_NAME="${MODEL_NAME:-Llama3-8B}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-NousResearch/Meta-Llama-3-8B-Instruct}"
NUM_PASSAGES="${NUM_PASSAGES:-4096}"
N_JOBS="${N_JOBS:-1}"
# Keep SEG07 results separate from Bedrock and older local runs.
RUN_ID="${RUN_ID:-Llama3-8B_pointwise_seg07_new}"

if [ ! -x "$PYTHON" ]; then
    echo "Python executable not found: $PYTHON" >&2
    echo "Set PYTHON to the SEG07 environment's Python interpreter." >&2
    exit 1
fi
if [ ! -d "$PROJECT/LLM_prompt_attack" ]; then
    echo "Project directory is invalid: $PROJECT" >&2
    exit 1
fi

export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$RESEARCH_ROOT/ir_datasets}"
export HF_HOME="${HF_HOME:-$RESEARCH_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

cd "$PROJECT"
RUN_DIR="LLM_prompt_attack/outputs/$RUN_ID"
mkdir -p "$RUN_DIR"
if [ ! -f "$RUN_DIR/status.tsv" ]; then
    printf 'model\tdataset\tattack\tprompt\tstatus\n' > "$RUN_DIR/status.tsv"
fi

hostname > "$RUN_DIR/environment.log"
date --iso-8601=seconds >> "$RUN_DIR/environment.log"
nvidia-smi >> "$RUN_DIR/environment.log" 2>&1 || true
getconf GNU_LIBC_VERSION >> "$RUN_DIR/environment.log"
"$PYTHON" -m pip freeze > "$RUN_DIR/packages.txt"
git rev-parse HEAD > "$RUN_DIR/source_revision.txt"
cp LLM_prompt_attack/prompts.py "$RUN_DIR/prompts.py"

if ! curl -fsS --max-time 10 "$BASE_URL/models" > "$RUN_DIR/models.json"; then
    echo "No local Llama3-8B server reachable at $BASE_URL/models." >&2
    echo "Start local_transformers_openai_server.py on SEG07 first." >&2
    exit 1
fi
if ! "$PYTHON" -c 'import json, sys; model = sys.argv[1]; data = json.load(open(sys.argv[2], encoding="utf-8")); raise SystemExit(0 if any(item.get("id") == model for item in data.get("data", [])) else 1)' "$MODEL_NAME" "$RUN_DIR/models.json"; then
    echo "The server at $BASE_URL does not advertise $MODEL_NAME." >&2
    cat "$RUN_DIR/models.json" >&2
    exit 1
fi

FAILED=0
for YEAR in 2019 2020; do
    DATASET="msmarco-passage/trec-dl-$YEAR"
    for ATTACK in so sd qi; do
        if [ "$ATTACK" = qi ]; then
            PROMPTS=(standard defense_qi)
        else
            PROMPTS=(standard defense)
        fi
        for PROMPT_MODE in "${PROMPTS[@]}"; do
            TAG="Llama3-8B_trec-dl-${YEAR}_pointwise_${ATTACK}_${PROMPT_MODE}"
            RESULT_PATH="$RUN_DIR/result_${TAG}.jsonl"
            DETAIL_PATH="$RUN_DIR/detail_${TAG}.json"
            CHECKPOINT_PATH="$RUN_DIR/${TAG}.checkpoint.json"
            RESUME_ARGS=()

            if [ -s "$RESULT_PATH" ] && [ -s "$DETAIL_PATH" ] && \
                [ -f "$CHECKPOINT_PATH" ] && \
                "$PYTHON" -c 'import json, sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("complete") is True)' "$CHECKPOINT_PATH" | grep -qx True; then
                echo "Skipping completed $TAG"
                printf '%s\t%s\t%s\t%s\tskipped-complete\n' \
                    "$MODEL_NAME" "$DATASET" "$ATTACK" "$PROMPT_MODE" >> "$RUN_DIR/status.tsv"
                continue
            fi
            if [ -f "$CHECKPOINT_PATH" ]; then
                RESUME_ARGS=(--resume)
            fi

            echo "Running $TAG"
            set +u
            if "$PYTHON" LLM_prompt_attack/pointwise_ranking_attack_openai.py \
                    --provider openai \
                    --base_url "$BASE_URL" \
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
                "$MODEL_NAME" "$DATASET" "$ATTACK" "$PROMPT_MODE" "$STATUS" \
                >> "$RUN_DIR/status.tsv"
            "$PYTHON" Results/pointwise/update_pointwise.py --input-dir "$RUN_DIR"
        done
    done
done

echo "Finished. Results: $RUN_DIR"
echo "Status: $RUN_DIR/status.tsv"
exit "$FAILED"
