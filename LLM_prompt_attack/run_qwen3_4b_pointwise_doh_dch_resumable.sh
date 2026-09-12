#!/usr/bin/env bash
# Run directly on the RMIT server with:
#   bash LLM_prompt_attack/run_qwen3_4b_pointwise_doh_dch_resumable.sh
# Re-run the identical command after an interruption. Stable checkpoints in
# RUN_ID allow the new process to continue saved pointwise model calls.
# This script reuses a Qwen3-4B vLLM server already listening on loopback; it
# never starts or stops that server.

set -uo pipefail

RESEARCH_ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/$USER}"
ENV_FILE="${ENV_FILE:-$RESEARCH_ROOT/qwen3-ranking-env.sh}"
PROJECT="${PROJECT:-$RESEARCH_ROOT/LLM-Ranker-Attack}"
PYTHON="${PYTHON:-python}"
NUM_PASSAGES="${NUM_PASSAGES:-4096}"
N_JOBS="${N_JOBS:-1}"
PORT="${PORT:-8000}"
RUN_ID="${RUN_ID:-Qwen3-4B_pointwise_doh_dch}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
elif [ "${CONDA_PREFIX:-}" != "$RESEARCH_ROOT/environments/qwen3-vllm" ]; then
    echo "Activate $RESEARCH_ROOT/environments/qwen3-vllm or provide ENV_FILE." >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$RESEARCH_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$RESEARCH_ROOT/cache/vllm}"
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$RESEARCH_ROOT/ir_datasets}"

if [ ! -x "$(command -v "$PYTHON")" ]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi
cd "$PROJECT"
case "$(pwd -P)/" in
    "$RESEARCH_ROOT/"*) ;;
    *) echo "Repository must be under $RESEARCH_ROOT" >&2; exit 1 ;;
esac

RUN_DIR="LLM_prompt_attack/outputs/$RUN_ID"
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd -P)"
if [ ! -f "$RUN_DIR/status.tsv" ]; then
    printf 'model\tdataset\tattack\tprompt\tstatus\n' > "$RUN_DIR/status.tsv"
fi

hostname > "$RUN_DIR/environment.log"
nvidia-smi >> "$RUN_DIR/environment.log"
getconf GNU_LIBC_VERSION >> "$RUN_DIR/environment.log"
"$PYTHON" -m pip freeze > "$RUN_DIR/packages.txt"
git rev-parse HEAD > "$RUN_DIR/source_revision.txt"
cp LLM_prompt_attack/prompts.py "$RUN_DIR/prompts.py"

if ! curl -fsS --max-time 10 "$BASE_URL/models" > "$RUN_DIR/models.json"; then
    echo "No reachable Qwen3-4B vLLM server at $BASE_URL/models." >&2
    echo "Start the approved loopback server first, then resubmit this batch job." >&2
    exit 1
fi
if ! grep -q 'Qwen3-4B' "$RUN_DIR/models.json"; then
    echo "The server at $BASE_URL does not advertise Qwen3-4B." >&2
    exit 1
fi

FAILED=0
for YEAR in 2019 2020; do
    DATASET="msmarco-passage/trec-dl-$YEAR"
    for ATTACK in so sd; do
        for PROMPT_MODE in standard defense; do
            TAG="Qwen3-4B_trec-dl-${YEAR}_pointwise_${ATTACK}_${PROMPT_MODE}"
            RESULT_PATH="$RUN_DIR/result_${TAG}.jsonl"
            DETAIL_PATH="$RUN_DIR/detail_${TAG}.json"
            CHECKPOINT_PATH="$RUN_DIR/${TAG}.checkpoint.json"
            RESUME_ARGS=()

            if [ -s "$RESULT_PATH" ] && [ -s "$DETAIL_PATH" ] && \
                [ -f "$CHECKPOINT_PATH" ] && \
                "$PYTHON" -c 'import json, sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("complete") is True)' "$CHECKPOINT_PATH" | grep -qx True; then
                echo "Skipping completed $TAG"
                printf 'Qwen3-4B\t%s\t%s\t%s\tskipped-complete\n' \
                    "$DATASET" "$ATTACK" "$PROMPT_MODE" >> "$RUN_DIR/status.tsv"
                continue
            fi
            if [ -f "$CHECKPOINT_PATH" ]; then
                RESUME_ARGS=(--resume)
            fi

            echo "Running $TAG"
            if "$PYTHON" LLM_prompt_attack/pointwise_ranking_attack_openai.py \
                --provider openai \
                --base_url "$BASE_URL" \
                --model_name Qwen3-4B \
                --tokenizer_model Qwen/Qwen3-4B \
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
            printf 'Qwen3-4B\t%s\t%s\t%s\t%s\n' \
                "$DATASET" "$ATTACK" "$PROMPT_MODE" "$STATUS" >> "$RUN_DIR/status.tsv"
            "$PYTHON" Results/update_attack_outcomes.py
        done
    done
done

echo "Finished. Per-run status: $RUN_DIR/status.tsv"
exit "$FAILED"
