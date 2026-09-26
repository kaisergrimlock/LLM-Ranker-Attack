#!/usr/bin/env bash
# Replace completed GPT-OSS ablation runs whose valid attacked count is zero.
# Existing artifacts are moved to a timestamped backup before rerunning.

set -uo pipefail

RESEARCH_ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/$USER}"
PROJECT="${PROJECT:-$RESEARCH_ROOT/LLM-Ranker-Attack}"
PYTHON="${PYTHON:-$RESEARCH_ROOT/environments/llama3-8b/bin/python}"
AWS_REGION="${AWS_REGION:-ap-southeast-2}"
AWS_PROFILE="${AWS_PROFILE:-default}"
N_JOBS="${N_JOBS:-2}"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT/LLM_prompt_attack/outputs/zero_valid_ablation_backups/$(date +%Y%m%d_%H%M%S)}"

if [ ! -x "$PYTHON" ]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi
cd "$PROJECT" || exit 1

export AWS_PROFILE AWS_REGION
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$RESEARCH_ROOT/ir_datasets}"
export HF_HOME="${HF_HOME:-$RESEARCH_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export BEDROCK_MAX_TOKENS="${BEDROCK_MAX_TOKENS:-512}"

if ! aws sts get-caller-identity --region "$AWS_REGION" >/dev/null; then
    echo "AWS credentials are unavailable; refresh them before running this script." >&2
    exit 1
fi

CSV_PATH="Results/ablation_study.csv"
if [ ! -f "$CSV_PATH" ]; then
    echo "Missing $CSV_PATH; run Results/update_ablation_study.py first." >&2
    exit 1
fi

mkdir -p "$BACKUP_DIR"
FAILED=0

# Emit only GPT-OSS rows whose completed evaluation had no valid attacked calls.
while IFS=$'\t' read -r SOURCE MODEL SETTING PARADIGM ATTACK PROMPT PASSAGES; do
    [ -n "$SOURCE" ] || continue
    case "$PARADIGM" in
        Pointwise) SCRIPT=pointwise_ranking_attack_openai.py; SIZE=(--num_passages "$PASSAGES");;
        Pairwise) SCRIPT=pairwise_ranking_attack_openai.py; SIZE=(--num_pairs "$PASSAGES");;
        Setwise|Listwise) SCRIPT="${PARADIGM,,}_ranking_attack_openai.py"; SIZE=(--num_sets "$PASSAGES");;
        *) echo "Skipping unsupported paradigm: $PARADIGM" >&2; continue;;
    esac
    case "$ATTACK" in
        DOH) ATTACK_CODE=so;;
        DCH) ATTACK_CODE=sd;;
        *) echo "Skipping unsupported attack: $ATTACK" >&2; continue;;
    esac
    case "$PROMPT" in
        Default) PROMPT_MODE=standard;;
        Defense) PROMPT_MODE=defense;;
        *) echo "Skipping unsupported prompt: $PROMPT" >&2; continue;;
    esac

    RESULT_PATH="$PROJECT/$SOURCE"
    BASE_PATH="${RESULT_PATH%.jsonl}"
    DETAIL_PATH="${BASE_PATH}.json"
    CHECKPOINT_PATH="${BASE_PATH}.checkpoint.json"
    if [ ! -f "$RESULT_PATH" ]; then
        echo "Missing source artifact: $RESULT_PATH" >&2
        FAILED=1
        continue
    fi

    TAG="${MODEL}_${PARADIGM,,}_${SETTING}_${ATTACK_CODE}_${PROMPT_MODE}"
    echo "Rerunning $TAG (dataset=TREC-DL-2019, region=$AWS_REGION)"

    # Preserve the zero-valid artifacts; the fresh run uses the original paths.
    mkdir -p "$BACKUP_DIR"
    for OLD in "$RESULT_PATH" "$DETAIL_PATH" "$CHECKPOINT_PATH"; do
        if [ -e "$OLD" ]; then
            mv "$OLD" "$BACKUP_DIR/$(basename "$OLD")"
        fi
    done

    export GPT_OSS_REASONING_EFFORT="$SETTING"
    set +u
    if "$PYTHON" "LLM_prompt_attack/$SCRIPT" \
        --provider amazon-bedrock \
        --aws_region "$AWS_REGION" \
        --model_name openai.gpt-oss-20b-1:0 \
        --dataset_name msmarco-passage/trec-dl-2019 \
        "${SIZE[@]}" \
        --seed 42 \
        --n_jobs "$N_JOBS" \
        --attack_type "$ATTACK_CODE" \
        --attack_position back \
        --prompt_mode "$PROMPT_MODE" \
        --result_json_path "$RESULT_PATH" \
        --detailed_results "$DETAIL_PATH" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --checkpoint_batch_size 32 \
        2>&1 | tee "${BASE_PATH}.rerun.log"; then
        STATUS=0
    else
        STATUS=$?
        FAILED=1
    fi
    set -u
    echo "$TAG status=$STATUS"
done < <("$PYTHON" - "$CSV_PATH" <<'PY'
import csv, sys
with open(sys.argv[1], newline='', encoding='utf-8') as handle:
    for row in csv.DictReader(handle):
        if row['Model'] == 'GPT-OSS-20B' and int(float(row['Valid attacked'])) == 0:
            print('\t'.join(row[k] for k in ('Source', 'Model', 'Setting', 'Paradigm', 'Attack', 'Prompt', 'Passages')))
PY
)

"$PYTHON" Results/update_ablation_study.py
echo "Backups stored in $BACKUP_DIR"
exit "$FAILED"
