#!/usr/bin/env bash
#SBATCH --job-name=llama3-70b-missing-doh-dch
#SBATCH --partition=SCT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=llama3-70b-missing-doh-dch-%j.log

# Complete only the Llama 3 70B DOH/DCH evaluations absent from
# Results/attack_outcomes.csv as of 2026-09-10. Invalidation or clean-stage
# exclusions remain in the result metadata and are included by the outcome
# summary as failed requested passages.
set -uo pipefail

RESEARCH_ROOT="${RESEARCH_ROOT:-/research/remote/petabyte/users/$USER}"
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$RESEARCH_ROOT/LLM-Ranker-Attack}}"
PYTHON="${PYTHON:-$RESEARCH_ROOT/environments/llama3-8b/bin/python}"
AWS_REGION="${AWS_REGION:-us-west-2}"
AWS_PROFILE="${AWS_PROFILE:-default}"
NUM_SAMPLES="${NUM_SAMPLES:-4096}"
N_JOBS="${N_JOBS:-1}"
RUN_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)_$$}"

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
export BEDROCK_MAX_TOKENS="${BEDROCK_MAX_TOKENS:-1024}"

if ! aws sts get-caller-identity --region "$AWS_REGION" >/dev/null; then
    echo "AWS credentials are unavailable. Before sbatch, run:" >&2
    echo "  aws sso login --profile $AWS_PROFILE --use-device-code" >&2
    exit 1
fi

cd "$PROJECT"
RUN_DIR="LLM_prompt_attack/outputs/Llama3-70B_missing_doh_dch_${RUN_ID}"
mkdir -p "$RUN_DIR"
printf 'dataset\tattack\tparadigm\tprompt\texit_status\n' > "$RUN_DIR/status.tsv"
hostname > "$RUN_DIR/environment.log"
date --iso-8601=seconds >> "$RUN_DIR/environment.log"
aws sts get-caller-identity --region "$AWS_REGION" >> "$RUN_DIR/environment.log"
"$PYTHON" -m pip freeze > "$RUN_DIR/packages.txt"
git rev-parse HEAD > "$RUN_DIR/source_revision.txt"
cp LLM_prompt_attack/prompts.py "$RUN_DIR/prompts.py"

run_evaluation() {
    local dataset="$1"
    local attack="$2"
    local scheme="$3"
    local prompt_mode="$4"
    local sample_args=()
    local tag="${dataset##*/}_${scheme}_${attack}_${prompt_mode}"
    local status

    if [ "$scheme" = "pairwise" ]; then
        sample_args=(--pos_rel 3 --neg_rel 0 --num_pairs "$NUM_SAMPLES")
    else
        sample_args=(--num_sets "$NUM_SAMPLES" --set_size 4)
    fi

    echo "Running $tag"
    if "$PYTHON" "LLM_prompt_attack/${scheme}_ranking_attack_openai.py" \
        --provider amazon-bedrock --aws_region "$AWS_REGION" \
        --model_name meta.llama3-70b-instruct-v1:0 \
        --tokenizer_model NousResearch/Meta-Llama-3-70B-Instruct \
        --dataset_name "$dataset" --seed 42 --n_jobs "$N_JOBS" \
        --attack_type "$attack" --attack_position back --prompt_mode "$prompt_mode" \
        "${sample_args[@]}" \
        --result_json_path "$RUN_DIR/result_Llama3-70B_${tag}.jsonl" \
        --detailed_results "$RUN_DIR/detail_Llama3-70B_${tag}.json" \
        2>&1 | tee "$RUN_DIR/run_Llama3-70B_${tag}.log"; then
        status=0
    else
        status=$?
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$dataset" "$attack" "$scheme" "$prompt_mode" "$status" >> "$RUN_DIR/status.tsv"
    "$PYTHON" Results/update_attack_outcomes.py
    return "$status"
}

already_completed() {
    local dataset="$1"
    local attack="$2"
    local scheme="$3"
    local prompt_mode="$4"
    local year="${dataset##*-}"
    local attack_label prompt_label scheme_label

    case "$attack" in
        so) attack_label='DOH' ;;
        sd) attack_label='DCH' ;;
    esac
    case "$prompt_mode" in
        standard) prompt_label='Default' ;;
        defense) prompt_label='Defense' ;;
    esac
    scheme_label="${scheme^}"

    awk -F, \
        -v dataset="TREC-DL-$year" \
        -v attack="$attack_label" \
        -v scheme="$scheme_label" \
        -v prompt="$prompt_label" \
        'NR > 1 && $1 == dataset && $2 == "Llama3 70B" && $3 == scheme && $4 == attack && $5 == prompt { found = 1 } END { exit !found }' \
        Results/attack_outcomes.csv
}

# 24 combinations are possible: two datasets, two attacks, three paradigms,
# and standard/defense prompts. The nine already completed runs are omitted.
MISSING_RUNS=(
    'msmarco-passage/trec-dl-2019 so pairwise standard'
    'msmarco-passage/trec-dl-2019 so pairwise defense'
    'msmarco-passage/trec-dl-2019 so setwise standard'
    'msmarco-passage/trec-dl-2019 so setwise defense'
    'msmarco-passage/trec-dl-2019 sd pairwise defense'
    'msmarco-passage/trec-dl-2019 sd setwise defense'
    'msmarco-passage/trec-dl-2019 sd listwise defense'
    'msmarco-passage/trec-dl-2020 so setwise standard'
    'msmarco-passage/trec-dl-2020 so setwise defense'
    'msmarco-passage/trec-dl-2020 so listwise standard'
    'msmarco-passage/trec-dl-2020 so listwise defense'
    'msmarco-passage/trec-dl-2020 sd pairwise standard'
    'msmarco-passage/trec-dl-2020 sd pairwise defense'
    'msmarco-passage/trec-dl-2020 sd setwise defense'
    'msmarco-passage/trec-dl-2020 sd listwise defense'
)

FAILED=0
for run in "${MISSING_RUNS[@]}"; do
    read -r dataset attack scheme prompt_mode <<< "$run"
    if already_completed "$dataset" "$attack" "$scheme" "$prompt_mode"; then
        echo "Skipping completed ${dataset##*/}_${scheme}_${attack}_${prompt_mode}"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$dataset" "$attack" "$scheme" "$prompt_mode" 'SKIPPED_EXISTING' \
            >> "$RUN_DIR/status.tsv"
        continue
    fi
    if ! run_evaluation "$dataset" "$attack" "$scheme" "$prompt_mode"; then
        FAILED=1
    fi
done

echo "Completed missing-run batch; status: $RUN_DIR/status.tsv"
exit "$FAILED"
