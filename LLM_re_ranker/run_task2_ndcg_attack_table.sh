#!/usr/bin/env bash
set -euo pipefail

# Reproduce the attack-outcome model/attack matrix with Task 2 NDCG@10 scoring.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
RETRIEVAL_PYTHON="${RETRIEVAL_PYTHON:-$PYTHON}"
REGION="${AWS_REGION:-ap-southeast-2}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/LLM_re_ranker/outputs/task2_ndcg_attack_table_$RUN_TAG}"
BEDROCK_MAX_TOKENS="${BEDROCK_MAX_TOKENS:-512}"
GPT_OSS_BEDROCK_MAX_TOKENS="${GPT_OSS_BEDROCK_MAX_TOKENS:-2048}"
RETRIEVAL_DEPTH=1000
RERANK_HITS=100
SETWISE_CHILDREN=3
DRY_RUN=1

usage() {
  cat <<'EOF'
Usage: bash LLM_re_ranker/run_task2_ndcg_attack_table.sh [--execute]

Without --execute, print the planned matrix without calling model APIs.
Set RUN_TAG to reuse the same output directory and resume completed conditions.
The matrix uses the five Bedrock models from the existing Task 2 launcher.
EOF
}

has_top_1000_per_query() {
  awk 'NF >= 6 { counts[$1]++; found = 1 } END { if (!found) exit 1; for (qid in counts) if (counts[qid] < 1000) exit 1 }' "$1"
}

for arg in "$@"; do
  case "$arg" in
    --execute) DRY_RUN=0 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$PROJECT_ROOT"

declare -a MODEL_SPECS=(
  'GPT-OSS-20B|amazon-bedrock|openai.gpt-oss-20b-1:0'
  'Llama-3-8B|amazon-bedrock|meta.llama3-8b-instruct-v1:0'
  'Llama3-70B|amazon-bedrock|meta.llama3-70b-instruct-v1:0'
  'Qwen3-32B|amazon-bedrock|qwen.qwen3-32b-v1:0'
  'Qwen3-4B|amazon-bedrock|qwen.qwen3-4b-v1:0'
)
declare -a ATTACKS=(none so sd qi)
declare -a DATASETS=(
  'msmarco-passage/trec-dl-2019|dl19|LLM_re_ranker/run.msmarco-v1-passage.bm25-default.dl19.txt'
  'msmarco-passage/trec-dl-2020|dl20|LLM_re_ranker/run.msmarco-v1-passage.bm25-default.dl20.txt'
)

if [[ "$DRY_RUN" -eq 0 ]]; then
  command -v "$PYTHON" >/dev/null || { echo "Python executable not found: $PYTHON" >&2; exit 1; }
fi

mkdir -p "$RUN_ROOT"
export AWS_REGION="$REGION"
echo "Run tag: $RUN_TAG"
echo "Output root: $RUN_ROOT"
echo "BM25 retrieval depth: $RETRIEVAL_DEPTH; reranked depth: $RERANK_HITS"
echo "Setwise comparison size: 4 (one parent plus $SETWISE_CHILDREN children)"
echo "Models: ${#MODEL_SPECS[@]}; datasets: ${#DATASETS[@]}; conditions: ${ATTACKS[*]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo 'DRY RUN ONLY. Pass --execute to make model API calls.'
fi

for dataset_spec in "${DATASETS[@]}"; do
  IFS='|' read -r dataset dataset_tag bm25_run <<< "$dataset_spec"
  if [[ ! -f "$bm25_run" ]] || ! has_top_1000_per_query "$bm25_run"; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "Would create top-$RETRIEVAL_DEPTH BM25 run: $bm25_run ($dataset)"
    else
      echo "Building top-$RETRIEVAL_DEPTH BM25 run for $dataset"
      "$RETRIEVAL_PYTHON" retrieval/retrieve_trec_dl19.py \
        --ir-dataset-name "$dataset" --output "$bm25_run" \
        --depth "$RETRIEVAL_DEPTH" --overwrite
    fi
  fi
  if [[ "$DRY_RUN" -eq 0 ]] && ! has_top_1000_per_query "$bm25_run"; then
    echo "BM25 run does not provide at least $RETRIEVAL_DEPTH hits per query: $bm25_run" >&2
    exit 1
  fi

  run_dir="$RUN_ROOT/$dataset_tag"
  mkdir -p "$run_dir"
  for model_spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r model_tag provider model <<< "$model_spec"
    for attack in "${ATTACKS[@]}"; do
      output="$run_dir/${model_tag}.hits${RERANK_HITS}.${attack}.txt"
      marker="$output.complete"
      if [[ -f "$marker" ]]; then
        echo "Already complete: $output"
        continue
      fi
      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "Would run: $model_tag | $dataset_tag | $attack | $provider"
        continue
      fi

      common_args=(
        run --provider "$provider" --model_name_or_path "$model"
        --run_path "$bm25_run" --save_path "$output"
        --ir_dataset_name "$dataset" --hits "$RERANK_HITS"
        --query_length 32 --passage_length 128
        --attack_type "$attack" --attack_position back
      )
      if [[ "$provider" == 'amazon-bedrock' ]]; then
        model_tokens="$BEDROCK_MAX_TOKENS"
        [[ "$model" == openai.gpt-oss* ]] && model_tokens="$GPT_OSS_BEDROCK_MAX_TOKENS"
        common_args+=(--aws_region "$REGION" --bedrock_max_tokens "$model_tokens")
        common_args+=(--invalid_output_policy skip-query)
      fi
      common_args+=(setwise --num_child "$SETWISE_CHILDREN" --method heapsort --k 10)
      echo "Running $model_tag | $dataset_tag | $attack"
      "$PYTHON" LLM_re_ranker/run_attack.py "${common_args[@]}"
      [[ -s "$output" ]] || { echo "Missing/empty run output: $output" >&2; exit 1; }
      touch "$marker"
    done
  done

  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$PYTHON" Results/evaluate_task2_ndcg.py \
      --dataset "$dataset" --run-dir "$run_dir" \
      --output "$run_dir/ndcg_at_10.csv"
  fi
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "Evaluation complete. Per-dataset NDCG@10 CSVs are under $RUN_ROOT."
fi
