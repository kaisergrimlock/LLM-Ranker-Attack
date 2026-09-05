#!/usr/bin/env bash
set -euo pipefail

# =========================
# Configuration
# =========================
MODELS=(
  "Qwen/Qwen3-1.7B"
  # "Qwen/Qwen3-8B"
  # "Qwen/Qwen3-14B"
  # "Qwen/Qwen3-32B"
  # "google/gemma-3-12b-it"
  # "google/gemma-3-27b-it"
)

DATASETS=(
  "msmarco-passage/trec-dl-2019"
  "msmarco-passage/trec-dl-2020"  
  # "beir/trec-covid"
  # "beir/webis-touche2020/v2"
  # "beir/scifact/test"
  # "beir/dbpedia-entity/test"
)

SETTINGS=(setwise listwise pairwise) # Options: setwise, listwise, pairwise
ATTACKS=(so sd) # Options: so (DOH), sd (DCH), qi (query injection)
POSITIONS=(back front) # Options: back, front
PROMPT_MODES=(${PROMPT_MODES:-standard}) # Pairwise options: standard, defense

# =========================
# Experiment Parameters
# =========================
NUM_SAMPLES="${NUM_SAMPLES:-4096}"
SET_SIZE="${SET_SIZE:-4}"
N_JOBS="${N_JOBS:-4}"

# =========================
# vLLM Server Parameters
# =========================
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
BASE_PORT="${BASE_PORT:-8000}"

mkdir -p logs outputs

sanitize() {
  local s="${1//\//-}"
  s="${s// /_}"
  echo "$s"
}

generate_run_script() {
  local model="$1"
  local dataset="$2"
  local setting="$3"

  local model_short="${model##*/}"
  local model_tag
  model_tag="$(sanitize "$model_short")"
  local dataset_tag
  dataset_tag="$(sanitize "$dataset")"

  local script_tag="${model_tag}_${dataset_tag}_${setting}"
  local script_file="run_${script_tag}.sh"

  cat > "${script_file}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "Starting experiment: MODEL_NAME_PLACEHOLDER on DATASET_PLACEHOLDER (SETTING_PLACEHOLDER)"
echo "============================================================"

MODEL_NAME="MODEL_PLACEHOLDER"
MODEL_SHORT="MODEL_SHORT_PLACEHOLDER"
DATASET="DATASET_PLACEHOLDER"
SETTING="SETTING_PLACEHOLDER"

NUM_SAMPLES=NUM_SAMPLES_PLACEHOLDER
SET_SIZE=SET_SIZE_PLACEHOLDER
N_JOBS=N_JOBS_PLACEHOLDER

GPU_MEMORY_UTILIZATION=GPU_MEM_PLACEHOLDER
MAX_MODEL_LEN=MAX_LEN_PLACEHOLDER
MAX_NUM_SEQS=MAX_SEQS_PLACEHOLDER
SERVER_WAIT_TIMEOUT=SERVER_TIMEOUT_PLACEHOLDER

ATTACKS=(ATTACKS_PLACEHOLDER)
POSITIONS=(POSITIONS_PLACEHOLDER)
PROMPT_MODES=(PROMPT_MODES_PLACEHOLDER)

PORT=BASE_PORT_PLACEHOLDER
BASE_URL="http://localhost:${PORT}/v1"

SERVER_PID=""
SERVER_LOG="logs/vllm_server_${MODEL_SHORT}_${PORT}.log"

cleanup() {
  echo "[Cleanup] Stopping vLLM server..."
  if [ -n "${SERVER_PID}" ]; then
    kill ${SERVER_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
  fi
  pkill -f "vllm.entrypoints" 2>/dev/null || true
  echo "[Cleanup] Done."
}
trap cleanup EXIT INT TERM

start_vllm_server() {
  echo "[Server] Starting vLLM on port ${PORT}"
  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --port ${PORT} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --dtype bfloat16 \
    --trust-remote-code \
    --served-model-name "${MODEL_SHORT}" \
    > "${SERVER_LOG}" 2>&1 &

  SERVER_PID=$!
  local waited=0
  while [ ${waited} -lt ${SERVER_WAIT_TIMEOUT} ]; do
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
      echo "[Server] Process died unexpectedly"
      tail -200 "${SERVER_LOG}" || true
      exit 1
    fi
    if curl -sf --max-time 5 "${BASE_URL}/models" >/dev/null 2>&1; then
      echo "[Server] Ready after ${waited}s"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "[Server] Startup timeout"
  tail -200 "${SERVER_LOG}" || true
  exit 1
}

start_vllm_server

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="outputs/${DATASET}/${MODEL_SHORT}/${SETTING}"
mkdir -p "${OUT_DIR}"

run_experiment() {
  local attack="$1"
  local pos="$2"
  local prompt_mode="$3"

  local prompt_suffix=""
  if [ "${SETTING}" = "pairwise" ]; then
    prompt_suffix="_${prompt_mode}"
  fi

  local out_file="${OUT_DIR}/result_${MODEL_SHORT}_${SETTING}_${attack}_${pos}${prompt_suffix}_${TIMESTAMP}.jsonl"
  local detail_file="${OUT_DIR}/detail_${MODEL_SHORT}_${SETTING}_${attack}_${pos}${prompt_suffix}_${TIMESTAMP}.json"

  echo ""
  echo "=== Running: ${SETTING} | attack=${attack} | position=${pos} ==="
  echo "Output: ${out_file}"

  COMMON_ARGS=(
    --model_name "${MODEL_SHORT}"
    --base_url "${BASE_URL}"
    --attack_type "${attack}"
    --attack_position "${pos}"
    --dataset_name "${DATASET}"
    --n_jobs ${N_JOBS}
    --result_json_path "${out_file}"
    --tokenizer_model "${MODEL_NAME}"
    --detailed_results "${detail_file}"
  )

  case "${SETTING}" in
    setwise)
      python setwise_ranking_attack_openai.py "${COMMON_ARGS[@]}" --num_sets ${NUM_SAMPLES} --set_size ${SET_SIZE}
      ;;
    listwise)
      python listwise_ranking_attack_openai.py "${COMMON_ARGS[@]}" --num_sets ${NUM_SAMPLES} --set_size ${SET_SIZE}
      ;;
    pairwise)
      python pairwise_ranking_attack_openai.py "${COMMON_ARGS[@]}" \
        --num_pairs ${NUM_SAMPLES} \
        --prompt_mode "${prompt_mode}"
      ;;
  esac
}

for attack in "${ATTACKS[@]}"; do
  for pos in "${POSITIONS[@]}"; do
    if [ "${attack}" = "qi" ] && [ "${pos}" != "back" ]; then
      continue
    fi
    if [ "${SETTING}" = "pairwise" ]; then
      for prompt_mode in "${PROMPT_MODES[@]}"; do
        run_experiment "${attack}" "${pos}" "${prompt_mode}"
      done
    else
      run_experiment "${attack}" "${pos}" "standard"
    fi
  done
done

echo ""
echo "============================================================"
echo "All experiments completed for: ${MODEL_SHORT} | ${DATASET} | ${SETTING}"
echo "============================================================"
EOF

  # Replace placeholders
  sed -i "s|MODEL_NAME_PLACEHOLDER|${model_short}|g" "${script_file}"
  sed -i "s|MODEL_PLACEHOLDER|${model}|g" "${script_file}"
  sed -i "s|MODEL_SHORT_PLACEHOLDER|${model_short}|g" "${script_file}"
  sed -i "s|DATASET_PLACEHOLDER|${dataset}|g" "${script_file}"
  sed -i "s|SETTING_PLACEHOLDER|${setting}|g" "${script_file}"
  sed -i "s|NUM_SAMPLES_PLACEHOLDER|${NUM_SAMPLES}|g" "${script_file}"
  sed -i "s|SET_SIZE_PLACEHOLDER|${SET_SIZE}|g" "${script_file}"
  sed -i "s|N_JOBS_PLACEHOLDER|${N_JOBS}|g" "${script_file}"
  sed -i "s|GPU_MEM_PLACEHOLDER|${GPU_MEMORY_UTILIZATION}|g" "${script_file}"
  sed -i "s|MAX_LEN_PLACEHOLDER|${MAX_MODEL_LEN}|g" "${script_file}"
  sed -i "s|MAX_SEQS_PLACEHOLDER|${MAX_NUM_SEQS}|g" "${script_file}"
  sed -i "s|SERVER_TIMEOUT_PLACEHOLDER|${SERVER_WAIT_TIMEOUT}|g" "${script_file}"
  sed -i "s|ATTACKS_PLACEHOLDER|${ATTACKS[*]}|g" "${script_file}"
  sed -i "s|POSITIONS_PLACEHOLDER|${POSITIONS[*]}|g" "${script_file}"
  sed -i "s|PROMPT_MODES_PLACEHOLDER|${PROMPT_MODES[*]}|g" "${script_file}"
  sed -i "s|BASE_PORT_PLACEHOLDER|${BASE_PORT}|g" "${script_file}"

  chmod +x "${script_file}"
  echo "${script_file}"
}

count=0
for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      script_file="$(generate_run_script "$model" "$dataset" "$setting")"
      count=$((count + 1))
      echo "Generated: ${script_file}"
    done
  done
done

echo ""
echo "============================================================"
echo "Total scripts generated: ${count}"
echo "============================================================"
echo ""
echo "To run an experiment, execute any generated script, e.g.:"
echo "  bash run_Qwen3-1.7B_beir-trec-covid_setwise.sh"
echo ""
