#!/usr/bin/env bash
# Resume only the missing Llama3-70B Listwise keyword-injection runs.
# The shared resumable runner skips the completed 2019 Default run and resumes
# the three incomplete jobs from their existing checkpoints.

set -euo pipefail

PROJECT="${PROJECT:-/research/remote/petabyte/users/${USER}/LLM-Ranker-Attack}"
RUNNER="$PROJECT/LLM_prompt_attack/run_keyword_injection_all_paradigms_resumable.sh"

if [[ ! -f "$RUNNER" ]]; then
    echo "Runner not found: $RUNNER" >&2
    exit 1
fi

cd "$PROJECT"

MODEL_TAGS=Llama3-70B \
PARADIGMS=listwise \
AWS_REGION="${AWS_REGION:-us-east-1}" \
LLAMA70_AWS_REGION="${LLAMA70_AWS_REGION:-us-east-1}" \
RUN_ID=keyword_injection_all_paradigms_resumable \
bash "$RUNNER"
