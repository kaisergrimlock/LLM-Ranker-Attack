---
name: azure-ranking-evaluation
description: Run, troubleshoot, and scale this repository's GPT-5.6 setwise ranking-attack smoke tests and evaluations through an Azure OpenAI v1 endpoint. Use for deployment checks, environment setup, MS MARCO commands, result inspection, and Azure-specific failures; do not use for Bedrock or local-GPU runs.
---

# Azure Ranking Evaluation

Use the repository's `azure-openai` provider and the Python interpreter from the project environment. Keep credentials out of commands, logs, results, and Git.

## Establish the run context

On the Linux evaluation server, resolve paths rather than assuming the active Conda `base` interpreter has the dependencies:

```bash
export PROJECT="${PROJECT:-$(git rev-parse --show-toplevel)}"
export ENV="${ENV:-$PROJECT/.conda-bedrock}"
export PYTHON="${PYTHON:-$ENV/bin/python}"

cd "$PROJECT"
set -a
source .env
set +a

test -x "$PYTHON"
test -n "$AZURE_OPENAI_API_KEY"
test -n "$AZURE_OPENAI_BASE_URL"
test -n "$AZURE_OPENAI_DEPLOYMENT"
```

Expected `.env` keys are:

```dotenv
AZURE_OPENAI_API_KEY=replace-with-secret
AZURE_OPENAI_BASE_URL=https://resource-name.openai.azure.com/openai/v1
AZURE_OPENAI_DEPLOYMENT=gpt-5.6-terra
```

The root `.gitignore` excludes `.env` and `.env.*`. Confirm that remains true before changing a credential file. Never print the key as a diagnostic; test only whether it is non-empty.

## Verify the GPT-5.6 deployment

An Azure `/models` result is only a catalog hint. A successful inference request proves that the resource has a callable deployment under the supplied name. Run at most one probe per candidate unless the user asks for more:

```bash
"$PYTHON" \
  "$PROJECT/.agents/skills/azure-ranking-evaluation/scripts/probe_azure_deployment.py"
```

To inspect GPT-5.6 catalog candidates before probing the configured deployment:

```bash
"$PYTHON" \
  "$PROJECT/.agents/skills/azure-ranking-evaluation/scripts/probe_azure_deployment.py" \
  --list-filter 'gpt-5.6' \
  --no-probe
```

`DeploymentNotFound` means the endpoint and key reached Azure but that deployment name is unavailable. Obtain the actual deployment name from the Azure resource owner; do not keep guessing from catalog IDs. The live probe uses the same GPT-5.6 settings as the evaluator: Responses API, reasoning effort `none`, low verbosity, and a small output budget.

## Run offline checks first

From the repository root:

```bash
"$PYTHON" -m unittest discover -s LLM_prompt_attack/tests -v
```

This validates request translation without spending Azure tokens. If `ir_datasets` is missing, the wrong interpreter is being used or the project environment is incomplete; do not install into Conda `base` as a workaround.

## Run a setwise smoke test

Use 10 sampled sets, one worker, unique output names, and pipe failure propagation:

```bash
cd "$PROJECT/LLM_prompt_attack"
mkdir -p outputs
RUN_ID="azure-smoke-$(date +%Y%m%d-%H%M%S)"
set -o pipefail

"$PYTHON" setwise_ranking_attack_openai.py \
  --provider azure-openai \
  --base_url "$AZURE_OPENAI_BASE_URL" \
  --model_name "$AZURE_OPENAI_DEPLOYMENT" \
  --dataset_name msmarco-passage/trec-dl-2019 \
  --attack_type so \
  --attack_position back \
  --num_sets 10 \
  --set_size 4 \
  --seed 42 \
  --n_jobs 1 \
  --result_json_path "outputs/$RUN_ID.jsonl" \
  --detailed_results "outputs/$RUN_ID-details.json" \
  2>&1 | tee "outputs/$RUN_ID.log"
```

Treat the smoke test as successful only when the command exits zero, both phases complete, `total_queries` is nonzero, and the valid-ranking counts are plausible. Attack success rate alone does not establish a healthy run.

Inspect the summary without assuming a JSONL file has only one record:

```bash
tail -n 1 "outputs/$RUN_ID.jsonl" | "$PYTHON" -m json.tool
```

Detailed results contain dataset text and attack prompts. Store and share them according to the dataset and project rules.

## Scale the evaluation

After the 10-set run succeeds, rerun with an explicitly chosen sample count and conservative concurrency. Use a new output prefix so the summary is not appended to an older experiment:

```bash
cd "$PROJECT/LLM_prompt_attack"
NUM_SETS="${NUM_SETS:-100}"
N_JOBS="${N_JOBS:-2}"
RUN_ID="azure-${AZURE_OPENAI_DEPLOYMENT}-${NUM_SETS}-$(date +%Y%m%d-%H%M%S)"
set -o pipefail

"$PYTHON" setwise_ranking_attack_openai.py \
  --provider azure-openai \
  --base_url "$AZURE_OPENAI_BASE_URL" \
  --model_name "$AZURE_OPENAI_DEPLOYMENT" \
  --dataset_name msmarco-passage/trec-dl-2019 \
  --attack_type so \
  --attack_position back \
  --num_sets "$NUM_SETS" \
  --set_size 4 \
  --seed 42 \
  --n_jobs "$N_JOBS" \
  --result_json_path "outputs/$RUN_ID.jsonl" \
  --detailed_results "outputs/$RUN_ID-details.json" \
  2>&1 | tee "outputs/$RUN_ID.log"
```

For a long SSH run, use the server's approved persistent-session mechanism such as `tmux`; do not put the API key directly in a process argument.

`--num_sets N` does **not** mean "evaluate every qrel once." `prepare_sets` randomly samples eligible query/document sets and can select a query more than once. A healthy run normally makes approximately `2 * N` model calls: one original and one attacked call per valid set, with additional calls possible from retries. If the requested experiment requires exhaustive, unique qrel coverage, change the dataset preparation logic and define the unit of evaluation before launching it.

Azure pricing can depend on the resource agreement and deployment. Estimate cost from measured input/output token usage and the applicable Azure rates; do not infer cost from call count alone. The current result files do not record API token usage, so add usage accounting or run a separately instrumented representative sample when a defensible estimate is required.

## Diagnose common failures

- `DeploymentNotFound` or HTTP 404: `--model_name` is not a callable deployment on this resource. Verify it with the probe and ask the resource owner for the deployment name.
- HTTP 401/403: confirm the sourced key belongs to the endpoint and that access policy permits inference. Do not reveal the key while testing.
- HTTP 429 or transient service errors: reduce `N_JOBS`, preserve the log, and retry only within the scope the user authorized.
- `No module named ir_datasets`: use `"$ENV/bin/python"` and install dependencies into that environment only if the user asked for setup.
- zero valid rankings: inspect raw responses in the detailed JSON. The ranker accepts one label from `A` through the set-size label; prose-only or empty answers are invalid.
- probe succeeds but evaluator fails: compare the probe request with `RankingClient.generate` and preserve the GPT-5.6 Responses API settings covered by `LLM_prompt_attack/tests/test_llm_client.py`.
