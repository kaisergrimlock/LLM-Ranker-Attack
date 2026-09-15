# Checkpoint Isolation Worklog

## Purpose

Ranking evaluations must never resume response records generated with different
model-generation settings. The checkpoint fingerprint includes the model,
provider, evaluation parameters, and these generation settings:

- `qwen_thinking_mode`
- `gpt_oss_reasoning_effort`
- `bedrock_max_tokens`
- Pointwise's explicit `thinking_mode`

## Stale-checkpoint behavior

When `--resume` finds a checkpoint whose fingerprint differs from the current
run, the old checkpoint is not read for phase records and is never overwritten.
The evaluator derives a deterministic SHA-256 suffix from the requested
fingerprint and creates or reuses a sibling checkpoint:

```text
<run>.settings-<12-hex-digest>.checkpoint.json
```

The evaluator prints the differing fields and the effective checkpoint path.
Repeating the same command with the same settings reuses that derived file.
Completed JSONL result files are not removed or modified by checkpoint
isolation.

## Server workflow

Pull the implementation and rerun the same command; no manual checkpoint move
is required:

```bash
cd /research/remote/petabyte/users/$USER/LLM-Ranker-Attack
git pull --ff-only origin main
bash LLM_prompt_attack/run_gptoss_20b_pointwise_doh_dch_resumable.sh
```

For GPT-OSS, the token budget can be selected explicitly:

```bash
GPT_OSS_BEDROCK_MAX_TOKENS=4096 \
bash LLM_prompt_attack/run_gptoss_20b_pointwise_doh_dch_resumable.sh
```

The original stale checkpoint remains available for audit. Derived checkpoints
are stored beside it in the same output directory.

## Validation

Focused checkpoint tests:

```bash
python -m pytest LLM_prompt_attack/tests/test_evaluation_checkpoint.py -q
```

The tests cover GPT-OSS reasoning changes, Qwen thinking changes, token-budget
changes, legacy checkpoints without generation keys, deterministic reuse, and
preservation of the original checkpoint.
