---
name: rmit-gpu-ranking
description: Set up, troubleshoot, and run this repository's Qwen/vLLM ranking-attack experiments on RMIT SEG or SCT shared GPU servers. Use for server access, Slurm allocations, petabyte storage, P100/V100 compatibility, local OpenAI-compatible serving, smoke tests, and full GPU jobs; do not use for Bedrock, Azure OpenAI, or an unrelated GPU environment.
---

# RMIT GPU Ranking

Use the repository's canonical procedure in
[`docs/rmit-gpu-server-qwen3-4b.md`](../../../docs/rmit-gpu-server-qwen3-4b.md).
Read it completely before proposing commands or changing server job files.

## Preserve the infrastructure constraints

- Treat SEG and SCT as shared services. Inspect the live GPU before direct SEG
  execution; on SCT, keep vLLM and the attack inside the same Slurm allocation.
- Store repositories, environments, model caches, `ir_datasets`, and outputs
  under `/research/remote/petabyte/users/$USER`, not the home directory.
- Inspect `/research/remote/collections` and prefer a symbolic link before
  downloading or copying an existing collection.
- Current vLLM requires compute capability 7.0 or later. V100 qualifies; P100
  does not. Verify with PyTorch rather than inferring the assigned GPU.
- Treat glibc as a first-class compatibility constraint. A V100 with glibc 2.17
  cannot use xFormers wheels tagged `manylinux_2_28`.
- On the listed P100/V100 hardware, use FP16 (`--dtype half`), not BF16.
- Bind vLLM to loopback unless the user has an approved secured network design.
- Disable Qwen3 thinking for these ranking-label experiments.
- Never use a broad `pkill` cleanup on a shared node. Track and stop only the
  server PID created by the current job.
- Do not place credentials in job scripts, logs, results, or environment files.

## Establish context before setup

Determine which service the user has access to: the SCT Slurm cluster, the SEG
V100 node, or only the SEG P100 nodes. Do not invent the SCT login hostname.
Capture `hostname`, `nvidia-smi`, `getconf GNU_LIBC_VERSION`, Python version,
container runtimes, `nvcc`, scheduler commands, `/tmp` space, and petabyte
storage availability. Match vLLM/PyTorch to the observed driver; the CUDA value
reported by `nvidia-smi` does not prove that a compiler or headers are installed.

When package installation is requested, use a clean environment under petabyte
storage and Python 3.11. Do not mutate Conda `base` or use its Python 3.13. Run
long installations in `tmux`, log them with `tee`, and capture
`${PIPESTATUS[0]}` immediately after the pipeline.

Select one installation route from the canonical guide:

- On glibc 2.28 or newer, prefer matching binary wheels.
- On glibc 2.17 with Apptainer or Singularity, prefer a supported container.
- On the SEG V100 with glibc 2.17 and no container, use the documented pinned
  stack: Conda binaries for NumPy/SciPy/Pillow, Conda CUDA/GCC tools,
  Torch 2.6.0+cu124, source-built xFormers 0.0.29.post2 for `sm_70`, then
  vLLM 0.8.5.post1 under constraints. Treat only the xFormers build as verified
  until the final vLLM server and ranking smoke test have also passed.

Do not run the repository's current `requirements.in` unchanged on that legacy
node: its vLLM 0.11.0 and NumPy 2 pins are not the tested V100 path. Do not omit
xFormers on Volta or silently upgrade it; the documented vLLM version selects
xFormers on capability 7.0, and newer xFormers releases have dropped V100.

When source-building xFormers, first verify Torch imports and expose the
pip-installed NVIDIA include/library directories. Require the `cusparse.h`
preprocessor check to pass before compiling. Use `MAX_JOBS=2`, or 1 under
memory pressure, on the shared node.

## Run progression

1. Verify access, GPU capability, availability, and permitted storage.
2. Configure persistent Hugging Face, vLLM, package, and `ir_datasets` caches
   under the user's petabyte directory. Create the documented credential-free
   activation file and source it after reconnecting or inside a batch job.
3. Choose the installation route from glibc/container/GPU evidence; create or
   activate a clean Linux environment and verify every pinned import plus CUDA.
4. Start `Qwen/Qwen3-4B` as `Qwen3-4B` with FP16, 8K context, two sequences,
   loopback binding, and thinking disabled.
5. Verify `/v1/models`, then run 16 pairs with one client worker.
6. Inspect clean and attacked outputs and GPU memory before scaling.
7. Increase one resource dimension at a time and run 4,096 pairs only after the
   smoke test passes.
8. Use `sbatch` for full runs; preserve logs, package versions, node/GPU data,
   model revision, and experiment metadata beside the outputs.

Treat a successful HTTP readiness check as necessary but insufficient. A valid
smoke run must complete both evaluation phases, produce nonzero valid rankings,
and write unique result and detailed-output files.

If installation fails, preserve the complete log and find the first native
`fatal error`, `FAILED`, missing header, or killed compiler process. Do not use
the final setuptools traceback as the root cause. If installation appears
stalled at `Installing collected packages`, inspect the process and filesystem
I/O from another tmux pane before interrupting it.

## Stop conditions

Stop and report the evidence when:

- only a P100 is available for a requested vLLM run;
- the assigned GPU is occupied outside a scheduler allocation;
- CUDA is unavailable or compute capability is below 7.0;
- the operating system cannot use a required wheel and neither the documented
  source-build route nor an approved container is available;
- the requested context/concurrency does not fit GPU memory;
- the central collection and intended cache location cannot be determined; or
- continuing requires server access, storage permission, or scheduler policy
  that the user has not obtained.

Offer a bounded next diagnostic rather than silently changing inference engine,
precision, model checkpoint, dataset, or evaluation parameters.
