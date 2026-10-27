# AutoJev listwise reranker worklog

## 2026-10-27

- Added `LLM_re_ranker/llmrankers/autojev_listwise.py`.
- The adapter uses the existing Jev/AutoJev `v1/systemone` HTTP API.
- Each sliding window is submitted as one multi-choice relevance question;
  returned choice probabilities determine the window order.
- Preserved the repository listwise controls: `window_size`, `step_size`, and
  `num_repeat`.
- Added `--provider autojev ... listwise` routing in `run_attack.py`.
- This is API-backed listwise window ranking; the endpoint does not expose a
  direct permutation response.
- Next validation: run one query with the AutoJev server, inspect the TREC run
  file, then submit the full 43-query TREC-DL 2019 job through Slurm.
