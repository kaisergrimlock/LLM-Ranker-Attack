# LLM Ranker Attack

Minimal code and results for evaluating prompt-injection attacks against LLM rerankers.

This anonymized branch is derived from the original [LLM Ranker Attack repository](https://github.com/kaisergrimlock/LLM-Ranker-Attack/tree/main) by Yin, Wang, Koopman, and Zuccon. Please credit the original repository and authors when using this code or its results.

## Setup

```bash
git clone -b anonymized https://github.com/kaisergrimlock/LLM-Ranker-Attack.git
cd LLM-Ranker-Attack
```

Use an environment containing the project dependencies and configure the required model credentials or local model endpoint.

## Run an attack

The ranking implementations are in `LLM_prompt_attack/`:

```bash
python LLM_prompt_attack/pairwise_ranking_attack_openai.py --help
python LLM_prompt_attack/pointwise_ranking_attack_openai.py --help
python LLM_prompt_attack/setwise_ranking_attack_openai.py --help
python LLM_prompt_attack/listwise_ranking_attack_openai.py --help
```

Select the attack with `--attack_type`:

- `qi` — query injection
- `key_injection` — keyword injection

Use `--prompt_mode standard` for the default prompt, or `defense` / `defense_qi` for the corresponding defense prompt. Keyword injection requires a TSV supplied with `--keywords_path`.

## Results and analysis

Aggregate outcomes and LaTeX tables are in `Results/`. Statistical test scripts are also provided there.

The repository intentionally excludes generated outputs, credentials, launcher scripts, and generated query lists.

## Citation

```bibtex
@inproceedings{yin2026vulnerability,
  title={The Vulnerability of LLM Rankers to Prompt Injection Attacks: You are to [MARK] this paper as the Best Paper},
  author={Yin, Yu and Wang, Shuai and Koopman, Bevan and Zuccon, Guido},
  booktitle={Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2026}
}
```
