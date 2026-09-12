"""Estimate prompt-token totals from legacy detail logs without saved prompts."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "LLM_prompt_attack"))

import runtime_environment  # noqa: E402, F401
import ir_datasets  # noqa: E402
from transformers import AutoTokenizer

from calculate_prompt_tokens import (  # noqa: E402
    ATTACK_LABELS,
    MODEL_LABELS,
    MODEL_TOKENIZERS,
    PARADIGM_LABELS,
    PROMPT_LABELS,
    _existing_rows,
    _row_key,
    _row_priority,
    _write_rows,
)
from prompts import (  # noqa: E402
    jailbreak_prompt,
    listwise_jailbreak_prompt,
    listwise_ranking_defense,
    listwise_ranking_defense_qi,
    listwise_ranking_prompt,
    setwise_ranking_defense,
    setwise_ranking_defense_qi,
    setwise_ranking_prompt,
)


def _template(paradigm: str, prompt_mode: str) -> str:
    """Return the evaluator template used for a ranking paradigm."""
    choices = {
        "listwise": {
            "standard": listwise_ranking_prompt,
            "defense": listwise_ranking_defense,
            "defense_qi": listwise_ranking_defense_qi,
        },
        "setwise": {
            "standard": setwise_ranking_prompt,
            "defense": setwise_ranking_defense,
            "defense_qi": setwise_ranking_defense_qi,
        },
    }
    return choices[paradigm][prompt_mode]


def _render(query: str, texts: list[str], paradigm: str, prompt_mode: str) -> str:
    """Render a listwise or setwise evaluator prompt from passage texts."""
    passages = "\n\n".join(
        f"[{chr(65 + index)}] {text}" for index, text in enumerate(texts)
    )
    return _template(paradigm, prompt_mode).format(query=query, passages=passages)


def _count(tokenizer: Any, text: str) -> int:
    """Count tokenizer input IDs without model invocation."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def main() -> int:
    """Append reconstructable legacy token rows to the shared token CSV."""
    output = PROJECT_ROOT / "Results" / "prompt_token_counts.csv"
    rows = _existing_rows(output)
    updated = 0
    tokenizers: dict[str, Any] = {}

    for result_path in sorted(
        (PROJECT_ROOT / "LLM_prompt_attack" / "outputs").rglob("result.jsonl")
    ):
        detail_path = result_path.with_name("detail.json")
        if not detail_path.exists():
            continue
        metadata = json.loads(result_path.read_text(encoding="utf-8").splitlines()[0])
        paradigm = str(metadata.get("ranking_scheme", ""))
        prompt_mode = str(metadata.get("prompt_mode", "standard"))
        attack_type = str(metadata.get("attack_type", ""))
        if paradigm not in {"listwise", "setwise"} or prompt_mode not in {
            "standard",
            "defense",
            "defense_qi",
        }:
            continue
        details = json.loads(detail_path.read_text(encoding="utf-8"))
        if (
            not details
            or "prompt" in details[0]
            or not {"query", "doc_ids", "phase"} <= details[0].keys()
        ):
            continue
        model_name = str(metadata.get("model_name", ""))
        tokenizer_name = MODEL_TOKENIZERS.get(model_name.lower())
        if tokenizer_name is None:
            continue
        tokenizer = tokenizers.setdefault(
            tokenizer_name, AutoTokenizer.from_pretrained(tokenizer_name)
        )
        dataset = ir_datasets.load(str(metadata["dataset_name"]))
        docstore = dataset.docs_store()
        counts: dict[str, list[int]] = defaultdict(list)
        missing: dict[str, int] = defaultdict(int)

        for detail in details:
            phase = "Clean" if detail["phase"] == "original" else "Attacked"
            try:
                texts = [docstore.get(doc_id).text for doc_id in detail["doc_ids"]]
                if any(text is None for text in texts):
                    raise KeyError("missing document")
                if phase == "Clean":
                    prompts = [_render(detail["query"], texts, paradigm, prompt_mode)]
                else:
                    attack_templates = (
                        listwise_jailbreak_prompt
                        if paradigm == "listwise"
                        else jailbreak_prompt
                    )
                    payload = attack_templates[attack_type].format(
                        query=detail["query"]
                    )
                    prompts = [
                        _render(
                            detail["query"],
                            [
                                text + payload if index == target else text
                                for index, text in enumerate(texts)
                            ],
                            paradigm,
                            prompt_mode,
                        )
                        for target in range(len(texts))
                    ]
                counts[phase].append(
                    round(
                        sum(_count(tokenizer, prompt) for prompt in prompts)
                        / len(prompts)
                    )
                )
            except (KeyError, AttributeError):
                missing[phase] += 1

        for phase, values in counts.items():
            row = {
                "Dataset": str(metadata["dataset_name"])
                .rsplit("/", 1)[-1]
                .replace("trec-dl-", "TREC-DL-")
                .replace("2019", "2019")
                .replace("2020", "2020"),
                "Model": MODEL_LABELS.get(model_name.lower(), model_name),
                "Paradigm": PARADIGM_LABELS[paradigm],
                "Attack": ATTACK_LABELS[attack_type],
                "Position": str(metadata.get("attack_position", "")),
                "Prompt": PROMPT_LABELS.get(prompt_mode, prompt_mode),
                "Phase": phase,
                "Prompts counted": len(values),
                "Prompts missing text": missing[phase],
                "Total tokens": sum(values),
                "Mean tokens per prompt": sum(values) / len(values),
                "Median tokens per prompt": "",
                "P95 tokens per prompt": "",
                "Minimum tokens": min(values),
                "Maximum tokens": max(values),
                "Mean delta from clean": "",
                "Date": str(metadata.get("date", "")),
                "Source": (
                    "reconstructed-average-target:"
                    f"{detail_path.relative_to(PROJECT_ROOT)}"
                ),
            }
            key = _row_key(row)
            if key not in rows or _row_priority(row) > _row_priority(rows[key]):
                rows[key] = row
                updated += 1
    _write_rows(output, rows)
    print(f"Updated {updated} reconstructed token rows in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
