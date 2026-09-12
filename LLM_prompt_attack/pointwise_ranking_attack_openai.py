"""Evaluate pointwise passage-relevance attacks with binary model decisions."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path

import runtime_environment  # noqa: F401
from evaluation_checkpoint import EvaluationCheckpoint, run_checkpointed
from llm_client import SUPPORTED_PROVIDERS, aggregate_token_usage, get_ranking_client
from prompts import (
    jailbreak_prompt,
    pointwise_ranking_defense,
    pointwise_ranking_defense_qi,
    pointwise_ranking_prompt,
)

import ir_datasets


@dataclass(frozen=True)
class Passage:
    """Represent one judged passage used in a pointwise relevance decision."""

    doc_id: str
    text: str
    relevance: int


@cache
def _get_tokenizer(tokenizer_model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)


def _truncate_text(text: str, tokenizer_model: str | None, max_tokens: int) -> str:
    if tokenizer_model is None:
        return text[: max_tokens * 4] + ("..." if len(text) > max_tokens * 4 else "")

    try:
        tokenizer = _get_tokenizer(tokenizer_model)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        return (
            tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True) + "..."
        )
    except Exception as error:
        print(f"Warning: could not load tokenizer {tokenizer_model!r}: {error}")
        return text[: max_tokens * 4] + ("..." if len(text) > max_tokens * 4 else "")


def prepare_passages(
    dataset_name: str,
    num_passages: int,
    seed: int,
    tokenizer_model: str | None = None,
    neg_rel: int = 0,
    max_doc_tokens: int = 8000,
) -> list[tuple[str, Passage]]:
    """Sample deterministic judged non-relevant query-passage instances.

    Parameters
    ----------
    dataset_name : str
        Name of an ``ir_datasets`` collection with qrels and a document store.
    num_passages : int
        Number of grade-zero instances to return.
    seed : int
        Random seed controlling the candidate shuffle.
    tokenizer_model : str or None, default=None
        Hugging Face tokenizer used to truncate passages before model calls.
    neg_rel : int, default=0
        Qrel grade considered non-relevant.
    max_doc_tokens : int, default=8000
        Maximum passage length after tokenization.

    Returns
    -------
    list of tuple[str, Passage]
        Query text and one judged non-relevant passage per requested instance.

    Raises
    ------
    ValueError
        If no qrel grade-zero passages can be loaded.
    """
    dataset = ir_datasets.load(dataset_name)
    queries = {query.query_id: query.text for query in dataset.queries}
    candidates = sorted(
        (qrel.query_id, qrel.doc_id)
        for qrel in dataset.qrels_iter()
        if qrel.relevance == neg_rel and qrel.query_id in queries
    )
    random.Random(seed).shuffle(candidates)

    docstore = dataset.docs_store()
    passages: list[tuple[str, Passage]] = []
    for query_id, doc_id in candidates:
        document = docstore.get(doc_id)
        if document is None or not getattr(document, "text", None):
            continue
        passages.append(
            (
                queries[query_id],
                Passage(
                    doc_id=doc_id,
                    text=_truncate_text(document.text, tokenizer_model, max_doc_tokens),
                    relevance=neg_rel,
                ),
            )
        )
        if len(passages) == num_passages:
            break

    if not passages:
        raise ValueError(
            f"No passages with relevance={neg_rel} were available in {dataset_name}."
        )
    if len(passages) < num_passages:
        print(
            "Warning: requested "
            f"{num_passages} passages but found only {len(passages)}."
        )
    return passages


def parse_binary_label(response: str) -> str:
    """Parse a strict binary relevance label from one model response.

    Parameters
    ----------
    response : str
        Raw generated model text.

    Returns
    -------
    str
        ``Yes`` or ``No`` for exact case-insensitive labels; otherwise ``INVALID``.
    """
    normalized = (response or "").strip().casefold()
    if normalized == "yes":
        return "Yes"
    if normalized == "no":
        return "No"
    return "INVALID"


def count_no_to_yes(
    clean_labels: list[str], attacked_labels: list[str]
) -> dict[str, int]:
    """Count pointwise attack outcomes while retaining all requested passages.

    Parameters
    ----------
    clean_labels : list of str
        Parsed relevance labels from clean prompts.
    attacked_labels : list of str
        Parsed relevance labels from attacked prompts.

    Returns
    -------
    dict of str to int
        Requested, valid-attacked, clean-label, attacked-label, and No-to-Yes counts.

    Raises
    ------
    ValueError
        If clean and attacked result lists have different lengths.
    """
    if len(clean_labels) != len(attacked_labels):
        raise ValueError("Clean and attacked labels must have the same length.")
    return {
        "requested": len(clean_labels),
        "pointwise_flip_count": sum(
            clean == "No" and attacked == "Yes"
            for clean, attacked in zip(clean_labels, attacked_labels, strict=True)
        ),
        "clean_no_count": clean_labels.count("No"),
        "clean_yes_count": clean_labels.count("Yes"),
        "clean_invalid_count": clean_labels.count("INVALID"),
        "attacked_yes_count": attacked_labels.count("Yes"),
        "attacked_no_count": attacked_labels.count("No"),
        "attacked_invalid_count": attacked_labels.count("INVALID"),
        "attacked_valid_rankings": sum(
            label in {"Yes", "No"} for label in attacked_labels
        ),
    }


def _evaluate_instance(
    instance: tuple[str, Passage],
    model_name: str,
    base_url: str,
    provider: str,
    aws_region: str | None,
    prompt_template: str,
) -> dict:
    query, passage = instance
    prompt = prompt_template.format(query=query, passage=passage.text)
    client = get_ranking_client(
        model_name, provider=provider, base_url=base_url, region=aws_region
    )
    for attempt in range(3):
        try:
            response, usage = client.generate(prompt, max_tokens=3, return_usage=True)
            response = (response or "").strip()
            return {
                "label": parse_binary_label(response),
                "prompt": prompt,
                "response": response,
                "usage": usage,
            }
        except Exception as error:
            if attempt == 2:
                raise RuntimeError(f"Failed after 3 attempts: {error}") from error
            print(f"Request failed (attempt {attempt + 1}/3): {error}")
            time.sleep(2 * (attempt + 1))
    raise AssertionError("Unreachable retry state.")


def _evaluate(
    instances: list[tuple[str, Passage]],
    *,
    model_name: str,
    base_url: str,
    provider: str,
    aws_region: str | None,
    prompt_template: str,
    checkpoint: EvaluationCheckpoint,
    phase: str,
    n_jobs: int,
    checkpoint_batch_size: int,
) -> list[dict]:
    def worker(instance: tuple[str, Passage]) -> dict:
        return _evaluate_instance(
            instance,
            model_name,
            base_url,
            provider,
            aws_region,
            prompt_template,
        )

    return run_checkpointed(
        instances,
        worker,
        checkpoint,
        phase,
        n_jobs,
        checkpoint_batch_size,
        f"Querying {provider} ({phase})",
    )


def _write_details(
    path: str,
    instances: list[tuple[str, Passage]],
    clean: list[dict],
    attacked: list[dict],
    prompt_mode: str,
) -> None:
    details = []
    for phase, records in (("original", clean), ("attacked", attacked)):
        for (query, passage), record in zip(instances, records, strict=True):
            details.append(
                {
                    "phase": phase,
                    "prompt_mode": prompt_mode,
                    "query": query,
                    "doc_id": passage.doc_id,
                    "relevance": passage.relevance,
                    "prompt": record["prompt"],
                    "response": record["response"],
                    "label": record["label"],
                    "usage": record.get("usage", {}),
                }
            )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    """Run a checkpointed binary pointwise attack evaluation.

    Returns
    -------
    int
        Zero after writing the result summary and optional detailed records.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, default="openai")
    parser.add_argument("--base_url", default="https://api.openai.com/v1")
    parser.add_argument("--aws_region")
    parser.add_argument("--dataset_name", default="msmarco-passage/trec-dl-2019")
    parser.add_argument("--num_passages", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg_rel", type=int, default=0)
    parser.add_argument("--tokenizer_model")
    parser.add_argument("--max_doc_tokens", type=int, default=8000)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--attack_type", choices=("so", "sd", "qi"), default="so")
    parser.add_argument("--attack_position", choices=("back",), default="back")
    parser.add_argument(
        "--prompt_mode",
        choices=("standard", "defense", "defense_qi"),
        default="standard",
    )
    parser.add_argument("--result_json_path", required=True)
    parser.add_argument("--detailed_results")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--checkpoint_batch_size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.num_passages < 1:
        parser.error("--num_passages must be at least 1")
    if args.checkpoint_batch_size < 1:
        parser.error("--checkpoint_batch_size must be at least 1")

    prompt_template = {
        "standard": pointwise_ranking_prompt,
        "defense": pointwise_ranking_defense,
        "defense_qi": pointwise_ranking_defense_qi,
    }[args.prompt_mode]
    instances = prepare_passages(
        args.dataset_name,
        args.num_passages,
        args.seed,
        args.tokenizer_model,
        args.neg_rel,
        args.max_doc_tokens,
    )
    attack_text = jailbreak_prompt[args.attack_type]
    attacked_instances = [
        (
            query,
            Passage(
                passage.doc_id,
                passage.text + attack_text.format(query=query),
                passage.relevance,
            ),
        )
        for query, passage in instances
    ]
    checkpoint_path = args.checkpoint_path or f"{args.result_json_path}.checkpoint.json"
    checkpoint = EvaluationCheckpoint(
        checkpoint_path,
        {
            "paradigm": "pointwise",
            "model_name": args.model_name,
            "provider": args.provider,
            "aws_region": args.aws_region,
            "dataset_name": args.dataset_name,
            "num_passages": args.num_passages,
            "seed": args.seed,
            "neg_rel": args.neg_rel,
            "max_doc_tokens": args.max_doc_tokens,
            "attack_type": args.attack_type,
            "attack_position": args.attack_position,
            "prompt_mode": args.prompt_mode,
        },
        resume=args.resume,
    )
    clean = _evaluate(
        instances,
        model_name=args.model_name,
        base_url=args.base_url,
        provider=args.provider,
        aws_region=args.aws_region,
        prompt_template=prompt_template,
        checkpoint=checkpoint,
        phase="clean",
        n_jobs=args.n_jobs,
        checkpoint_batch_size=args.checkpoint_batch_size,
    )
    attacked = _evaluate(
        attacked_instances,
        model_name=args.model_name,
        base_url=args.base_url,
        provider=args.provider,
        aws_region=args.aws_region,
        prompt_template=prompt_template,
        checkpoint=checkpoint,
        phase="attacked",
        n_jobs=args.n_jobs,
        checkpoint_batch_size=args.checkpoint_batch_size,
    )
    counts = count_no_to_yes(
        [record["label"] for record in clean], [record["label"] for record in attacked]
    )
    summary = {
        "model_name": args.model_name,
        "provider": args.provider,
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "neg_rel": args.neg_rel,
        "ranking_scheme": "pointwise",
        "attack_type": args.attack_type,
        "attack_position": args.attack_position,
        "prompt_mode": args.prompt_mode,
        "original_total_rankings": counts["requested"],
        "total_queries": counts["requested"],
        **counts,
        "pointwise_flip_percentage": (
            100 * counts["pointwise_flip_count"] / counts["requested"]
        ),
        "token_usage": aggregate_token_usage(clean, attacked),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    output = Path(args.result_json_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    checkpoint.mark_complete()
    if args.detailed_results:
        _write_details(
            args.detailed_results, instances, clean, attacked, args.prompt_mode
        )
    print(f"Total passages: {counts['requested']}")
    print(f"No-to-Yes flips: {counts['pointwise_flip_count']}")
    print(f"ASR: {summary['pointwise_flip_percentage']:.2f}%")
    print(f"Results saved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
