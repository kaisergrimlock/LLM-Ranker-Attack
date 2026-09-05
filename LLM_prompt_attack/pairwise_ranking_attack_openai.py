import argparse
import itertools
import json
import os
import time
from datetime import datetime

import runtime_environment  # noqa: F401 - select repository caches before ir_datasets
import ir_datasets
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from dataclasses import dataclass
from prompts import jailbreak_prompt, pairwise_ranking_defense, pairwise_ranking_prompt
from dataset_config import get_dataset_config, get_pos_neg_levels
from collections import defaultdict
from joblib import Parallel, delayed
from llm_client import SUPPORTED_PROVIDERS, get_ranking_client

@dataclass
class Document:
    doc_id: str
    text: str
    relevance: int


# Maximum tokens per document to avoid context length issues
MAX_DOC_TOKENS = 8000

# Global tokenizer cache
_tokenizer_cache = {}


def get_tokenizer(model_name: str):
    """Get tokenizer for the specified model, with caching."""
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _tokenizer_cache[model_name] = tokenizer
        return tokenizer
    except Exception as e:
        _tokenizer_cache[model_name] = None
        print(f"Warning: Failed to load tokenizer for {model_name}: {e}")
        print("Falling back to character-based truncation.")
        return None


def truncate_text(text: str, model_name: str = None, max_tokens: int = MAX_DOC_TOKENS) -> str:
    """
    Truncate text to max_tokens using model's tokenizer.
    
    Args:
        text: The text to truncate
        model_name: HuggingFace model name for tokenizer (e.g., 'Qwen/Qwen3-1.7B')
        max_tokens: Maximum number of tokens to keep
        
    Returns:
        Truncated text
    """
    if model_name is None:
        # Fallback to character-based truncation (rough estimate: 4 chars per token)
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:
            truncated = truncated[:last_space]
        return truncated + "..."
    
    tokenizer = get_tokenizer(model_name)
    if tokenizer is None:
        # Fallback to character-based truncation
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
    
    # Tokenize and truncate
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    
    # Decode truncated tokens back to text
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_text + "..."


def prepare_pairs(dataset_name: str, pos_rel: int = None, neg_rel: int = None, num_pairs: int = 1024, seed: int = 42, model_name: str = None):
    """
    Prepare document pairs for pairwise ranking evaluation.
    
    Supports datasets with varying relevance level schemes. If pos_rel/neg_rel are not 
    specified, they will be auto-detected based on the dataset configuration.
    
    For datasets like SciFact that only annotate positive docs (no explicit negatives),
    negative documents are randomly sampled from the corpus.
    
    Args:
        dataset_name: ir_datasets dataset name (e.g., 'msmarco-passage/trec-dl-2019', 'beir/trec-covid')
        pos_rel: Positive relevance level (optional, auto-detected if None)
        neg_rel: Negative relevance level (optional, auto-detected if None)
        num_pairs: Total number of pairs to generate
        seed: Random seed for reproducibility
        model_name: HuggingFace model name for tokenizer-based truncation (optional)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Auto-detect pos/neg levels if not specified
    pos_rel, neg_rel = get_pos_neg_levels(dataset_name, pos_rel, neg_rel)
    config = get_dataset_config(dataset_name)
    print(f"Dataset: {dataset_name}, Relevance levels: {config['rel_levels']}")
    print(f"Using pos_rel={pos_rel}, neg_rel={neg_rel}")
    
    # Warn if dataset has limited relevance levels
    if "warning" in config:
        print(f"⚠️  Warning: {config['warning']}")
    
    dataset = ir_datasets.load(dataset_name)
    docstore = dataset.docs_store()

    queries = {q.query_id: q.text for q in dataset.queries}
    qrels_df = pd.DataFrame(dataset.qrels_iter())
    
    # Check if we need negative sampling (e.g., SciFact)
    needs_negative_sampling = config.get("needs_negative_sampling", False)
    
    if needs_negative_sampling:
        print(f"📝 Using negative sampling strategy for {dataset_name}")
        return _prepare_pairs_with_negative_sampling(
            dataset, docstore, queries, qrels_df, pos_rel, num_pairs, seed, model_name
        )
    
    # Standard approach: use existing relevance labels
    actual_levels = sorted(qrels_df['relevance'].unique().tolist())
    if pos_rel not in actual_levels:
        raise ValueError(f"pos_rel={pos_rel} not found in dataset. Available levels: {actual_levels}")
    if neg_rel not in actual_levels:
        raise ValueError(f"neg_rel={neg_rel} not found in dataset. Available levels: {actual_levels}")

    pairs = []
    skipped_queries = 0
    for query_id, group in qrels_df.groupby('query_id'):
        rel_docs = group.groupby('relevance')
        if pos_rel not in rel_docs.groups or neg_rel not in rel_docs.groups:
            skipped_queries += 1
            continue
        pos_docs = rel_docs.get_group(pos_rel)['doc_id'].tolist()
        neg_docs = rel_docs.get_group(neg_rel)['doc_id'].tolist()

        possible_pairs = list(itertools.product(pos_docs, neg_docs))
        sampled = random.sample(possible_pairs, min(len(possible_pairs), 100))
        for pos_doc_id, neg_doc_id in sampled:
            try:
                pos_doc_data = docstore.get(pos_doc_id)
                neg_doc_data = docstore.get(neg_doc_id)
                if pos_doc_data is None or neg_doc_data is None:
                    continue
                pos_doc = Document(pos_doc_id, truncate_text(pos_doc_data.text, model_name), pos_rel)
                neg_doc = Document(neg_doc_id, truncate_text(neg_doc_data.text, model_name), neg_rel)
                pairs.append((queries[query_id], pos_doc, neg_doc))
                pairs.append((queries[query_id], neg_doc, pos_doc))
            except (KeyError, Exception):
                continue  # Skip if doc retrieval fails

    if not pairs:
        raise ValueError(f"No document pairs found with relevance levels pos={pos_rel}, neg={neg_rel}. "
                         f"Available levels in dataset: {actual_levels}. "
                         f"This may happen when queries don't have docs at both levels.")
    
    if skipped_queries > 0:
        total_queries = qrels_df['query_id'].nunique()
        print(f"Note: {skipped_queries}/{total_queries} queries skipped (missing docs at pos={pos_rel} or neg={neg_rel})")

    random.shuffle(pairs)
    return pairs[:num_pairs]


def _prepare_pairs_with_negative_sampling(dataset, docstore, queries, qrels_df, pos_rel, num_pairs, seed, model_name: str = None):
    """
    Prepare pairs by sampling negatives from corpus.
    
    Used for datasets like SciFact where only positive (relevant) docs are annotated.
    Negative docs are randomly sampled from the corpus, excluding known positives.
    """
    random.seed(seed)
    
    # Get all positive doc IDs per query
    query_pos_docs = defaultdict(list)
    for _, row in qrels_df.iterrows():
        if row['relevance'] >= pos_rel:
            query_pos_docs[row['query_id']].append(row['doc_id'])
    
    # Get all doc IDs from corpus by iterating through docs
    print("Loading corpus document IDs...")
    all_doc_ids = [doc.doc_id for doc in dataset.docs_iter()]
    print(f"Corpus size: {len(all_doc_ids)} documents")
    
    pairs = []
    for query_id, pos_doc_ids in query_pos_docs.items():
        if query_id not in queries:
            continue
        query_text = queries[query_id]
        
        # Sample negatives: docs not in this query's positive set
        positive_set = set(pos_doc_ids)
        candidate_neg_ids = [did for did in all_doc_ids if did not in positive_set]
        
        # Sample negative docs (same number as positives, max 10 per query)
        num_neg_to_sample = min(len(pos_doc_ids), len(candidate_neg_ids), 10)
        if num_neg_to_sample == 0:
            continue
            
        neg_doc_ids = random.sample(candidate_neg_ids, num_neg_to_sample)
        
        # Create pairs
        for pos_doc_id in pos_doc_ids[:10]:  # Limit positives per query
            for neg_doc_id in neg_doc_ids:
                try:
                    pos_doc_data = docstore.get(pos_doc_id)
                    neg_doc_data = docstore.get(neg_doc_id)
                    if pos_doc_data is None or neg_doc_data is None:
                        continue
                    pos_doc = Document(pos_doc_id, truncate_text(pos_doc_data.text, model_name), pos_rel)
                    neg_doc = Document(neg_doc_id, truncate_text(neg_doc_data.text, model_name), 0)  # 0 = sampled negative
                    pairs.append((query_text, pos_doc, neg_doc))
                    pairs.append((query_text, neg_doc, pos_doc))  # Both orders
                except (KeyError, Exception):
                    # Skip if document retrieval fails
                    continue
    
    if not pairs:
        raise ValueError("No pairs could be generated with negative sampling.")
    
    print(f"Generated {len(pairs)} pairs using negative sampling")
    random.shuffle(pairs)
    return pairs[:num_pairs]


def _process_single_query_pairwise(
    query,
    doc1,
    doc2,
    model_name,
    base_url,
    provider="openai",
    aws_region=None,
    prompt_template=pairwise_ranking_prompt,
):
    """Worker function for parallel processing of a single query."""
    # Create client in worker process
    client = get_ranking_client(
        model_name, provider=provider, base_url=base_url, region=aws_region
    )
    
    prompt = prompt_template.format(query=query, doc1=doc1.text, doc2=doc2.text)
    
    # Retry logic for vLLM robustness
    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            content = client.generate(prompt, max_tokens=3)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                raise RuntimeError(f"Failed after {max_retries} attempts: {e}")

    content = (content or "").strip()
    if not content:
        return {"label": "INVALID", "prompt": prompt, "response": ""}

    # Expect output 'Passage A' or 'Passage B' or just 'A'/'B'
    if content.upper().startswith("PASSAGE A") or content.upper().startswith("A"):
        label = "A"
    else:
        label = "B"
    
    return {
        "label": label,
        "prompt": prompt,
        "response": content
    }


def get_choices_openai(
    pairs, model_name: str, base_url: str, n_jobs=-1,
    return_detailed: bool = False, provider: str = "openai", aws_region: str = None,
    prompt_template: str = pairwise_ranking_prompt,
):
    """Get choices using parallel processing with joblib.
    
    Args:
        return_detailed: If True, return (choices, detailed_results). If False, only return choices.
    """
    
    # Use joblib to parallelize the API calls
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_process_single_query_pairwise)(
            query, doc1, doc2, model_name, base_url, provider, aws_region,
            prompt_template,
        )
        for query, doc1, doc2 in tqdm(pairs, desc=f"Querying {provider}")
    )
    
    # Extract labels
    choices = [r["label"] for r in results]
    
    if return_detailed:
        return choices, results
    else:
        return choices


def validate_rankings(rankings):
    """
    Validate that all rankings are either 'A' or 'B'
    Returns a tuple of (valid_indices, valid_rankings)
    """
    valid_indices = []
    valid_rankings = []
    
    for i, label in enumerate(rankings):
        if label in ("A", "B"):
            valid_indices.append(i)
            valid_rankings.append(label)
            
    return valid_indices, valid_rankings


def apply_attack(results, pairs, attack_prompt: str, attack_position: str = "back"):
    """Inject attack text into the passage rejected by the clean ranking.

    Parameters
    ----------
    results : list of str
        Clean pairwise choices, represented as ``A`` or ``B``.
    pairs : list of tuple
        Query and two-document instances corresponding to ``results``.
    attack_prompt : str
        Static attack text or a template containing ``{query}``.
    attack_position : {"front", "back"}, default="back"
        Position of the rendered text relative to the target passage.

    Returns
    -------
    list of tuple
        Pairwise instances with one attacked passage per instance.

    Raises
    ------
    ValueError
        If the result and pair counts differ.
    """
    if len(results) != len(pairs):
        raise ValueError(
            f"Length mismatch: results ({len(results)}) and pairs ({len(pairs)})"
        )
    attacked_pairs = []
    for i, result in enumerate(results):
        query, doc1, doc2 = pairs[i]
        rendered_attack_prompt = attack_prompt.format(query=query)
        if result == "A":
            if attack_position == "front":
                attacked_doc = Document(
                    doc2.doc_id, rendered_attack_prompt + doc2.text, doc2.relevance
                )
            else:  # back
                attacked_doc = Document(
                    doc2.doc_id, doc2.text + rendered_attack_prompt, doc2.relevance
                )
            attacked_pairs.append((query, doc1, attacked_doc))
        else:
            if attack_position == "front":
                attacked_doc = Document(
                    doc1.doc_id, rendered_attack_prompt + doc1.text, doc1.relevance
                )
            else:  # back
                attacked_doc = Document(
                    doc1.doc_id, doc1.text + rendered_attack_prompt, doc1.relevance
                )
            attacked_pairs.append((query, attacked_doc, doc2))
    return attacked_pairs


def count_flipped_queries(original_results, attacked_results):
    assert len(original_results) == len(attacked_results), "lengths do not match!"
    flipped_count = sum(1 for o, a in zip(original_results, attacked_results) if o != a)
    total = len(original_results)
    print(f"Total Queries: {total}")
    print(f"Flipped Queries: {flipped_count}")
    flipped_rate = flipped_count / total * 100 if total else 0.0
    print(f"Percentage Flipped: {flipped_rate:.2f}%")
    return flipped_count, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Provider model name or Bedrock model ID",
    )
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, default="openai")
    parser.add_argument(
        "--aws_region",
        type=str,
        default=None,
        help="Bedrock region; defaults to BEDROCK_REGION/AWS_REGION or ap-southeast-2",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="msmarco-passage/trec-dl-2019"
    )
    parser.add_argument("--num_pairs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pos_rel",
        type=int,
        default=None,
        help="Positive relevance level. If not specified, auto-detected from dataset config.",
    )
    parser.add_argument(
        "--neg_rel",
        type=int,
        default=None,
        help="Negative relevance level. If not specified, auto-detected from dataset config.",
    )
    parser.add_argument(
        "--result_json_path", type=str, default="outputs/results_pairwise_openai.jsonl"
    )
    parser.add_argument("--attack_type", choices=["so", "sd", "qi"], default="so")
    parser.add_argument(
        "--attack_position",
        choices=["front", "back"],
        default="back",
        help="Position to place the attack prompt: 'front' or 'back' of the passage",
    )
    parser.add_argument(
        "--prompt_mode",
        choices=["standard", "defense"],
        default="standard",
        help="Ranking prompt used for both the clean projection and attacked comparison.",
    )
    parser.add_argument("--ignore_existing", action="store_true")
    parser.add_argument(
        "--n_jobs", type=int, default=4, help="Number of concurrent model requests"
    )
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument(
        "--tokenizer_model",
        type=str,
        default=None,
        help="HuggingFace model name for tokenizer-based truncation (e.g., 'Qwen/Qwen3-1.7B'). If not set, uses character-based truncation.",
    )
    parser.add_argument(
        "--detailed_results",
        type=str,
        default=None,
        help="Path to save detailed results (query, prompt, response, label) in JSON format",
    )
    args = parser.parse_args()
    if args.attack_type == "qi" and args.attack_position != "back":
        parser.error("--attack_type qi appends the query; use --attack_position back")
    ranking_prompt = (
        pairwise_ranking_defense
        if args.prompt_mode == "defense"
        else pairwise_ranking_prompt
    )

    # Prepare dataset
    pairs = prepare_pairs(
        args.dataset_name, args.pos_rel, args.neg_rel, args.num_pairs, args.seed, args.tokenizer_model
    )

    # Original evaluation
    print(f"Running original evaluation with {args.provider}...")
    original_results, original_detailed = get_choices_openai(
        pairs, args.model_name, args.base_url, args.n_jobs,
        return_detailed=True, provider=args.provider, aws_region=args.aws_region,
        prompt_template=ranking_prompt,
    )
    
    # Validate rankings before proceeding
    valid_indices, valid_rankings = validate_rankings(original_results)
    if len(valid_indices) < len(original_results):
        print(f"Warning: {len(original_results) - len(valid_indices)} rankings were invalid and will be skipped.")
        # Filter pairs to only include those with valid rankings
        valid_pairs = [pairs[i] for i in valid_indices]
    else:
        valid_pairs = pairs
        valid_rankings = original_results
    
    print(f"Proceeding with {len(valid_rankings)} valid rankings.")

    # Attack evaluation
    attacked_pairs = apply_attack(valid_rankings, valid_pairs, jailbreak_prompt[args.attack_type], args.attack_position)
    print(f"Running attacked evaluation with {args.provider}...")
    attacked_results, attacked_detailed = get_choices_openai(
        attacked_pairs, args.model_name, args.base_url, args.n_jobs,
        return_detailed=True, provider=args.provider, aws_region=args.aws_region,
        prompt_template=ranking_prompt,
    )
    
    # Validate attacked results too
    valid_attack_indices, valid_attack_rankings = validate_rankings(attacked_results)
    if len(valid_attack_indices) < len(attacked_results):
        print(f"Warning: {len(attacked_results) - len(valid_attack_indices)} attacked rankings were invalid and will be skipped.")
        # Filter to only include valid attacked rankings
        final_original = [valid_rankings[i] for i in valid_attack_indices]
        final_attacked = valid_attack_rankings
    else:
        final_original = valid_rankings
        final_attacked = attacked_results

    # Count flips
    flipped_count, total = count_flipped_queries(final_original, final_attacked)

    # Save results
    os.makedirs(os.path.dirname(args.result_json_path), exist_ok=True)
    results = {
        "model_name": args.model_name,
        "provider": args.provider,
        "dataset_name": args.dataset_name,
        "ranking_scheme": "pairwise",
        "attack_type": args.attack_type,
        "attack_position": args.attack_position,
        "prompt_mode": args.prompt_mode,
        "flipped_count": flipped_count,
        "total_queries": total,
        "original_valid_rankings": len(valid_rankings),
        "original_total_rankings": len(original_results),
        "attacked_valid_rankings": len(final_attacked),
        "flipped_percentage": flipped_count / total * 100 if total else 0.0,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(args.result_json_path, "a") as f:
        f.write(json.dumps(results, ensure_ascii=False) + "\n")
    
    print(f"Results saved to: {args.result_json_path}")
    
    # Save detailed results if requested
    if args.detailed_results:
        os.makedirs(os.path.dirname(args.detailed_results) if os.path.dirname(args.detailed_results) else ".", exist_ok=True)
        detailed_data = []
        
        # Add original results
        for i, (query, doc1, doc2) in enumerate(pairs):
            if i < len(original_detailed):
                detailed_data.append({
                    "phase": "original",
                    "prompt_mode": args.prompt_mode,
                    "query": query,
                    "doc1_id": doc1.doc_id,
                    "doc2_id": doc2.doc_id,
                    "response": original_detailed[i]["response"],
                    "label": original_detailed[i]["label"]
                })
        
        # Add attacked results
        for i, (query, doc1, doc2) in enumerate(attacked_pairs):
            if i < len(attacked_detailed):
                detailed_data.append({
                    "phase": "attacked",
                    "prompt_mode": args.prompt_mode,
                    "query": query,
                    "doc1_id": doc1.doc_id,
                    "doc2_id": doc2.doc_id,
                    "prompt": attacked_detailed[i]["prompt"],
                    "response": attacked_detailed[i]["response"],
                    "label": attacked_detailed[i]["label"]
                })
        
        with open(args.detailed_results, "w", encoding="utf-8") as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)
        
        print(f"Detailed results saved to: {args.detailed_results}")

if __name__ == "__main__":
    main()
