
import ir_datasets
import pandas as pd
from collections import defaultdict
import random
from ir_datasets_compat import enable_windows_download_compat


enable_windows_download_compat()

# Dataset configuration mapping
# - rel_levels: All relevance levels in the dataset (sorted ascending)
# - pos_threshold: Minimum relevance to be considered "positive/relevant"
# - default_pos: Default positive relevance level for pairwise (highest level = most relevant)
# - default_neg: Default negative relevance level for pairwise (lowest level = least relevant)
# - needs_negative_sampling: True if dataset lacks explicit negative labels (e.g., SciFact)
# 
# Design rationale for default_pos/neg:
#   - default_pos: Use the HIGHEST relevance level (most clearly relevant documents)
#   - default_neg: Use the LOWEST relevance level (least relevant / non-relevant documents)
#   - This maximizes the contrast between positive and negative samples

DATASET_CONFIG = {
    # TREC Deep Learning Track (MSMARCO)
    # Levels: 0=Irrelevant, 1=Related, 2=Highly Relevant, 3=Perfectly Relevant
    "msmarco-passage/trec-dl-2019": {
        "rel_levels": [0, 1, 2, 3],
        "pos_threshold": 2,
        "default_pos": 3,
        "default_neg": 0,
        "needs_negative_sampling": False,
    },
    "msmarco-passage/trec-dl-2020": {
        "rel_levels": [0, 1, 2, 3],
        "pos_threshold": 2,
        "default_pos": 3,
        "default_neg": 0,
        "needs_negative_sampling": False,
    },
    # BEIR Datasets
    # TREC-COVID: -1=Not Judged, 0=Not Relevant, 1=Partially Relevant, 2=Relevant
    "beir/trec-covid": {
        "rel_levels": [-1, 0, 1, 2],
        "pos_threshold": 1,
        "default_pos": 2,
        "default_neg": 0,
        "needs_negative_sampling": False,
    },
    # Touche (Argument Retrieval): -2 to 5 scale
    "beir/webis-touche2020/v2": {
        "rel_levels": [0, 1, 2],
        "pos_threshold": 2,
        "default_pos": 2,
        "default_neg": 0,
        "needs_negative_sampling": False,
    },
    # SciFact: Only relevant docs are annotated (relevance=1)
    # Negative samples must be obtained via random sampling from corpus
    "beir/scifact/test": {
        "rel_levels": [1],  # Only relevance=1 exists in qrels
        "pos_threshold": 1,
        "default_pos": 1,
        "default_neg": 0,  # Virtual level for sampled negatives
        "needs_negative_sampling": True,  # Must sample negatives from corpus
        "warning": "SciFact only annotates relevant docs. Negatives will be sampled from corpus.",
    },
    # DBpedia-Entity: 0=Irrelevant, 1=Relevant, 2=Highly Relevant
    "beir/dbpedia-entity/test": {
        "rel_levels": [0, 1, 2],
        "pos_threshold": 1,
        "default_pos": 2,
        "default_neg": 0,
        "needs_negative_sampling": False,
    },
}


def sample_negative_docs(dataset, qrels_df, query_id: str, num_negatives: int, seed: int = 42) -> list:
    """
    Sample negative documents from corpus for a query.
    
    Samples random documents that are NOT in the qrels for this query.
    This is used for datasets like SciFact that only annotate positive docs.
    
    Args:
        dataset: ir_datasets dataset object
        qrels_df: DataFrame of qrels
        query_id: The query ID to sample negatives for
        num_negatives: Number of negative docs to sample
        seed: Random seed
        
    Returns:
        List of (doc_id, doc_text) tuples
    """
    random.seed(seed)
    
    # Get all doc IDs that are judged for this query (these are positives)
    positive_doc_ids = set(qrels_df[qrels_df['query_id'] == query_id]['doc_id'].tolist())
    
    # Get docstore
    docstore = dataset.docs_store()
    
    # Sample from corpus, excluding positives
    # Note: For large corpora, we sample from doc IDs directly
    all_doc_ids = list(docstore.keys())
    candidate_ids = [did for did in all_doc_ids if did not in positive_doc_ids]
    
    if len(candidate_ids) < num_negatives:
        # If not enough candidates, use all available
        sampled_ids = candidate_ids
    else:
        sampled_ids = random.sample(candidate_ids, num_negatives)
    
    return [(doc_id, docstore.get(doc_id).text) for doc_id in sampled_ids]


def auto_detect_relevance_levels(dataset_name: str) -> dict:
    """
    Auto-detect relevance levels from the dataset's qrels.
    
    Returns:
        dict with rel_levels, pos_threshold, default_pos, default_neg
    """
    try:
        dataset = ir_datasets.load(dataset_name)
        qrels_df = pd.DataFrame(dataset.qrels_iter())
        rel_levels = sorted(qrels_df['relevance'].unique().tolist())
        
        # Heuristic: positive threshold is the median level (rounded up)
        mid_idx = len(rel_levels) // 2
        pos_threshold = rel_levels[mid_idx]
        
        # Default pos/neg: use highest and lowest
        default_pos = rel_levels[-1]
        default_neg = rel_levels[0]
        
        return {
            "rel_levels": rel_levels,
            "pos_threshold": pos_threshold,
            "default_pos": default_pos,
            "default_neg": default_neg,
        }
    except Exception as e:
        raise ValueError(f"Failed to auto-detect relevance levels for {dataset_name}: {e}")


def get_dataset_config(dataset_name: str) -> dict:
    """
    Get configuration for a dataset.
    
    If the dataset is not in the predefined config, it will auto-detect
    the relevance levels from the qrels.
    
    Args:
        dataset_name: The ir_datasets dataset name
        
    Returns:
        dict with keys: rel_levels, pos_threshold, default_pos, default_neg
    """
    if dataset_name in DATASET_CONFIG:
        return DATASET_CONFIG[dataset_name]
    
    print(f"Dataset '{dataset_name}' not in predefined config, auto-detecting...")
    config = auto_detect_relevance_levels(dataset_name)
    print(f"  Detected relevance levels: {config['rel_levels']}")
    print(f"  Using pos_threshold={config['pos_threshold']}, "
          f"default_pos={config['default_pos']}, default_neg={config['default_neg']}")
    return config


def get_pos_neg_levels(dataset_name: str, pos_rel: int = None, neg_rel: int = None) -> tuple:
    """
    Get positive and negative relevance levels for pairwise comparison.
    
    Args:
        dataset_name: The ir_datasets dataset name
        pos_rel: Override positive relevance level (optional)
        neg_rel: Override negative relevance level (optional)
        
    Returns:
        tuple of (pos_rel, neg_rel)
    """
    config = get_dataset_config(dataset_name)
    
    if pos_rel is None:
        pos_rel = config["default_pos"]
    if neg_rel is None:
        neg_rel = config["default_neg"]
    
    return pos_rel, neg_rel


def get_available_levels_for_query(query_rel_docs: dict, qid: str) -> list:
    """
    Get available relevance levels for a specific query.
    
    Args:
        query_rel_docs: Dict mapping query_id -> relevance -> [doc_ids]
        qid: Query ID
        
    Returns:
        List of available relevance levels for this query
    """
    return list(query_rel_docs[qid].keys())


def print_dataset_info(dataset_name: str):
    """Print information about a dataset's relevance levels."""
    config = get_dataset_config(dataset_name)
    print(f"\nDataset: {dataset_name}")
    print(f"  Relevance levels: {config['rel_levels']}")
    print(f"  Positive threshold: {config['pos_threshold']}")
    print(f"  Default pos/neg for pairwise: {config['default_pos']}/{config['default_neg']}")
