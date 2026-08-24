import logging
import ir_datasets
from pyserini.search.lucene import LuceneSearcher
from pyserini.search._base import get_topics
from llmrankers.rankers import SearchResult
from llmrankers.bedrock_setwise import BedrockSetwiseLlmRanker
from tqdm import tqdm
import argparse
import sys
import json
import time
import random
from pathlib import Path
from collections import defaultdict

random.seed(929)
logger = logging.getLogger(__name__)


def parse_args(parser, commands):
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args


def write_run_file(path, results, tag):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for qid, _, ranking in results:
            rank = 1
            for doc in ranking:
                docid = doc.docid
                score = doc.score
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\t{tag}\n")
                rank += 1


def main(args):

    provider = args.run.provider
    if provider == 'local' and args.run.openai_key:
        # Preserve the original CLI behavior for existing generated job files.
        provider = 'openai'

    if provider != 'local' and not args.setwise:
        raise ValueError(f'Provider {provider!r} is currently supported only with setwise ranking.')

    if args.pointwise:
        from llmrankers.pointwise import PointwiseLlmRanker, MonoT5LlmRanker

        if 'monot5' in args.run.model_name_or_path:
            ranker = MonoT5LlmRanker(model_name_or_path=args.run.model_name_or_path,
                                     tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                     device=args.run.device,
                                     cache_dir=args.run.cache_dir,
                                     method=args.pointwise.method,
                                     batch_size=args.pointwise.batch_size)
        else:
            ranker = PointwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                        tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                        device=args.run.device,
                                        cache_dir=args.run.cache_dir,
                                        method=args.pointwise.method,
                                        batch_size=args.pointwise.batch_size)

    elif args.setwise:
        if provider == 'amazon-bedrock':
            ranker = BedrockSetwiseLlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                region=args.run.aws_region,
                num_child=args.setwise.num_child,
                method=args.setwise.method,
                k=args.setwise.k,
            )
        elif provider == 'openai':
            from llmrankers.setwise_attack import OpenAiSetwiseLlmRanker

            ranker = OpenAiSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                            api_key=args.run.openai_key,
                                            num_child=args.setwise.num_child,
                                            method=args.setwise.method,
                                            k=args.setwise.k)
        else:
            from llmrankers.setwise_attack import SetwiseLlmRanker

            ranker = SetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                      tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                      device=args.run.device,
                                      cache_dir=args.run.cache_dir,
                                      num_child=args.setwise.num_child,
                                      scoring=args.run.scoring,
                                      method=args.setwise.method,
                                      num_permutation=args.setwise.num_permutation,
                                      k=args.setwise.k)

    elif args.pairwise:
        from llmrankers.pairwise import (
            DuoT5LlmRanker,
            OpenAiPairwiseLlmRanker,
            PairwiseLlmRanker,
        )

        if args.pairwise.method != 'allpair':
            args.pairwise.batch_size = 2
            logger.info(f'Setting batch_size to 2.')

        if args.run.openai_key:
            ranker = OpenAiPairwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                             api_key=args.run.openai_key,
                                             method=args.pairwise.method,
                                             k=args.pairwise.k)

        elif 'duot5' in args.run.model_name_or_path:
            ranker = DuoT5LlmRanker(model_name_or_path=args.run.model_name_or_path,
                                    tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                    device=args.run.device,
                                    cache_dir=args.run.cache_dir,
                                    method=args.pairwise.method,
                                    batch_size=args.pairwise.batch_size,
                                    k=args.pairwise.k)
        else:
            ranker = PairwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                       tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                       device=args.run.device,
                                       cache_dir=args.run.cache_dir,
                                       method=args.pairwise.method,
                                       batch_size=args.pairwise.batch_size,
                                       k=args.pairwise.k)

    elif args.listwise:
        from llmrankers.listwise import OpenAiListwiseLlmRanker, ListwiseLlmRanker

        if args.run.openai_key:
            ranker = OpenAiListwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                             api_key=args.run.openai_key,
                                             window_size=args.listwise.window_size,
                                             step_size=args.listwise.step_size,
                                             num_repeat=args.listwise.num_repeat)
        else:
            ranker = ListwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                       tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                       device=args.run.device,
                                       cache_dir=args.run.cache_dir,
                                       window_size=args.listwise.window_size,
                                       step_size=args.listwise.step_size,
                                       scoring=args.run.scoring,
                                       num_repeat=args.listwise.num_repeat)
    else:
        raise ValueError('Must specify either --pointwise, --setwise, --pairwise or --listwise.')

    query_map = {}
    qrels_map = defaultdict(dict)  # qrels_map[qid][docid] = rel (GT). empty if not available
    if args.run.ir_dataset_name is not None:
        dataset = ir_datasets.load(args.run.ir_dataset_name)

        # queries
        for query in dataset.queries_iter():
            qid = str(query.query_id)
            text = query.text
            query_map[qid] = ranker.truncate(text, args.run.query_length)

        # docstore
        docstore = dataset.docs_store()

        # qrels (GT)
        # Some datasets may not provide qrels_iter; keep empty map in that case.
        try:
            for r in dataset.qrels_iter():
                qrels_map[str(r.query_id)][str(r.doc_id)] = int(r.relevance)
        except Exception:
            pass

    else:
        topics = get_topics(args.run.pyserini_index+'-test')
        for topic_id in list(topics.keys()):
            text = topics[topic_id]['title']
            query_map[str(topic_id)] = ranker.truncate(text, args.run.query_length)
        docstore = LuceneSearcher.from_prebuilt_index(args.run.pyserini_index+'.flat')
        # qrels_map stays empty in this branch (no GT)


    logger.info(f'Loading first stage run from {args.run.run_path}.')
    first_stage_rankings = []
    with open(args.run.run_path, 'r') as f:
        current_qid = None
        current_ranking = []
        for line in tqdm(f):
            qid, _, docid, _, score, _ = line.strip().split()
            qid = str(qid)
            docid = str(docid)

            if qid != current_qid:
                if current_qid is not None:
                    first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.run.hits]))
                current_ranking = []
                current_qid = qid

            if len(current_ranking) >= args.run.hits:
                continue

            if args.run.ir_dataset_name is not None:
                d = docstore.get(docid)
                text = d.text
                if hasattr(d, "title") and d.title:
                    text = f'{d.title} {text}'
            else:
                data = json.loads(docstore.doc(docid).raw())
                text = data['text']
                if 'title' in data:
                    text = f'{data["title"]} {text}'
            text = ranker.truncate(text, args.run.passage_length)
            sr = SearchResult(docid=docid, score=float(score), text=text)
            sr.gt_rel = qrels_map.get(str(qid), {}).get(str(docid), None)  # None = unjudged
            current_ranking.append(sr)
        first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.run.hits]))

    if args.run.max_queries is not None:
        if args.run.max_queries < 1:
            raise ValueError('--max_queries must be at least 1.')
        first_stage_rankings = first_stage_rankings[:args.run.max_queries]

    if not first_stage_rankings:
        raise RuntimeError('The first-stage run did not contain any rankings.')


    reranked_results = []
    total_comparisons = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    tic = time.time()
    for qid, query, ranking in tqdm(first_stage_rankings):
        if args.run.shuffle_ranking is not None:
            if args.run.shuffle_ranking == 'random':
                random.shuffle(ranking)
            elif args.run.shuffle_ranking == 'inverse':
                ranking = ranking[::-1]
            else:
                raise ValueError(f'Invalid shuffle ranking method: {args.run.shuffle_ranking}.')
        reranked_results.append((qid, query, ranker.rerank(query, ranking, attack_prompt=args.run.attack_type, attack_position=args.run.attack_position)))
        total_comparisons += ranker.total_compare
        total_prompt_tokens += ranker.total_prompt_tokens
        total_completion_tokens += ranker.total_completion_tokens
    toc = time.time()

    print(f'Avg comparisons: {total_comparisons/len(reranked_results)}')
    print(f'Avg prompt tokens: {total_prompt_tokens/len(reranked_results)}')
    print(f'Avg completion tokens: {total_completion_tokens/len(reranked_results)}')
    print(f'Avg time per query: {(toc-tic)/len(reranked_results)}')

    write_run_file(args.run.save_path, reranked_results, 'LLMRankers')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(title='sub-commands')

    run_parser = commands.add_parser('run')
    run_parser.add_argument('--run_path', type=str, help='Path to the first stage run file (TREC format) to rerank.')
    run_parser.add_argument('--save_path', type=str, help='Path to save the reranked run file (TREC format).')
    run_parser.add_argument('--model_name_or_path', type=str,
                            help='Path to the pretrained model or model identifier from huggingface.co/models')
    run_parser.add_argument('--tokenizer_name_or_path', type=str, default=None,
                            help='Path to the pretrained tokenizer or tokenizer identifier from huggingface.co/tokenizers')
    run_parser.add_argument('--ir_dataset_name', type=str, default=None)
    run_parser.add_argument('--pyserini_index', type=str, default=None)
    run_parser.add_argument('--hits', type=int, default=100)
    run_parser.add_argument('--query_length', type=int, default=128)
    run_parser.add_argument('--passage_length', type=int, default=128)
    run_parser.add_argument('--device', type=str, default='cuda')
    run_parser.add_argument('--cache_dir', type=str, default=None)
    run_parser.add_argument('--openai_key', type=str, default=None)
    run_parser.add_argument('--provider', type=str, default='local',
                            choices=['local', 'openai', 'amazon-bedrock'])
    run_parser.add_argument('--aws_region', type=str, default=None)
    run_parser.add_argument('--max_queries', type=int, default=None,
                            help='Rerank only the first N queries (useful for smoke tests).')
    run_parser.add_argument('--scoring', type=str, default='generation', choices=['generation', 'likelihood'])
    run_parser.add_argument('--shuffle_ranking', type=str, default=None, choices=['inverse', 'random'])
    run_parser.add_argument("--attack_type", choices=["none", "so", "sd"], default="none",
                        help="Attack type: none (disable), so, or sd.")
    run_parser.add_argument("--attack_position", choices=["front", "back"], default="back",
                        help="Position to place the attack prompt: 'front' or 'back' of the passage")

    pointwise_parser = commands.add_parser('pointwise')
    pointwise_parser.add_argument('--method', type=str, default='yes_no',
                                  choices=['qlm', 'yes_no'])
    pointwise_parser.add_argument('--batch_size', type=int, default=2)

    pairwise_parser = commands.add_parser('pairwise')
    pairwise_parser.add_argument('--method', type=str, default='allpair',
                                 choices=['allpair', 'heapsort', 'bubblesort'])
    pairwise_parser.add_argument('--batch_size', type=int, default=2)
    pairwise_parser.add_argument('--k', type=int, default=10)

    setwise_parser = commands.add_parser('setwise')
    setwise_parser.add_argument('--num_child', type=int, default=3)
    setwise_parser.add_argument('--method', type=str, default='heapsort',
                                choices=['heapsort', 'bubblesort'])
    setwise_parser.add_argument('--k', type=int, default=10)
    setwise_parser.add_argument('--num_permutation', type=int, default=1)

    listwise_parser = commands.add_parser('listwise')
    listwise_parser.add_argument('--window_size', type=int, default=3)
    listwise_parser.add_argument('--step_size', type=int, default=1)
    listwise_parser.add_argument('--num_repeat', type=int, default=1)

    args = parse_args(parser, commands)

    if args.run.ir_dataset_name is not None and args.run.pyserini_index is not None:
        raise ValueError('Must specify either --ir_dataset_name or --pyserini_index, not both.')

    arg_dict = vars(args)
    if arg_dict['run'] is None or sum(arg_dict[arg] is not None for arg in arg_dict) != 2:
        raise ValueError('Need to set --run and can only set one of --pointwise, --pairwise, --setwise, --listwise')
    main(args)
