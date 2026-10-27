"""Construct ranking strategies from a provider/model configuration.

Ranking algorithms live in their strategy modules; this module is the single
provider-selection boundary used by the command-line runner.  Adding a model
does not require another pointwise/listwise/pairwise/setwise implementation.
"""


def create_ranker(args, provider):
    run = args.run

    if args.pointwise:
        if provider == 'autojev':
            from .pointwise import AutoJevPointwiseRanker
            return AutoJevPointwiseRanker(
                endpoint=run.autojev_endpoint,
                method=args.pointwise.method,
                model=run.model_name_or_path or 'autojev',
            )
        if provider == 'amazon-bedrock':
            from .bedrock_pointwise import BedrockPointwiseLlmRanker
            return BedrockPointwiseLlmRanker(
                model_name_or_path=run.model_name_or_path,
                region=run.aws_region,
                max_tokens=run.bedrock_max_tokens,
                top_logprobs=run.bedrock_top_logprobs,
            )

        from .pointwise import PointwiseLlmRanker, MonoT5LlmRanker
        ranker_type = MonoT5LlmRanker if 'monot5' in run.model_name_or_path else PointwiseLlmRanker
        return ranker_type(
            model_name_or_path=run.model_name_or_path,
            tokenizer_name_or_path=run.tokenizer_name_or_path,
            device=run.device,
            cache_dir=run.cache_dir,
            method=args.pointwise.method,
            batch_size=args.pointwise.batch_size,
        )

    if args.setwise:
        if provider == 'amazon-bedrock':
            from .bedrock_setwise import BedrockSetwiseLlmRanker
            return BedrockSetwiseLlmRanker(
                model_name_or_path=run.model_name_or_path,
                region=run.aws_region,
                num_child=args.setwise.num_child,
                method=args.setwise.method,
                k=args.setwise.k,
                max_tokens=run.bedrock_max_tokens,
                invalid_output_policy=run.invalid_output_policy,
            )
        if provider == 'openai':
            from .setwise_attack import OpenAiSetwiseLlmRanker
            return OpenAiSetwiseLlmRanker(
                model_name_or_path=run.model_name_or_path,
                api_key=run.openai_key,
                num_child=args.setwise.num_child,
                method=args.setwise.method,
                k=args.setwise.k,
            )

        from .setwise_attack import SetwiseLlmRanker
        return SetwiseLlmRanker(
            model_name_or_path=run.model_name_or_path,
            tokenizer_name_or_path=run.tokenizer_name_or_path,
            device=run.device,
            cache_dir=run.cache_dir,
            num_child=args.setwise.num_child,
            scoring=run.scoring,
            method=args.setwise.method,
            num_permutation=args.setwise.num_permutation,
            k=args.setwise.k,
        )

    if args.pairwise:
        from .pairwise import DuoT5LlmRanker, OpenAiPairwiseLlmRanker, PairwiseLlmRanker
        if args.pairwise.method != 'allpair':
            args.pairwise.batch_size = 2
        if run.openai_key:
            return OpenAiPairwiseLlmRanker(
                model_name_or_path=run.model_name_or_path,
                api_key=run.openai_key,
                method=args.pairwise.method,
                k=args.pairwise.k,
            )
        if 'duot5' in run.model_name_or_path:
            return DuoT5LlmRanker(
                model_name_or_path=run.model_name_or_path,
                tokenizer_name_or_path=run.tokenizer_name_or_path,
                device=run.device,
                cache_dir=run.cache_dir,
                method=args.pairwise.method,
                batch_size=args.pairwise.batch_size,
                k=args.pairwise.k,
            )
        return PairwiseLlmRanker(
            model_name_or_path=run.model_name_or_path,
            tokenizer_name_or_path=run.tokenizer_name_or_path,
            device=run.device,
            cache_dir=run.cache_dir,
            method=args.pairwise.method,
            batch_size=args.pairwise.batch_size,
            k=args.pairwise.k,
        )

    if args.listwise:
        if provider == 'autojev':
            from .autojev_listwise import AutoJevListwiseRanker
            return AutoJevListwiseRanker(
                endpoint=run.autojev_endpoint,
                model=run.model_name_or_path or 'autojev',
                window_size=args.listwise.window_size,
                step_size=args.listwise.step_size,
                num_repeat=args.listwise.num_repeat,
            )
        from .listwise import OpenAiListwiseLlmRanker, ListwiseLlmRanker
        if run.openai_key:
            return OpenAiListwiseLlmRanker(
                model_name_or_path=run.model_name_or_path,
                api_key=run.openai_key,
                window_size=args.listwise.window_size,
                step_size=args.listwise.step_size,
                num_repeat=args.listwise.num_repeat,
            )
        return ListwiseLlmRanker(
            model_name_or_path=run.model_name_or_path,
            tokenizer_name_or_path=run.tokenizer_name_or_path,
            device=run.device,
            cache_dir=run.cache_dir,
            window_size=args.listwise.window_size,
            step_size=args.listwise.step_size,
            scoring=run.scoring,
            num_repeat=args.listwise.num_repeat,
        )

    raise ValueError('Must specify either --pointwise, --pairwise, --setwise or --listwise.')
