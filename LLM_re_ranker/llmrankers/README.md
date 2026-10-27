# Ranking architecture

The ranker package separates the ranking strategy from the model provider.

## Strategies

- `pointwise.py`: score each passage independently.
- `pairwise.py`: compare passages in pairs.
- `setwise_attack.py`: rank small candidate sets with heapsort/bubblesort.
- `listwise.py`: rank sliding windows of candidates.

## Provider adapters

- local Hugging Face models are selected by model path and device;
- OpenAI-compatible models use the API-key path;
- Bedrock models use the native Converse client;
- Jev/AutoJev models use the `v1/systemone` HTTP endpoint.

`factory.py` is the provider/model selection boundary used by `run_attack.py`.
Adding a model should normally require configuration or a provider adapter, not
a new copy of every ranking strategy.

The Jev listwise adapter uses choice probabilities over each sliding window.
This is the closest listwise operation supported by the current AutoJev API;
the API does not yet return a complete permutation directly.
