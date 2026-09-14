import json
import sys
import unittest
from pathlib import Path


RERANKER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RERANKER_ROOT))

from llmrankers.bedrock_pointwise import (
    BedrockPointwiseLlmRanker,
    MissingLogprobsError,
)
from llmrankers.rankers import SearchResult


class FakeBedrockClient:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        return {"body": json.dumps(next(self.payloads)).encode("utf-8")}


class BedrockPointwiseRankerTests(unittest.TestCase):
    def test_reranks_using_normalized_yes_no_logprobs(self):
        client = FakeBedrockClient([
            {"choices": [{"logprobs": {"content": [{"token": " Yes", "logprob": -0.1, "top_logprobs": [{"token": " No", "logprob": -2.1}]}]}}], "usage": {"prompt_tokens": 10, "completion_tokens": 1}},
            {"choices": [{"logprobs": {"content": [{"token": " No", "logprob": -0.2, "top_logprobs": [{"token": " Yes", "logprob": -1.2}]}]}}], "usage": {"prompt_tokens": 11, "completion_tokens": 1}},
        ])
        ranker = BedrockPointwiseLlmRanker("qwen.test", client=client)
        results = ranker.rerank("query", [SearchResult("a", 0, "first"), SearchResult("b", 0, "second")])

        self.assertEqual(["a", "b"], [result.docid for result in results])
        self.assertEqual(2, ranker.total_compare)
        self.assertEqual(21, ranker.total_prompt_tokens)
        self.assertEqual(2, ranker.total_completion_tokens)
        self.assertEqual({"Yes": -1.2, "No": -0.2}, ranker.last_label_logprobs)
        request = json.loads(client.calls[0]["body"])
        self.assertTrue(request["logprobs"])
        self.assertEqual(20, request["top_logprobs"])
        self.assertEqual(1, request["max_tokens"])

    def test_rejects_missing_label_probability(self):
        client = FakeBedrockClient([{"choices": [{"logprobs": {"content": [{"token": " Yes", "logprob": -0.1, "top_logprobs": []}]}}]}])
        ranker = BedrockPointwiseLlmRanker("qwen.test", client=client)

        with self.assertRaises(MissingLogprobsError):
            ranker.rerank("query", [SearchResult("a", 0, "first")])


if __name__ == "__main__":
    unittest.main()
