import os
import sys
import unittest
from pathlib import Path


RERANKER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RERANKER_ROOT))

from llmrankers.bedrock_setwise import BedrockSetwiseLlmRanker
from llmrankers.rankers import SearchResult
from prompts import JAILBREAK_PROMPTS


class FakeBedrockClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {"message": {"content": [{"text": next(self.outputs)}]}},
            "usage": {"inputTokens": 20, "outputTokens": 2},
        }


class BedrockSetwiseRankerTests(unittest.TestCase):
    def test_converse_shape_usage_and_attack_injection(self):
        client = FakeBedrockClient(["Passage B"])
        ranker = BedrockSetwiseLlmRanker(
            "qwen.test", region="ap-southeast-2", client=client
        )
        relevant = SearchResult("d1", 1.0, "relevant text")
        relevant.gt_rel = 2
        nonrelevant = SearchResult("d2", 0.5, "nonrelevant text")
        nonrelevant.gt_rel = 0

        old_limit = os.environ.get("BEDROCK_MAX_TOKENS")
        os.environ["BEDROCK_MAX_TOKENS"] = "256"
        try:
            label = ranker.compare(
                "query", [relevant, nonrelevant], "so", "back"
            )
        finally:
            if old_limit is None:
                os.environ.pop("BEDROCK_MAX_TOKENS", None)
            else:
                os.environ["BEDROCK_MAX_TOKENS"] = old_limit

        self.assertEqual("B", label)
        self.assertEqual(20, ranker.total_prompt_tokens)
        self.assertEqual(2, ranker.total_completion_tokens)
        call = client.calls[0]
        self.assertEqual("qwen.test", call["modelId"])
        self.assertEqual(256, call["inferenceConfig"]["maxTokens"])
        self.assertEqual(0, call["inferenceConfig"]["temperature"])
        prompt = call["messages"][0]["content"][0]["text"]
        self.assertNotIn(JAILBREAK_PROMPTS["so"], prompt.split("Passage B:")[0])
        self.assertIn("nonrelevant text " + JAILBREAK_PROMPTS["so"], prompt)

    def test_invalid_output_is_bounded(self):
        client = FakeBedrockClient(["unknown", "still unknown", "no label"])
        ranker = BedrockSetwiseLlmRanker("qwen.test", client=client)
        docs = [SearchResult("d1", 1.0, "one"), SearchResult("d2", 0.5, "two")]

        with self.assertRaises(RuntimeError):
            ranker.compare("query", docs)

        self.assertEqual(3, len(client.calls))

    def test_truncate_uses_deterministic_word_limit(self):
        ranker = BedrockSetwiseLlmRanker("qwen.test", client=FakeBedrockClient([]))
        self.assertEqual("one two", ranker.truncate(" one  two three ", 2))


if __name__ == "__main__":
    unittest.main()
