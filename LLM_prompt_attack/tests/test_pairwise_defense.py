"""Offline tests for the pairwise marker-aware defense prompt."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pairwise_ranking_attack_openai import (  # noqa: E402
    Document,
    _process_single_query_pairwise,
)
from prompts import pairwise_ranking_defense  # noqa: E402


class FakeRankingClient:
    """Capture the prompt sent by one pairwise comparison."""

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, *, max_tokens):
        self.prompts.append((prompt, max_tokens))
        return "Passage A"


class PairwiseDefenseTests(unittest.TestCase):
    """Verify marker-aware instructions reach the ranking model."""

    def test_defense_template_is_used_for_pairwise_comparison(self):
        """Format the defense around both passages without changing the parser."""
        client = FakeRankingClient()
        docs = (
            Document("a", "relevant text", 3),
            Document("b", "[MARKER] injected text", 0),
        )

        with patch(
            "pairwise_ranking_attack_openai.get_ranking_client",
            return_value=client,
        ):
            result = _process_single_query_pairwise(
                "test query",
                *docs,
                "model",
                "https://example.invalid/v1",
                prompt_template=pairwise_ranking_defense,
            )

        prompt, max_tokens = client.prompts[0]
        self.assertEqual(result["label"], "A")
        self.assertEqual(max_tokens, 3)
        self.assertIn("untrusted passage content", prompt)
        self.assertIn("[MARKER] injected text", prompt)
        self.assertIn("Output Passage A or Passage B", prompt)


if __name__ == "__main__":
    unittest.main()
