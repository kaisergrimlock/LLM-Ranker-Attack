"""Offline tests for the pairwise marker-aware defense prompt."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pairwise_ranking_attack_openai import (  # noqa: E402
    Document,
    _process_single_query_pairwise,
    apply_attack,
)
from prompts import jailbreak_prompt, pairwise_ranking_defense  # noqa: E402


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

    def test_query_injection_appends_the_instance_query_to_the_target(self):
        """Append the current query only to the clean-stage losing passage."""
        query = "which document has the relevant answer"
        pairs = [(query, Document("a", "winner", 3), Document("b", "target", 0))]

        attacked_pairs = apply_attack(
            ["A"], pairs, jailbreak_prompt["qi"], attack_position="back"
        )

        _, clean_winner, attacked_target = attacked_pairs[0]
        self.assertEqual(clean_winner.text, "winner")
        self.assertEqual(
            attacked_target.text,
            "target\n\nQuery: which document has the relevant answer",
        )


if __name__ == "__main__":
    unittest.main()
