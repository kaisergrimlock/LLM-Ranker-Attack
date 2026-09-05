"""Verify query injection across the three reranking paradigms."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import listwise_ranking_attack_openai as listwise  # noqa: E402
import pairwise_ranking_attack_openai as pairwise  # noqa: E402
import setwise_ranking_attack_openai as setwise  # noqa: E402
from prompts import jailbreak_prompt, listwise_jailbreak_prompt  # noqa: E402


class QueryInjectionTests(unittest.TestCase):
    """Verify the exact query is appended only to the attack target."""

    def test_pairwise_query_injection(self):
        """Append the query to the passage rejected during clean ranking."""
        query = "which document has the relevant answer"
        pairs = [
            (
                query,
                pairwise.Document("a", "winner", 3),
                pairwise.Document("b", "target", 0),
            )
        ]

        attacked_pairs = pairwise.apply_attack(
            ["A"], pairs, jailbreak_prompt["qi"], attack_position="back"
        )

        _, clean_winner, attacked_target = attacked_pairs[0]
        self.assertEqual(clean_winner.text, "winner")
        self.assertEqual(attacked_target.text, f"target\n\nQuery: {query}")

    def test_setwise_query_injection(self):
        """Append the query to the only non-selected passage in a set."""
        query = "setwise query"
        sets = [
            (
                query,
                [
                    setwise.Document("a", "winner", 3),
                    setwise.Document("b", "target", 0),
                ],
            )
        ]

        attacked_sets, attack_labels = setwise.apply_attack(
            ["A"], sets, jailbreak_prompt["qi"], attack_position="back"
        )

        self.assertEqual(attack_labels, ["B"])
        self.assertEqual(attacked_sets[0][1][0].text, "winner")
        self.assertEqual(attacked_sets[0][1][1].text, f"target\n\nQuery: {query}")

    def test_listwise_query_injection(self):
        """Append the query to the only passage below the clean winner."""
        query = "listwise query"
        sets = [
            (
                query,
                [
                    listwise.Document("a", "winner", 3),
                    listwise.Document("b", "target", 0),
                ],
            )
        ]

        attacked_sets, attack_labels = listwise.apply_attack(
            [["A", "B"]],
            sets,
            listwise_jailbreak_prompt["qi"],
            attack_position="back",
        )

        self.assertEqual(attack_labels, ["B"])
        self.assertEqual(attacked_sets[0][1][0].text, "winner")
        self.assertEqual(attacked_sets[0][1][1].text, f"target\n\nQuery: {query}")

    def test_static_attack_text_is_unchanged(self):
        """Keep existing static DOH and DCH prompts compatible."""
        pairs = [
            (
                "unused query",
                pairwise.Document("a", "winner", 3),
                pairwise.Document("b", "target", 0),
            )
        ]

        attacked_pairs = pairwise.apply_attack(
            ["A"], pairs, " static attack", attack_position="back"
        )

        self.assertEqual(attacked_pairs[0][2].text, "target static attack")


if __name__ == "__main__":
    unittest.main()
