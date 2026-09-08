"""Check keyword insertion and target isolation without model calls."""

import csv
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import listwise_ranking_attack_openai as listwise  # noqa: E402
import pairwise_ranking_attack_openai as pairwise  # noqa: E402
import setwise_ranking_attack_openai as setwise  # noqa: E402
from keyword_injection import KeywordInjection  # noqa: E402


class KeywordInjectionTests(unittest.TestCase):
    """Exercise shared insertion logic and all three target selectors."""

    def setUp(self):
        """Create a small query-keyword fixture."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "keywords.tsv"
        with self.path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, delimiter="\t")
            writer.writerow(["query", "keywords"])
            writer.writerow(["query", json.dumps(["KEY_ONE", "KEY TWO"])])

    def test_insertions_preserve_text_and_rng(self):
        """Keep original characters, intact phrases, and target RNG state."""
        text = "alpha  beta\ngamma delta epsilon zeta"
        state = random.getstate()
        attack = KeywordInjection(self.path, 42)
        result = attack("query", text)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(result, KeywordInjection(self.path, 42)("query", text))
        self.assertEqual(result.count("KEY_ONE"), 1)
        self.assertEqual(result.count("KEY TWO"), 1)
        self.assertEqual(result.replace(" KEY_ONE ", "").replace(" KEY TWO ", ""), text)
        variants = {
            KeywordInjection(self.path, seed)("query", text) for seed in range(8)
        }
        self.assertGreater(len(variants), 1)

    def test_targets_across_paradigms(self):
        """Modify only the rejected passage and retain document metadata."""
        for module in (pairwise, setwise, listwise):
            with self.subTest(module=module.__name__):
                attack = KeywordInjection(self.path, 42)
                winner = module.Document("a", "winner", 3)
                target = module.Document("b", "target passage", 2)
                if module is pairwise:
                    instances = [("query", winner, target)]
                    changed = module.apply_attack(["A"], instances, attack, "random")
                    documents = changed[0][1:]
                else:
                    instances = [("query", [winner, target])]
                    results = [["A", "B"]] if module is listwise else ["A"]
                    changed, labels = module.apply_attack(
                        results, instances, attack, "random"
                    )
                    self.assertEqual(labels, ["B"])
                    documents = changed[0][1]
                self.assertEqual(documents[0].text, "winner")
                self.assertEqual(documents[1].doc_id, "b")
                self.assertEqual(documents[1].relevance, 2)
                self.assertIn("KEY_ONE", documents[1].text)
                self.assertIn("KEY TWO", documents[1].text)
                self.assertEqual(target.text, "target passage")

    def test_missing_query_fails(self):
        """Require keyword coverage before evaluation."""
        attack = KeywordInjection(self.path, 42)
        with self.assertRaisesRegex(ValueError, "Missing keywords"):
            attack.validate_queries([("unmapped query", [])])

    def test_empty_passage(self):
        """Support a target with no existing word boundaries."""
        result = KeywordInjection(self.path, 42)("query", "")
        self.assertIn("KEY_ONE", result)
        self.assertIn("KEY TWO", result)


if __name__ == "__main__":
    unittest.main()
