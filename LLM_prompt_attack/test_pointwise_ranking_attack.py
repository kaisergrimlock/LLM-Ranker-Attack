"""Test pointwise binary-label accounting without invoking a model provider."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Results"))

import calculate_prompt_tokens as token_counter
import pointwise_ranking_attack_openai as pointwise
import update_attack_outcomes as outcome_updater
from evaluation_checkpoint import EvaluationCheckpoint, run_checkpointed


class PointwiseEvaluationTests(unittest.TestCase):
    """Check strict labels, passage attacks, and requested-denominator outcomes."""

    def test_parse_binary_label_is_strict(self):
        """Accept only a complete Yes or No response."""
        self.assertEqual(pointwise.parse_binary_label(" YES "), "Yes")
        self.assertEqual(pointwise.parse_binary_label("no"), "No")
        self.assertEqual(pointwise.parse_binary_label("Yes."), "INVALID")
        self.assertEqual(pointwise.parse_binary_label(""), "INVALID")

    def test_no_to_yes_counts_keep_invalid_and_clean_yes_in_denominator(self):
        """Count only clean-No to attacked-Yes changes as attack successes."""
        counts = pointwise.count_no_to_yes(
            ["No", "Yes", "INVALID", "No"],
            ["Yes", "Yes", "Yes", "INVALID"],
        )
        self.assertEqual(counts["requested"], 4)
        self.assertEqual(counts["pointwise_flip_count"], 1)
        self.assertEqual(counts["attacked_valid_rankings"], 3)

    def test_query_injection_is_appended_to_the_passage(self):
        """Keep the original passage and append the query-injection payload."""
        query = "what is a binary label"
        passage = pointwise.Passage("d1", "Original passage.", 0)
        attacked = passage.text + pointwise.jailbreak_prompt["qi"].format(query=query)
        self.assertTrue(attacked.startswith(passage.text))
        self.assertIn(query, attacked)

    def test_checkpoint_resume_uses_saved_responses(self):
        """Avoid duplicate provider calls when a matching checkpoint is resumed."""
        fingerprint = {"paradigm": "pointwise", "seed": 42}
        calls = []

        def worker(value):
            calls.append(value)
            return {"label": "No", "value": value}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "pointwise.checkpoint.json"
            first = EvaluationCheckpoint(checkpoint_path, fingerprint)
            self.assertEqual(
                run_checkpointed([1, 2], worker, first, "clean", 1, 1, "test"),
                [{"label": "No", "value": 1}, {"label": "No", "value": 2}],
            )
            resumed = EvaluationCheckpoint(checkpoint_path, fingerprint, resume=True)
            run_checkpointed([1, 2], worker, resumed, "clean", 1, 1, "test")
        self.assertEqual(calls, [1, 2])

    def test_mocked_run_writes_discoverable_outcomes_and_token_rows(self):
        """Write all pointwise artifacts without calling a dataset or model provider."""
        instances = [("query", pointwise.Passage("d1", "passage", 0))]
        clean = [
            {"label": "No", "prompt": "clean prompt", "response": "No", "usage": {}}
        ]
        attacked = [
            {
                "label": "Yes",
                "prompt": "attacked prompt",
                "response": "Yes",
                "usage": {},
            }
        ]

        class FakeTokenizer:
            """Count whitespace-separated words for the smoke test."""

            def encode(self, prompt, add_special_tokens=False):
                return prompt.split()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result_smoke.jsonl"
            detail_path = root / "detail_smoke.json"
            checkpoint_path = root / "smoke.checkpoint.json"
            argv = [
                "pointwise",
                "--model_name",
                "qwen/qwen3-4b",
                "--result_json_path",
                str(result_path),
                "--detailed_results",
                str(detail_path),
                "--checkpoint_path",
                str(checkpoint_path),
            ]
            with (
                patch.object(pointwise, "prepare_passages", return_value=instances),
                patch.object(pointwise, "_evaluate", side_effect=[clean, attacked]),
                patch.object(sys, "argv", argv),
            ):
                self.assertEqual(pointwise.main(), 0)

            summary = json.loads(result_path.read_text(encoding="utf-8"))
            details = json.loads(detail_path.read_text(encoding="utf-8"))
            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(summary["pointwise_flip_count"], 1)
            self.assertEqual(len(details), 2)
            with (
                patch.object(outcome_updater, "PROJECT_ROOT", root),
                patch.object(token_counter, "PROJECT_ROOT", root),
            ):
                outcome = outcome_updater._outcome_from_record(summary, result_path, 1)
                self.assertEqual(outcome["Attack success (%)"], 100)

                run = token_counter._selected_runs(root)[0]
                token_rows = token_counter._count_run_prompts(
                    run, FakeTokenizer(), "fake-tokenizer"
                )
            self.assertEqual({row["Paradigm"] for row in token_rows}, {"Pointwise"})
            self.assertEqual(
                {row["Phase"] for row in token_rows}, {"Clean", "Attacked"}
            )


if __name__ == "__main__":
    unittest.main()
