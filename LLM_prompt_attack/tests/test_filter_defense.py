"""Verify passage filtering preserves the existing ranking measurements."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import filter_defense  # noqa: E402
import listwise_ranking_attack_openai as listwise  # noqa: E402
import pairwise_ranking_attack_openai as pairwise  # noqa: E402
import setwise_ranking_attack_openai as setwise  # noqa: E402
from llm_client import RankingClient  # noqa: E402


class FilterDefenseTests(unittest.TestCase):
    """Exercise standard reranking and unchanged valid-instance denominators."""

    def test_all_evaluators_keep_metrics_and_standard_prompts(self):
        """Keep one valid success after identical clean/attacked exclusions."""
        for module, scheme in (
            (pairwise, "pairwise"),
            (setwise, "setwise"),
            (listwise, "listwise"),
        ):
            with self.subTest(scheme=scheme), tempfile.TemporaryDirectory() as temp:
                result_path = Path(temp) / "result.jsonl"
                docs = [module.Document(str(i), f"passage {i}", i) for i in range(4)]
                instances = [
                    ("query", *docs[:2]) if scheme == "pairwise" else ("query", docs)
                    for _ in range(3)
                ]
                expected_prompt = getattr(module, f"{scheme}_ranking_prompt")
                calls = []

                def rank(
                    rows,
                    *args,
                    expected_prompt=expected_prompt,
                    calls=calls,
                    scheme=scheme,
                    **kwargs,
                ):
                    self.assertEqual(kwargs["prompt_template"], expected_prompt)
                    calls.append(rows)
                    if len(calls) == 1:
                        labels = (
                            [["A", "B", "C", "D"]] * 2 + ["INVALID"]
                            if scheme == "listwise"
                            else ["A", "A", "INVALID"]
                        )
                    else:
                        self.assertEqual(len(rows), 2)
                        candidates = rows[0][1:] if scheme == "pairwise" else rows[0][1]
                        target = next(
                            i
                            for i, doc in enumerate(candidates)
                            if doc.text == "FILTERED"
                        )
                        label = chr(65 + target)
                        labels = (
                            [
                                [label]
                                + [chr(65 + i) for i in range(4) if i != target],
                                "INVALID",
                            ]
                            if scheme == "listwise"
                            else [label, "INVALID"]
                        )
                    return labels, []

                prepare = "prepare_pairs" if scheme == "pairwise" else "prepare_sets"
                client = Mock()
                client.generate.return_value = "FILTERED"
                argv = [
                    "test",
                    "--model_name",
                    "test-model",
                    "--attack_type",
                    "qi",
                    "--prompt_mode",
                    "filter_qi",
                    "--result_json_path",
                    str(result_path),
                    "--filter_cache_dir",
                    str(Path(temp) / "filter_cache"),
                ]
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(module, prepare, return_value=instances),
                    patch.object(module, "get_choices_openai", side_effect=rank),
                    patch.object(
                        filter_defense, "get_ranking_client", return_value=client
                    ),
                ):
                    module.main()
                result = json.loads(result_path.read_text())
                self.assertEqual(result["total_queries"], 1)
                self.assertEqual(result["original_total_rankings"], 3)
                self.assertEqual(result["original_valid_rankings"], 2)
                self.assertEqual(result["attacked_valid_rankings"], 1)
                metric = {
                    "pairwise": "flipped_percentage",
                    "setwise": "attack_success_rate",
                    "listwise": "attack_top_position_rate",
                }[scheme]
                self.assertEqual(result[metric], 100)
                self.assertEqual(result["reranker_prompt_mode"], "standard")
                expected_filter_calls = 1 if scheme == "pairwise" else 2
                self.assertEqual(client.generate.call_count, expected_filter_calls)
                audit = [
                    json.loads(line)
                    for line in Path(str(result_path) + ".filter.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertTrue(all(row["status"] == "ok" for row in audit))
                self.assertTrue(
                    all("Query: query" in row["injected_text"] for row in audit)
                )
                self.assertTrue(all(doc.text.startswith("passage") for doc in docs))

    def test_empty_filter_aborts_without_dropping_instances(self):
        """Record a failed filter and stop instead of reporting biased ASR."""
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(
                prompt_mode="filter_qi",
                model_name="model",
                provider="openai",
                base_url="http://localhost/v1",
                aws_region=None,
                filter_model=None,
                filter_provider=None,
                filter_base_url=None,
                filter_aws_region=None,
                filter_max_tokens=100,
                result_json_path=str(Path(temp) / "r.jsonl"),
            )
            clean = [
                (
                    "query",
                    pairwise.Document("a", "original", 0),
                    pairwise.Document("b", "winner", 3),
                )
            ]
            attacked = [
                (
                    "query",
                    pairwise.Document("a", "original\n\nQuery: query", 0),
                    clean[0][2],
                )
            ]
            client = Mock()
            client.generate.return_value = ""
            with patch.object(
                filter_defense, "get_ranking_client", return_value=client
            ):
                with self.assertRaisesRegex(RuntimeError, "Filtering failed"):
                    filter_defense.filter_attacked_instances(
                        clean, attacked, args, pairwise=True
                    )
            self.assertFalse(Path(args.result_json_path).exists())
            audit = json.loads(
                Path(args.result_json_path + ".filter.jsonl").read_text()
            )
            self.assertEqual(audit["status"], "error")

    def test_truncated_filter_outputs_rejected_for_each_provider(self):
        """Do not feed truncated text into the standard ranking evaluation."""
        bedrock = Mock()
        bedrock.converse.return_value = {"stopReason": "max_tokens"}
        openai = Mock()
        openai.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="partial"), finish_reason="length"
                )
            ]
        )
        azure = Mock()
        azure.responses.create.return_value = SimpleNamespace(status="incomplete")
        for provider, transport in (
            ("amazon-bedrock", bedrock),
            ("openai", openai),
            ("azure-openai", azure),
        ):
            with self.subTest(provider=provider):
                client = RankingClient("test", provider=provider, client=transport)
                with self.assertRaisesRegex(RuntimeError, "did not complete"):
                    client.generate("filter", max_tokens=100, require_complete=True)


if __name__ == "__main__":
    unittest.main()
