"""Tests for reusable Filter QI cache entries."""
# ruff: noqa: D101, D102, D103

import json
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import filter_defense  # noqa: E402

Document = namedtuple("Document", "doc_id text relevance")


def make_args(root, **overrides):
    values = dict(
        prompt_mode="filter_qi",
        model_name="ranker",
        provider="openai",
        base_url="http://ranker/v1",
        aws_region=None,
        filter_model="filter-model",
        filter_provider="openai",
        filter_base_url="http://filter/v1",
        filter_aws_region=None,
        filter_max_tokens=100,
        filter_cache_dir=str(Path(root) / "cache"),
        filter_cache_mode="read-write",
        dataset_name="dataset",
        result_json_path=str(Path(root) / "result.jsonl"),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def instances(text="original\n\nQuery: q"):
    clean = [("q", Document("doc", "original", 3), Document("n", "other", 0))]
    attacked = [("q", Document("doc", text, 3), Document("n", "other", 0))]
    return clean, attacked


class FilterCacheTests(unittest.TestCase):
    def test_key_is_stable_and_settings_are_part_of_key(self):
        metadata = {
            "filter_prompt_version": "v1",
            "filter_model": "m",
            "filter_provider": "p",
            "filter_base_url": "u",
            "filter_aws_region": None,
            "filter_max_tokens": 10,
        }
        first = filter_defense.cache_key(metadata, "d", "q", "id", "text")
        self.assertEqual(
            first, filter_defense.cache_key(metadata, "d", "q", "id", "text")
        )
        changed = dict(metadata, filter_max_tokens=11)
        self.assertNotEqual(
            first, filter_defense.cache_key(changed, "d", "q", "id", "text")
        )

    def test_second_run_hits_cache_and_cross_scheme_input_can_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            clean, attacked = instances()
            client = Mock()
            client.generate.return_value = "filtered"
            args = make_args(temp)
            with patch.object(
                filter_defense, "get_ranking_client", return_value=client
            ):
                result = filter_defense.filter_attacked_instances(
                    clean, attacked, args, pairwise=True
                )
            self.assertEqual(client.generate.call_count, 1)
            self.assertEqual(result[0][1].text, "filtered")
            self.assertEqual(
                len(list((Path(args.filter_cache_dir) / "injected").glob("*.json"))),
                1,
            )
            self.assertEqual(
                len(list((Path(args.filter_cache_dir) / "filtered").glob("*.json"))),
                1,
            )

            client.generate.reset_mock()
            args.result_json_path = str(Path(temp) / "second.jsonl")
            clean2, attacked2 = instances()
            with patch.object(
                filter_defense, "get_ranking_client", return_value=client
            ):
                filter_defense.filter_attacked_instances(
                    clean2, attacked2, args, pairwise=True
                )
            self.assertEqual(client.generate.call_count, 0)
            audit = json.loads(
                Path(str(args.result_json_path) + ".filter.jsonl").read_text()
            )
            self.assertTrue(audit["cache_hit"])
            self.assertEqual(audit["doc_id"], "doc")
            self.assertEqual(attacked2[0][1].relevance, 3)

    def test_malformed_entry_is_regenerated_and_failure_leaves_no_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            clean, attacked = instances()
            args = make_args(temp)
            client = Mock()
            client.generate.return_value = "filtered"
            with patch.object(
                filter_defense, "get_ranking_client", return_value=client
            ):
                filter_defense.filter_attacked_instances(
                    clean, attacked, args, pairwise=True
                )
            cache_file = next((Path(args.filter_cache_dir) / "filtered").glob("*.json"))
            cache_file.write_text("{}", encoding="utf-8")
            client.generate.reset_mock()
            clean2, attacked2 = instances()
            args.result_json_path = str(Path(temp) / "regen.jsonl")
            with patch.object(
                filter_defense, "get_ranking_client", return_value=client
            ):
                filter_defense.filter_attacked_instances(
                    clean2, attacked2, args, pairwise=True
                )
            self.assertEqual(client.generate.call_count, 1)

            client.generate.side_effect = RuntimeError("failure")
            cache_file.unlink()
            args.result_json_path = str(Path(temp) / "failed.jsonl")
            with patch.object(
                filter_defense, "get_ranking_client", return_value=client
            ):
                with self.assertRaises(RuntimeError):
                    clean3, attacked3 = instances()
                    filter_defense.filter_attacked_instances(
                        clean3, attacked3, args, pairwise=True
                    )
            self.assertFalse(
                any(
                    path.name.startswith(".") and path.suffix == ".tmp"
                    for path in Path(args.filter_cache_dir).iterdir()
                )
            )

    def test_token_limit_reuses_injected_passage_and_caches_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            clean, attacked = instances()
            args = make_args(temp)
            client = Mock()
            client.generate.side_effect = RuntimeError(
                "The maximum tokens you requested exceeds the model limit of 2048"
            )
            with patch.object(
                filter_defense, "get_ranking_client", return_value=client
            ):
                result = filter_defense.filter_attacked_instances(
                    clean, attacked, args, pairwise=True
                )

            self.assertEqual(result[0][1].text, attacked[0][1].text)
            audit = json.loads(
                Path(str(args.result_json_path) + ".filter.jsonl").read_text()
            )
            self.assertEqual(audit["status"], "ok")
            self.assertEqual(audit["filter_outcome"], "fallback_unfiltered_token_limit")
            self.assertEqual(
                len(list((Path(args.filter_cache_dir) / "filtered").glob("*.json"))), 1
            )

    def test_truncated_filter_response_reuses_injected_passage(self):
        with tempfile.TemporaryDirectory() as temp:
            clean, attacked = instances()
            args = make_args(temp)
            client = Mock()
            client.generate.side_effect = RuntimeError(
                "Filter response did not complete: max_tokens"
            )
            with patch.object(
                filter_defense, "get_ranking_client", return_value=client
            ):
                result = filter_defense.filter_attacked_instances(
                    clean, attacked, args, pairwise=True
                )

            self.assertEqual(result[0][1].text, attacked[0][1].text)

    def test_empty_filter_response_can_use_explicit_fallback_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            clean, attacked = instances()
            args = make_args(temp, filter_failure_policy="fallback-unfiltered")
            client = Mock()
            client.generate.return_value = ""
            with patch.object(
                filter_defense, "get_ranking_client", return_value=client
            ):
                result = filter_defense.filter_attacked_instances(
                    clean, attacked, args, pairwise=True
                )

            self.assertEqual(result[0][1].text, attacked[0][1].text)
            audit = json.loads(
                Path(str(args.result_json_path) + ".filter.jsonl").read_text()
            )
            self.assertEqual(
                audit["filter_outcome"], "fallback_unfiltered_empty_response"
            )

    def test_read_only_mode_requires_materialized_filtered_passage(self):
        """Prevent a reranking-only run from silently invoking the filter model."""
        with tempfile.TemporaryDirectory() as temp:
            clean, attacked = instances()
            args = make_args(temp, filter_cache_mode="read-only")
            client = Mock()
            with patch.object(
                filter_defense, "get_ranking_client", return_value=client
            ):
                with self.assertRaisesRegex(RuntimeError, "Filtering failed"):
                    filter_defense.filter_attacked_instances(
                        clean, attacked, args, pairwise=True
                    )
            client.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
