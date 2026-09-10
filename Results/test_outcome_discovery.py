"""Regression checks for result discovery and portable CSV merging."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import update_attack_outcomes as updater
import update_attack_table as table_updater


class OutcomeDiscoveryTests(unittest.TestCase):
    """Check filename-independent discovery without losing remote results."""

    def test_filter_defense_preserves_counts_and_remains_separate(self):
        """Keep filter rows distinct with the same requested-instance percentages."""
        for scheme, success_field in (
            ("pairwise", "flipped_count"),
            ("setwise", "attack_success_count"),
            ("listwise", "attack_top_position_count"),
        ):
            record = {
                "model_name": "qwen.qwen3-32b-v1:0",
                "dataset_name": "msmarco-passage/trec-dl-2019",
                "ranking_scheme": scheme,
                "attack_type": "qi",
                "attack_position": "back",
                "prompt_mode": "filter_qi",
                "original_total_rankings": 100,
                "attacked_valid_rankings": 80,
                "total_queries": 80,
                success_field: 20,
            }
            source = updater.PROJECT_ROOT / "LLM_prompt_attack/outputs/test.jsonl"
            row = updater._outcome_from_record(record, source, 1)
            self.assertEqual(row["Prompt"], "Filter QI")
            self.assertEqual(row["Discarded (%)"], 20)
            self.assertEqual(row["Attack success (%)"], 20)
            self.assertEqual(row["Valid attack failure (%)"], 60)
            baseline = updater._outcome_from_record(
                dict(record, prompt_mode="standard"), source, 1
            )
            self.assertNotEqual(updater._key(row), updater._key(baseline))

    def test_filenames_subfolders_duplicates_and_retained_rows(self):
        """Find both naming styles and retain existing rows on an empty host."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            outputs.mkdir()
            nested = outputs / "nested"
            nested.mkdir()
            record = {
                "model_name": "Qwen3-4B",
                "dataset_name": "msmarco-passage/trec-dl-2019",
                "ranking_scheme": "pairwise",
                "attack_type": "qi",
                "attack_position": "back",
                "prompt_mode": "standard",
                "total_queries": 4096,
                "flipped_count": 123,
            }
            payload = json.dumps(record) + "\n"
            (outputs / "result_prefix.jsonl").write_text(payload, encoding="utf-8")
            (nested / "suffix_result.jsonl").write_text(payload, encoding="utf-8")
            second = dict(record, dataset_name="msmarco-passage/trec-dl-2020")
            (outputs / "arbitrary.jsonl").write_text(
                json.dumps(second) + '\n[]\n{"unrelated": true}\ninvalid\n',
                encoding="utf-8",
            )
            with (
                patch.object(updater, "PROJECT_ROOT", root),
                patch.object(updater, "OUTPUT_DIR", outputs),
                patch.object(updater, "OUTCOME_CSV", root / "outcomes.csv"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                updater.main()
                rows = updater._existing_rows()
                self.assertEqual(len(rows), 2)
                self.assertTrue(
                    all(row["Attack success"] == 123 for row in rows.values())
                )
                # A later host with no raw results must retain the portable rows.
                empty = root / "empty"
                empty.mkdir()
                with patch.object(updater, "OUTPUT_DIR", empty):
                    updater.main()
                self.assertEqual(updater._existing_rows(), rows)

    def test_newer_result_wins_over_a_more_complete_older_result(self):
        """Prefer a corrected rerun by date, even when it has fewer valid outputs."""
        older = {
            "Date": "2026-09-09 12:00:00",
            "Source": "outputs/older.jsonl",
            "Line": 1,
            "Requested": 4096,
            "Valid attacked": 4096,
        }
        newer = {
            "Date": "2026-09-10 12:00:00",
            "Source": "outputs/newer.jsonl",
            "Line": 1,
            "Requested": 4096,
            "Valid attacked": 4000,
        }
        self.assertGreater(updater._priority(newer), updater._priority(older))

    def test_table_priority_uses_the_newest_raw_result(self):
        """Keep the latest raw table metric when rerun validity differs."""
        older = {
            "date": "2026-09-09 12:00:00",
            "path": "outputs/older.jsonl",
            "line": 1,
            "denominator": 4096,
        }
        newer = {
            "date": "2026-09-10 12:00:00",
            "path": "outputs/newer.jsonl",
            "line": 1,
            "denominator": 4000,
        }
        self.assertGreater(
            table_updater._priority(newer), table_updater._priority(older)
        )


if __name__ == "__main__":
    unittest.main()
