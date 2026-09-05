"""Regression checks for result discovery and portable CSV merging."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import update_attack_outcomes as updater


class OutcomeDiscoveryTests(unittest.TestCase):
    """Check filename-independent discovery without losing remote results."""

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


if __name__ == "__main__":
    unittest.main()
