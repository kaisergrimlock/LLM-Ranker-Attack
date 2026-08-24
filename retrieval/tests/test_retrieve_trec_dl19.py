from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "retrieve_trec_dl19.py"
SPEC = importlib.util.spec_from_file_location("retrieve_trec_dl19", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RetrievalScriptTests(unittest.TestCase):
    def test_topic_text_prefers_title(self):
        self.assertEqual(
            MODULE.topic_text({"title": "primary", "description": "secondary"}),
            "primary",
        )

    def test_queries_from_records_uses_local_query_ids_and_text(self):
        records = [
            SimpleNamespace(query_id="2", text=" second "),
            SimpleNamespace(query_id="1", text="first"),
        ]
        self.assertEqual(
            MODULE.queries_from_records(records),
            {"2": "second", "1": "first"},
        )

    def test_queries_from_records_rejects_duplicate_ids(self):
        records = [
            SimpleNamespace(query_id="1", text="first"),
            SimpleNamespace(query_id="1", text="duplicate"),
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate query ID"):
            MODULE.queries_from_records(records)

    def test_write_trec_run_uses_six_column_format(self):
        results = {
            "2": [SimpleNamespace(docid="d2", score=1.5)],
            "1": [SimpleNamespace(docid="d1", score=2.0)],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.txt"
            count = MODULE.write_trec_run(
                output,
                ["2", "1"],
                results,
                depth=1,
                run_tag="test-run",
                overwrite=False,
            )
            self.assertEqual(count, 2)
            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                [
                    "1 Q0 d1 1 2.00000000 test-run",
                    "2 Q0 d2 1 1.50000000 test-run",
                ],
            )

    def test_short_result_set_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.txt"
            with self.assertRaisesRegex(RuntimeError, "Expected 2 hits"):
                MODULE.write_trec_run(
                    output,
                    ["1"],
                    {"1": [SimpleNamespace(docid="d1", score=1.0)]},
                    depth=2,
                    run_tag="test-run",
                    overwrite=False,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
