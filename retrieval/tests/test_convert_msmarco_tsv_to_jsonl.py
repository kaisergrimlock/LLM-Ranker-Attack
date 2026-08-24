from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "convert_msmarco_tsv_to_jsonl.py"
SPEC = importlib.util.spec_from_file_location("convert_msmarco", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CollectionConversionTests(unittest.TestCase):
    def test_conversion_preserves_tabs_in_passage_text_and_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "collection.tsv"
            source.write_text("1\tfirst\n2\tsecond\twith tab\n3\tthird\n", encoding="utf-8")
            output = root / "jsonl"

            documents, shards = MODULE.convert_collection(
                source,
                output,
                documents_per_shard=2,
                expected_documents=3,
            )

            self.assertEqual((documents, shards), (3, 2))
            rows = []
            for path in sorted(output.glob("*.jsonl")):
                rows.extend(json.loads(line) for line in path.read_text().splitlines())
            self.assertEqual(
                rows,
                [
                    {"id": "1", "contents": "first"},
                    {"id": "2", "contents": "second\twith tab"},
                    {"id": "3", "contents": "third"},
                ],
            )
            marker = output.with_name(output.name + ".complete")
            self.assertEqual(json.loads(marker.read_text())["documents"], 3)

    def test_wrong_document_count_does_not_write_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "collection.tsv"
            source.write_text("1\tone\n", encoding="utf-8")
            output = root / "jsonl"
            with self.assertRaisesRegex(RuntimeError, "expected 2"):
                MODULE.convert_collection(
                    source,
                    output,
                    documents_per_shard=2,
                    expected_documents=2,
                )
            self.assertFalse(output.with_name(output.name + ".complete").exists())


if __name__ == "__main__":
    unittest.main()
