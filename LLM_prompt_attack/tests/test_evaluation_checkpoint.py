"""Test durable resume behavior for interrupted ranking evaluations."""

import tempfile
import unittest
from pathlib import Path

from evaluation_checkpoint import EvaluationCheckpoint, run_checkpointed


class EvaluationCheckpointTests(unittest.TestCase):
    """Exercise checkpoint persistence without making model requests."""

    def test_resume_reuses_completed_batches(self):
        """Resume only the batch that was not saved before an interruption."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranking.checkpoint.json"
            fingerprint = {"paradigm": "pairwise", "seed": 42}
            calls = []

            def interrupted_worker(item):
                calls.append(item)
                if item == 2:
                    raise RuntimeError("expired token")
                return {"label": item}

            checkpoint = EvaluationCheckpoint(path, fingerprint)
            with self.assertRaisesRegex(RuntimeError, "expired token"):
                run_checkpointed(
                    [0, 1, 2, 3],
                    interrupted_worker,
                    checkpoint,
                    "clean",
                    n_jobs=1,
                    batch_size=2,
                    description="test",
                )
            self.assertEqual(calls, [0, 1, 2])

            resumed_calls = []
            resumed = EvaluationCheckpoint(path, fingerprint, resume=True)
            records = run_checkpointed(
                [0, 1, 2, 3],
                lambda item: resumed_calls.append(item) or {"label": item},
                resumed,
                "clean",
                n_jobs=1,
                batch_size=2,
                description="test",
            )

            self.assertEqual(resumed_calls, [2, 3])
            self.assertEqual(
                records, [{"label": 0}, {"label": 1}, {"label": 2}, {"label": 3}]
            )
