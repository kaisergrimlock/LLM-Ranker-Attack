"""Test durable resume behavior for interrupted ranking evaluations."""

import tempfile
import unittest
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation_checkpoint import EvaluationCheckpoint, run_checkpointed
from llm_client import generation_fingerprint


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

    def test_reasoning_effort_mismatch_forks_checkpoint(self):
        """A GPT-OSS mismatch leaves the old file and forks settings."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranking.checkpoint.json"
            old = EvaluationCheckpoint(
                path,
                {
                    "model_name": "openai.gpt-oss-20b-1:0",
                    "gpt_oss_reasoning_effort": "low",
                    "qwen_thinking_mode": "default",
                    "bedrock_max_tokens": "512",
                },
            )
            requested = {
                "model_name": "openai.gpt-oss-20b-1:0",
                "gpt_oss_reasoning_effort": "high",
                "qwen_thinking_mode": "default",
                "bedrock_max_tokens": "512",
            }
            forked = EvaluationCheckpoint(path, requested, resume=True)
            self.assertNotEqual(forked.path, old.path)
            self.assertTrue(forked.path.exists())
            self.assertEqual(forked.state["fingerprint"], requested)
            resumed = EvaluationCheckpoint(path, requested, resume=True)
            self.assertEqual(resumed.path, forked.path)

    def test_token_budget_mismatch_creates_separate_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranking.checkpoint.json"
            fingerprint = {
                "model_name": "qwen.qwen3-32b-v1:0",
                "qwen_thinking_mode": "off",
                "gpt_oss_reasoning_effort": None,
                "bedrock_max_tokens": "512",
            }
            EvaluationCheckpoint(path, fingerprint)
            changed = {**fingerprint, "bedrock_max_tokens": "4096"}
            forked = EvaluationCheckpoint(path, changed, resume=True)
            self.assertNotEqual(forked.path, path)
            self.assertEqual(forked.state["fingerprint"], changed)

    def test_qwen_thinking_mode_mismatch_creates_separate_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranking.checkpoint.json"
            off = {
                "model_name": "qwen.qwen3-32b-v1:0",
                "qwen_thinking_mode": "off",
                "gpt_oss_reasoning_effort": None,
                "bedrock_max_tokens": "512",
            }
            on = {**off, "qwen_thinking_mode": "on"}
            first = EvaluationCheckpoint(path, off)
            second = EvaluationCheckpoint(path, on, resume=True)
            self.assertNotEqual(first.path, second.path)
            self.assertEqual(second.state["fingerprint"], on)

    def test_missing_fingerprint_keys_are_forked(self):
        """Legacy checkpoints lacking generation settings are ignored."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranking.checkpoint.json"
            EvaluationCheckpoint(path, {"model_name": "qwen.qwen3-32b-v1:0"})
            requested = {
                "model_name": "qwen.qwen3-32b-v1:0",
                "qwen_thinking_mode": "off",
                "gpt_oss_reasoning_effort": None,
                "bedrock_max_tokens": "4096",
            }
            forked = EvaluationCheckpoint(path, requested, resume=True)
            self.assertNotEqual(forked.path, path)
            self.assertEqual(forked.state["fingerprint"], requested)

    def test_completed_checkpoint_remains_readable(self):
        """Invalidating a checkpoint does not affect completed result artifacts."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranking.checkpoint.json"
            fingerprint = {"model_name": "openai.gpt-oss-20b-1:0", "budget": "512"}
            checkpoint = EvaluationCheckpoint(path, fingerprint)
            checkpoint.mark_complete()
            resumed = EvaluationCheckpoint(path, fingerprint, resume=True)
            self.assertTrue(resumed.state["complete"])


class GenerationFingerprintTests(unittest.TestCase):
    """Ensure provider-specific environment settings are isolated."""

    def test_provider_settings_are_scoped_to_model(self):
        with patch.dict(
            os.environ,
            {
                "QWEN_THINKING_MODE": "ON",
                "GPT_OSS_REASONING_EFFORT": "HIGH",
                "BEDROCK_MAX_TOKENS": "4096",
            },
            clear=False,
        ):
            qwen = generation_fingerprint("qwen.qwen3-32b-v1:0")
            gpt = generation_fingerprint("openai.gpt-oss-20b-1:0")
            other = generation_fingerprint("meta.llama3-8b-instruct-v1:0")
        self.assertEqual(qwen["qwen_thinking_mode"], "on")
        self.assertIsNone(qwen["gpt_oss_reasoning_effort"])
        self.assertEqual(gpt["gpt_oss_reasoning_effort"], "high")
        self.assertEqual(gpt["qwen_thinking_mode"], "default")
        self.assertEqual(other["qwen_thinking_mode"], "default")
        self.assertIsNone(other["gpt_oss_reasoning_effort"])
