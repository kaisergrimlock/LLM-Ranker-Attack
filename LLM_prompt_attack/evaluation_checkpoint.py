"""Persist completed Bedrock ranking calls so interrupted evaluations can resume."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from joblib import Parallel, delayed
from tqdm import tqdm


class EvaluationCheckpoint:
    """Store clean and attacked ranking responses for one deterministic evaluation.

    Parameters
    ----------
    path : str or pathlib.Path
        JSON checkpoint path.
    fingerprint : dict
        Immutable run settings used to prevent an incompatible resume.
    resume : bool
        Whether to load an existing checkpoint.
    """

    def __init__(self, path, fingerprint, resume=False):
        self.path = Path(path)
        self.fingerprint = fingerprint
        if self.path.exists():
            if not resume:
                raise FileExistsError(
                    f"Checkpoint already exists: {self.path}. "
                    "Use --resume to continue it "
                    "or choose a different --checkpoint_path."
                )
            with self.path.open(encoding="utf-8") as handle:
                self.state = json.load(handle)
            if self.state.get("fingerprint") != fingerprint:
                raise ValueError(
                    "Checkpoint settings do not match this evaluation. Choose a new "
                    "--checkpoint_path rather than mixing runs."
                )
        else:
            if resume:
                raise FileNotFoundError(f"No checkpoint exists at {self.path}.")
            self.state = {"version": 1, "fingerprint": fingerprint, "phases": {}}
            self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as handle:
            json.dump(self.state, handle, ensure_ascii=False)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, self.path)

    def completed(self, phase):
        """Return saved response records for a phase, indexed by instance number."""
        return self.state["phases"].setdefault(phase, {})

    def save_batch(self, phase, start_index, records):
        """Atomically persist a successfully completed batch of responses."""
        phase_records = self.state["phases"].setdefault(phase, {})
        for offset, record in enumerate(records):
            phase_records[str(start_index + offset)] = record
        self._save()

    def mark_complete(self):
        """Mark the final summary as written without deleting resumable evidence."""
        self.state["complete"] = True
        self._save()


def run_checkpointed(items, worker, checkpoint, phase, n_jobs, batch_size, description):
    """Evaluate unsaved items in batches and return records in input order.

    A failed batch is intentionally not saved: the next ``--resume`` retry repeats at
    most that batch, while every earlier batch remains durable.
    """
    if batch_size < 1:
        raise ValueError("checkpoint_batch_size must be at least 1")

    saved = checkpoint.completed(phase)
    missing = [index for index in range(len(items)) if str(index) not in saved]
    if saved:
        print(
            f"Resuming {phase}: {len(items) - len(missing)}/{len(items)} "
            "calls already saved."
        )

    for start in tqdm(range(0, len(missing), batch_size), desc=description):
        indices = missing[start : start + batch_size]
        records = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(worker)(items[index]) for index in indices
        )
        for index, record in zip(indices, records, strict=True):
            saved[str(index)] = record
        checkpoint._save()

    return [saved[str(index)] for index in range(len(items))]
