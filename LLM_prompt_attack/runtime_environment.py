"""Load repository environment defaults for ranking experiments."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def _read_env_file(path: Path) -> dict[str, str]:
    values = {}
    if not path.is_file():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def configure_runtime_environment(env_file: Path = DEFAULT_ENV_FILE) -> None:
    """Configure cache and Bedrock defaults before an experiment starts.

    Parameters
    ----------
    env_file : pathlib.Path, default=DEFAULT_ENV_FILE
        Repository environment file to load. Values already present in the
        process environment take precedence.

    Returns
    -------
    None
        Environment defaults are added to ``os.environ`` in place.
    """
    for key, value in _read_env_file(env_file).items():
        os.environ.setdefault(key, value)

    os.environ.setdefault("IR_DATASETS_HOME", str(PROJECT_ROOT / "ir_datasets"))
    os.environ.setdefault("BEDROCK_MAX_TOKENS", "1024")

    dataset_home = Path(os.environ["IR_DATASETS_HOME"]).expanduser()
    if not dataset_home.is_absolute():
        dataset_home = PROJECT_ROOT / dataset_home
    os.environ["IR_DATASETS_HOME"] = str(dataset_home.resolve())


configure_runtime_environment()
