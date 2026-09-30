"""Locate or download exactly the ablation files referenced by the CSV."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "Results" / "ablation_study.csv"
DEFAULT_OUTPUT = ROOT / "Results" / "ablation_server"
DEFAULT_ENV_FILE = ROOT / ".env"


def _load_env(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without adding a dotenv dependency."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _sources(csv_path: Path) -> list[str]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        values = {row.get("Source", "").replace("\\", "/") for row in csv.DictReader(handle)}
    return sorted(value for value in values if value)


def _relative_destination(source: str) -> Path:
    return Path(source.replace("/", "\\"))


def main(args: argparse.Namespace) -> int:
    dotenv = _load_env(args.env_file)
    ssh_password = os.environ.get("ABLATION_SSH_PASSWORD", dotenv.get("ABLATION_SSH_PASSWORD"))
    sources = _sources(args.csv)
    if args.source_contains:
        sources = [source for source in sources if args.source_contains in source]
    if args.include_checkpoints:
        checkpoint_sources = {
            Path(source).with_suffix(".checkpoint.json").as_posix()
            for source in sources
        }
        sources = sorted(set(sources) | checkpoint_sources)
    if not sources:
        print(f"No Source paths found in {args.csv}")
        return 1
    missing = []
    copied = 0
    failed = 0
    args.output.mkdir(parents=True, exist_ok=True)
    for source in sources:
        local_source = ROOT / Path(source)
        destination = args.output / _relative_destination(source)
        if local_source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if args.mode == "copy":
                shutil.copy2(local_source, destination)
                copied += 1
            print(f"local\t{source}")
            continue
        missing.append(source)
        print(f"missing\t{source}")

    if args.mode == "download" and missing:
        if not args.remote_host:
            print("--remote-host is required with --mode download")
            return 2
        for source in missing:
            destination = args.output / _relative_destination(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            remote_path = f"{args.remote_root.rstrip('/')}/{source.lstrip('/')}"
            command = ["scp"]
            command_env = None
            if ssh_password and shutil.which("sshpass"):
                command = ["sshpass", "-e", "scp"]
                command_env = os.environ.copy()
                command_env["SSHPASS"] = ssh_password
            elif ssh_password and not args.dry_run:
                print("Warning: ABLATION_SSH_PASSWORD is set, but sshpass is unavailable; scp will prompt.")
            if args.jump_host:
                command += ["-J", args.jump_host]
            command += [f"{args.remote_host}:{remote_path}", str(destination)]
            if args.dry_run:
                print("download\t" + " ".join(command))
            else:
                print("Downloading", source)
                result = subprocess.run(command, check=False, env=command_env)
                if result.returncode != 0:
                    failed += 1
                    continue
                copied += 1

    remaining = len(missing) if args.mode != "download" or args.dry_run else failed
    print(
        f"Manifest entries: {len(sources)}; local/downloaded: {copied}; "
        f"missing/failed: {remaining}"
    )
    return 0 if ((args.mode == "download" and failed == 0) or not missing) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("locate", "copy", "download"), default="locate")
    parser.add_argument("--remote-host", help="Remote login host, e.g. s3891987@segsresap07.int.its.rmit.edu.au")
    parser.add_argument("--jump-host", help="SSH jump host, e.g. s3891987@wight.seg.rmit.edu.au")
    parser.add_argument("--remote-root", default="/research/remote/petabyte/users/s3891987/LLM-Ranker-Attack")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--source-contains",
        help="Only sync CSV sources containing this substring.",
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Also sync each source's per-call .checkpoint.json companion.",
    )
    parser.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(parser.parse_args()))
