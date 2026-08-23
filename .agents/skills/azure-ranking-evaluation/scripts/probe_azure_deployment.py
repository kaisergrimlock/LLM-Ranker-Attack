#!/usr/bin/env python3
"""List Azure GPT-5.6 candidates or prove the configured deployment is callable."""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify one GPT-5.6 deployment with a minimal Responses API call."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        help="Azure deployment name; defaults to AZURE_OPENAI_DEPLOYMENT",
    )
    parser.add_argument(
        "--list-filter",
        metavar="TEXT",
        help="List catalog IDs containing TEXT (not proof of deployment)",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Only list catalog matches; do not make an inference request",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("AZURE_OPENAI_BASE_URL"),
        help="Azure OpenAI v1 base URL; defaults to AZURE_OPENAI_BASE_URL",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=16,
        help="Output budget for the probe (default: 16)",
    )
    args = parser.parse_args()
    if args.no_probe and args.list_filter is None:
        parser.error("--no-probe requires --list-filter")
    if not args.no_probe and not args.model:
        parser.error("set AZURE_OPENAI_DEPLOYMENT or pass --model")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    return args


def main() -> int:
    args = parse_args()
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("AZURE_OPENAI_API_KEY is not set", file=sys.stderr)
        return 2
    if not args.base_url:
        print("AZURE_OPENAI_BASE_URL is not set and --base-url was omitted", file=sys.stderr)
        return 2

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url.rstrip("/") + "/",
        timeout=60,
        max_retries=0,
    )

    if args.list_filter is not None:
        needle = args.list_filter.casefold()
        try:
            matches = sorted(
                model.id
                for model in client.models.list().data
                if needle in model.id.casefold()
            )
        except Exception as exc:
            print(f"Catalog lookup failed: {exc}", file=sys.stderr)
            return 1
        print("Catalog matches (not proof of deployment):")
        print(*(matches or ["(none)"]), sep="\n")

    if args.no_probe:
        return 0

    try:
        response = client.responses.create(
            model=args.model,
            input="Reply with only A.",
            max_output_tokens=args.max_output_tokens,
            reasoning={"effort": "none"},
            text={"verbosity": "low"},
        )
    except Exception as exc:
        print(f"{args.model}: UNAVAILABLE", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    print(f"{args.model}: AVAILABLE")
    print(f"response: {response.output_text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
