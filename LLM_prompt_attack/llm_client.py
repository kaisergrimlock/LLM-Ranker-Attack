"""Shared text-generation clients for ranking attack experiments."""

from __future__ import annotations

import os
import threading
from typing import Any

from runtime_environment import configure_runtime_environment

configure_runtime_environment()


DEFAULT_BEDROCK_REGION = "ap-southeast-2"
SUPPORTED_PROVIDERS = ("openai", "azure-openai", "amazon-bedrock")
_thread_state = threading.local()


def _normalise_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    """Convert provider token-usage fields to stable snake-case names."""
    usage = usage or {}
    aliases = {
        "input_tokens": ("inputTokens", "input_tokens", "prompt_tokens"),
        "output_tokens": ("outputTokens", "output_tokens", "completion_tokens"),
        "total_tokens": ("totalTokens", "total_tokens"),
        "cache_read_input_tokens": ("cacheReadInputTokens", "cached_tokens"),
        "cache_write_input_tokens": ("cacheWriteInputTokens",),
    }
    normalised = {}
    for name, keys in aliases.items():
        value = next((usage[key] for key in keys if usage.get(key) is not None), None)
        if value is not None:
            normalised[name] = int(value)
    if "total_tokens" not in normalised:
        normalised["total_tokens"] = normalised.get("input_tokens", 0) + normalised.get(
            "output_tokens", 0
        )
    return normalised


def aggregate_token_usage(*record_groups: list[dict[str, Any]]) -> dict[str, int]:
    """Sum response token usage recorded by ranking or filtering workers.

    Parameters
    ----------
    record_groups : list of dict
        Response-detail collections containing a ``usage`` or ``filter_usage`` mapping.

    Returns
    -------
    dict of str to int
        Aggregate input, output, total, and cache token counts.
    """
    totals: dict[str, int] = {}
    for records in record_groups:
        for record in records:
            for usage_key in ("usage", "filter_usage"):
                for name, value in (record.get(usage_key) or {}).items():
                    totals[name] = totals.get(name, 0) + int(value)
    return totals


class RankingClient:
    """Generate short ranking responses through OpenAI-compatible APIs or Bedrock."""

    def __init__(
        self,
        model_name: str,
        *,
        provider: str = "openai",
        base_url: str = "https://api.openai.com/v1",
        region: str | None = None,
        client: Any | None = None,
    ) -> None:
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider {provider!r}; choose from {SUPPORTED_PROVIDERS}"
            )

        self.model_name = model_name
        self.provider = provider
        self.base_url = base_url
        self.region = (
            region
            or os.getenv("BEDROCK_REGION")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or DEFAULT_BEDROCK_REGION
        )
        self._client = client if client is not None else self._create_client()

    def _create_client(self) -> Any:
        if self.provider == "amazon-bedrock":
            import boto3
            from botocore.config import Config

            return boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(
                    read_timeout=600,
                    connect_timeout=30,
                    retries={"max_attempts": 5, "mode": "adaptive"},
                ),
            )

        if self.provider == "azure-openai":
            from openai import OpenAI

            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Please set AZURE_OPENAI_API_KEY.")
            return OpenAI(
                api_key=api_key,
                base_url=self.base_url.rstrip("/") + "/",
                timeout=600,
                max_retries=5,
            )

        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "AAA")
        return OpenAI(
            api_key=api_key,
            base_url=self.base_url.rstrip("/") + "/",
            timeout=600,
            max_retries=5,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        require_complete: bool = False,
        reasoning_effort: str | None = None,
        return_usage: bool = False,
    ) -> str | tuple[str, dict[str, int]]:
        """Generate text, optionally rejecting incomplete passage-filter outputs."""
        if self.provider == "amazon-bedrock":
            # Reasoning-capable Bedrock models can consume a few tokens before
            # emitting the requested label. Keep the original small OpenAI
            # limits while giving Bedrock enough room to produce visible text.
            bedrock_max_tokens = max(
                max_tokens, int(os.getenv("BEDROCK_MAX_TOKENS", "1024"))
            )
            request = {
                "modelId": self.model_name,
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {
                    "maxTokens": bedrock_max_tokens,
                    "temperature": 0,
                },
            }
            if reasoning_effort is not None and self.model_name.lower().startswith(
                "openai.gpt-oss"
            ):
                request["additionalModelRequestFields"] = {
                    "reasoning_effort": reasoning_effort
                }
            response = self._client.converse(**request)
            content = response.get("output", {}).get("message", {}).get("content", [])
            if require_complete and response.get("stopReason") not in (
                "end_turn",
                "stop_sequence",
            ):
                raise RuntimeError(
                    "Filter response did not complete: "
                    + str(response.get("stopReason"))
                )
            text = "\n".join(
                block["text"] for block in content if block.get("text")
            ).strip()
            return (
                (text, _normalise_usage(response.get("usage")))
                if return_usage
                else text
            )

        if self.provider == "azure-openai":
            output_tokens = max(
                max_tokens, int(os.getenv("AZURE_OPENAI_MAX_OUTPUT_TOKENS", "128"))
            )
            response = self._client.responses.create(
                model=self.model_name,
                input=prompt,
                max_output_tokens=output_tokens,
                reasoning={
                    "effort": os.getenv("AZURE_OPENAI_REASONING_EFFORT", "none")
                },
                text={"verbosity": "low"},
            )
            if require_complete and getattr(response, "status", None) != "completed":
                raise RuntimeError("Filter response did not complete")
            text = (response.output_text or "").strip()
            usage = _normalise_usage(
                getattr(response, "usage", None) and vars(response.usage)
            )
            return (text, usage) if return_usage else text

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        message = response.choices[0].message
        if (
            require_complete
            and getattr(response.choices[0], "finish_reason", None) != "stop"
        ):
            raise RuntimeError("Filter response did not complete")
        extra = getattr(message, "model_extra", None) or {}
        content = getattr(message, "content", None)
        if (
            not require_complete
            and content is None
            and extra.get("reasoning_content") is not None
        ):
            content = extra["reasoning_content"]
        text = (content or "").strip()
        usage = _normalise_usage(
            getattr(response, "usage", None) and vars(response.usage)
        )
        return (text, usage) if return_usage else text


def get_ranking_client(
    model_name: str,
    *,
    provider: str = "openai",
    base_url: str = "https://api.openai.com/v1",
    region: str | None = None,
) -> RankingClient:
    """Reuse one transport per joblib worker thread."""
    key = (model_name, provider, base_url, region)
    clients = getattr(_thread_state, "clients", None)
    if clients is None:
        clients = {}
        _thread_state.clients = clients
    if key not in clients:
        clients[key] = RankingClient(
            model_name, provider=provider, base_url=base_url, region=region
        )
    return clients[key]
