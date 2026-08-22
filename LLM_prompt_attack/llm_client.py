"""Shared text-generation clients for ranking attack experiments."""

from __future__ import annotations

import os
import threading
from typing import Any


DEFAULT_BEDROCK_REGION = "ap-southeast-2"
SUPPORTED_PROVIDERS = ("openai", "amazon-bedrock")
_thread_state = threading.local()


class RankingClient:
    """Generate short ranking responses through OpenAI or Bedrock."""

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

        from autogen import OpenAIWrapper

        api_key = os.getenv("OPENAI_API_KEY", "AAA")
        return OpenAIWrapper(
            config_list=[
                {
                    "model": self.model_name,
                    "base_url": self.base_url,
                    "api_key": api_key,
                    "api_type": "openai",
                    "price": [0.05 / 1_000_000, 0.40 / 1_000_000],
                }
            ]
        )

    def generate(self, prompt: str, *, max_tokens: int) -> str:
        if self.provider == "amazon-bedrock":
            # Reasoning-capable Bedrock models can consume a few tokens before
            # emitting the requested label. Keep the original small OpenAI
            # limits while giving Bedrock enough room to produce visible text.
            bedrock_max_tokens = max(
                max_tokens, int(os.getenv("BEDROCK_MAX_TOKENS", "128"))
            )
            response = self._client.converse(
                modelId=self.model_name,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": bedrock_max_tokens, "temperature": 0},
            )
            content = response.get("output", {}).get("message", {}).get("content", [])
            return "\n".join(
                block["text"] for block in content if block.get("text")
            ).strip()

        response = self._client.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        message = response.choices[0].message
        extra = getattr(message, "model_extra", None) or {}
        content = getattr(message, "content", None)
        if content is None and extra.get("reasoning_content") is not None:
            content = extra["reasoning_content"]
        return (content or "").strip()


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
