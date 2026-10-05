"""Pointwise reranking from Bedrock OpenAI-compatible token log probabilities."""

from __future__ import annotations

import copy
import json
import math
import os
import re
from typing import Any, List

from prompts import JAILBREAK_PROMPTS

from .rankers import LlmRanker, SearchResult


class MissingLogprobsError(RuntimeError):
    """Raised when a Bedrock model does not return usable Yes/No log probabilities."""


class BedrockPointwiseLlmRanker(LlmRanker):
    """Rank passages by the Bedrock model's next-token Yes/No distribution.

    Bedrock's Converse API deliberately does not expose raw vocabulary logits.
    This ranker instead uses the OpenAI-compatible invocation shape and requires
    the provider to return both labels in ``top_logprobs`` for the first token.
    """

    def __init__(
        self,
        model_name_or_path: str,
        region: str | None = None,
        max_tokens: int | None = None,
        top_logprobs: int = 20,
        client: Any | None = None,
    ) -> None:
        if top_logprobs < 2:
            raise ValueError("top_logprobs must be at least 2")
        self.llm = model_name_or_path
        self.region = (
            region
            or os.getenv("BEDROCK_REGION")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "ap-southeast-2"
        )
        self.max_tokens = 1 if max_tokens is None else max_tokens
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self.top_logprobs = top_logprobs
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_label_logprobs: dict[str, float] | None = None
        self.last_document_scores: dict[str, dict[str, float]] = {}
        self._warned_generation_fallback = False
        self.client = client or self._create_client()

    def _create_client(self):
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

    @staticmethod
    def _body_json(response: dict[str, Any]) -> dict[str, Any]:
        body = response["body"]
        raw = body.read() if hasattr(body, "read") else body
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    @staticmethod
    def _label_logprobs(response: dict[str, Any]) -> tuple[float, float]:
        try:
            content = response["choices"][0]["logprobs"]["content"]
            first_token = content[0]
            candidates = [first_token, *first_token.get("top_logprobs", [])]
        except (IndexError, KeyError, TypeError) as exc:
            raise MissingLogprobsError(
                "Bedrock did not return OpenAI-compatible token log probabilities. "
                "Use a model/endpoint that supports logprobs, or use setwise ranking."
            ) from exc

        values: dict[str, float] = {}
        for candidate in candidates:
            token = str(candidate.get("token", "")).strip().casefold()
            if token in {"yes", "no"} and candidate.get("logprob") is not None:
                values[token] = float(candidate["logprob"])
        if set(values) != {"yes", "no"}:
            raise MissingLogprobsError(
                "Bedrock returned logprobs, but its first-token top_logprobs did not "
                "contain both 'Yes' and 'No'. Increase --bedrock_top_logprobs or use "
                "a model that exposes both labels."
            )
        return values["yes"], values["no"]

    def _score(self, query: str, passage: str) -> float:
        return self._score_with_attack(query, passage, "none", "back", None)

    @staticmethod
    def _generated_text(payload: dict[str, Any]) -> str:
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        content = message.get("content", choice.get("text", ""))
        if isinstance(content, list):
            return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return str(content or "")

    def _score_with_attack(self, query: str, passage: str, attack_prompt: str,
                           attack_position: str, gt_rel: int | None) -> float:
        if attack_prompt not in {"none", "so", "sd", "qi"}:
            raise ValueError(f"Unknown attack prompt: {attack_prompt}")
        if attack_prompt == "qi":
            query = f"{query} {JAILBREAK_PROMPTS['qi']}"
        if attack_prompt in {"so", "sd"} and gt_rel == 0:
            injection = JAILBREAK_PROMPTS[attack_prompt]
            passage = (f"{passage} {injection}" if attack_position == "back"
                       else f"{injection} {passage}")
        prompt = (
            "Passage: " + passage + "\nQuery: " + query + "\n"
            "Does the passage answer the query? Reply with exactly Yes or No."
        )
        response = self.client.invoke_model(
            modelId=self.llm,
            body=json.dumps(
                {
                    "model": self.llm,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                    "temperature": 0,
                    "logprobs": True,
                    "top_logprobs": self.top_logprobs,
                }
            ),
            contentType="application/json",
            accept="application/json",
        )
        payload = self._body_json(response)
        usage = payload.get("usage", {})
        self.total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        try:
            yes, no = self._label_logprobs(payload)
            self.last_label_logprobs = {"Yes": yes, "No": no}
            return 1 / (1 + math.exp(no - yes))
        except MissingLogprobsError:
            # GPT-OSS may return a valid answer but omit the counterfactual
            # Yes/No token from top_logprobs. Preserve the run by using the
            # generated binary answer as a documented fallback score.
            generated = self._generated_text(payload).casefold()
            labels = re.findall(r"\b(?:yes|no)\b", generated)
            if not labels:
                raise
            if not self._warned_generation_fallback:
                print(
                    "WARNING: Bedrock did not expose both Yes/No token logprobs; "
                    "using generated Yes/No scores for pointwise ranking."
                )
                self._warned_generation_fallback = True
            self.last_label_logprobs = None
            return 1.0 if labels[-1] == "yes" else 0.0

    def rerank(
        self,
        query: str,
        ranking: List[SearchResult],
        attack_prompt: str = "none",
        attack_position: str = "back",
    ) -> List[SearchResult]:
        if attack_position not in ("front", "back"):
            raise ValueError(f"Unknown attack position: {attack_position}")
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_document_scores = {}
        scored = copy.deepcopy(ranking)
        for document in scored:
            self.total_compare += 1
            document.score = self._score_with_attack(
                query, document.text, attack_prompt, attack_position,
                getattr(document, "gt_rel", None),
            )
            label_logprobs = self.last_label_logprobs or {}
            yes_logprob = label_logprobs.get("Yes")
            no_logprob = label_logprobs.get("No")
            if yes_logprob is not None and no_logprob is not None:
                self.last_document_scores[document.docid] = {
                    "yes_logprob": yes_logprob,
                    "no_logprob": no_logprob,
                    "logit_margin": yes_logprob - no_logprob,
                    "score": document.score,
                }
            document.text = None
        return sorted(scored, key=lambda document: document.score, reverse=True)

    def truncate(self, text: str, length: int) -> str:
        return " ".join(text.split()[:length])
