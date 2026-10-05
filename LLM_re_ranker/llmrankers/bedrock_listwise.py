"""Listwise reranking through Amazon Bedrock Converse."""

from __future__ import annotations

import copy
import os
import re
from typing import Any, List

from prompts import JAILBREAK_PROMPTS
from .rankers import LlmRanker, SearchResult


class BedrockListwiseLlmRanker(LlmRanker):
    CHARACTERS = list("ABCDEFGHIJKLMNOPQRSTUVW")

    def __init__(self, model_name_or_path: str, region=None, window_size=3, step_size=1,
                 num_repeat=1, max_tokens=None, invalid_output_policy="error", client: Any | None = None):
        self.llm = model_name_or_path
        self.region = region or os.getenv("AWS_REGION") or "us-east-1"
        if window_size < 1 or step_size < 1:
            raise ValueError("window_size and step_size must be positive")
        self.window_size = min(window_size, len(self.CHARACTERS))
        if window_size > len(self.CHARACTERS):
            print(
                f"WARNING: listwise window_size={window_size} exceeds the supported "
                f"label range; capping it at {len(self.CHARACTERS)}."
            )
        self.step_size, self.num_repeat = step_size, num_repeat
        self.max_tokens = int(os.getenv("BEDROCK_MAX_TOKENS", "512")) if max_tokens is None else max_tokens
        self.invalid_output_policy = invalid_output_policy
        self.total_compare = self.total_prompt_tokens = self.total_completion_tokens = 0
        self.client = client or self._create_client()

    def _create_client(self):
        import boto3
        from botocore.config import Config
        return boto3.client("bedrock-runtime", region_name=self.region,
                            config=Config(read_timeout=600, connect_timeout=30,
                                          retries={"max_attempts": 5, "mode": "adaptive"}))

    def compare(self, query, docs, attack_prompt="none", attack_position="back"):
        if attack_prompt not in {"none", "so", "sd", "qi"}:
            raise ValueError(f"Unknown attack prompt: {attack_prompt}")
        if attack_prompt == "qi":
            query = f"{query} {JAILBREAK_PROMPTS['qi']}"
        lines = []
        for i, doc in enumerate(docs):
            text = doc.text
            if attack_prompt in {"so", "sd"} and getattr(doc, "gt_rel", None) == 0:
                injection = JAILBREAK_PROMPTS[attack_prompt]
                text = f"{text} {injection}" if attack_position == "back" else f"{injection} {text}"
            lines.append(f'Passage {self.CHARACTERS[i]}: "{text}"')
        labels = self.CHARACTERS[:len(docs)]
        prompt = (f'Given a query "{query}", rank the following passages from most to least relevant.\n\n'
                  + "\n\n".join(lines) + "\n\nOutput only a ranking such as [A, B, C].")
        request = {"modelId": self.llm, "messages": [{"role": "user", "content": [{"text": prompt}]}],
                   "inferenceConfig": {"maxTokens": self.max_tokens, "temperature": 0}}
        if self.llm.lower().startswith("openai.gpt-oss"):
            request["additionalModelRequestFields"] = {"reasoning_effort": os.getenv("GPT_OSS_REASONING_EFFORT", "low")}
        response = self.client.converse(**request)
        usage = response.get("usage", {})
        self.total_prompt_tokens += int(usage.get("inputTokens", 0) or 0)
        self.total_completion_tokens += int(usage.get("outputTokens", 0) or 0)
        raw = "\n".join(block.get("text", "") for block in response.get("output", {}).get("message", {}).get("content", [])).upper()
        match = re.search(r"\[\s*([A-W](?:\s*,\s*[A-W])+)\s*\]", raw)
        if not match:
            match = re.search(r"(?:PASSAGE\s*)?([A-W](?:\s*(?:>|,|THEN)\s*(?:PASSAGE\s*)?[A-W])*)", raw)
        ranking = re.findall(r"[A-W]", match.group(1)) if match else []
        ranking = [x for i, x in enumerate(ranking) if x in labels and x not in ranking[:i]]
        if set(ranking) != set(labels):
            raise RuntimeError(f"Unparseable listwise Bedrock output: {raw!r}")
        return ranking

    def rerank(self, query, ranking: List[SearchResult], attack_prompt="none", attack_position="back"):
        self.total_compare = self.total_prompt_tokens = self.total_completion_tokens = 0
        ranking = copy.deepcopy(ranking)
        for _ in range(self.num_repeat):
            end = len(ranking)
            start = max(0, end - self.window_size)
            while start >= 0:
                window = ranking[start:end]
                labels = self.compare(query, window, attack_prompt, attack_position)
                ranking[start:end] = [window[ord(label) - 65] for label in labels]
                end -= self.step_size
                start = max(0, end - self.window_size)
        return [SearchResult(docid=doc.docid, score=-rank, text=None) for rank, doc in enumerate(ranking)]

    def truncate(self, text, length):
        return " ".join(text.split()[:length])
