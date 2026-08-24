"""GPU-free setwise reranking through Amazon Bedrock."""

from __future__ import annotations

import copy
import os
import re
from typing import Any, List

from prompts import JAILBREAK_PROMPTS

from .rankers import LlmRanker, SearchResult


class BedrockSetwiseLlmRanker(LlmRanker):
    CHARACTERS = list("ABCDEFGHIJKLMNOPQRSTUVW")

    def __init__(
        self,
        model_name_or_path: str,
        region: str | None = None,
        num_child: int = 3,
        method: str = "heapsort",
        k: int = 10,
        client: Any | None = None,
    ) -> None:
        self.llm = model_name_or_path
        self.region = (
            region
            or os.getenv("BEDROCK_REGION")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "ap-southeast-2"
        )
        self.num_child = num_child
        self.method = method
        self.k = k
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.system_prompt = (
            "You are RankGPT, an intelligent assistant specialized in selecting "
            "the most relevant passage from a pool of passages based on its "
            "relevance to the query."
        )
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

    def compare(
        self,
        query: str,
        docs: List[SearchResult],
        attack_prompt: str = "none",
        attack_position: str = "back",
    ) -> str:
        if attack_prompt not in ("none", "so", "sd"):
            raise ValueError(f"Unknown attack prompt: {attack_prompt}")
        if attack_position not in ("front", "back"):
            raise ValueError(f"Unknown attack position: {attack_position}")

        self.total_compare += 1
        passage_lines = []
        for i, doc in enumerate(docs):
            text = doc.text
            if getattr(doc, "gt_rel", None) == 0 and attack_prompt != "none":
                injection = JAILBREAK_PROMPTS[attack_prompt]
                text = (
                    f"{injection} {text}"
                    if attack_position == "front"
                    else f"{text} {injection}"
                )
            passage_lines.append(f'Passage {self.CHARACTERS[i]}: "{text}"')

        allowed = self.CHARACTERS[: len(docs)]
        prompt = (
            f'Given a query "{query}", which of the following passages is the '
            "most relevant one to the query?\n\n"
            + "\n\n".join(passage_lines)
            + "\n\nChoose exactly one passage. Output exactly one label from "
            + ", ".join(allowed)
            + ". Do not list alternatives or add an explanation."
        )
        max_tokens = int(os.getenv("BEDROCK_MAX_TOKENS", "32"))

        for attempt in range(1, 4):
            response = self.client.converse(
                modelId=self.llm,
                system=[{"text": self.system_prompt}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": 0,
                    "stopSequences": ["\n"],
                },
            )
            usage = response.get("usage", {})
            self.total_prompt_tokens += int(usage.get("inputTokens", 0) or 0)
            self.total_completion_tokens += int(usage.get("outputTokens", 0) or 0)
            raw_output = "\n".join(
                block["text"]
                for block in response.get("output", {})
                .get("message", {})
                .get("content", [])
                if block.get("text")
            ).strip()

            passage_match = re.search(r"\bPassage\s+([A-W])\b", raw_output, re.I)
            if passage_match and passage_match.group(1).upper() in allowed:
                return passage_match.group(1).upper()
            label = raw_output.upper()
            if label in allowed:
                return label
            first_line = next(
                (line.strip().upper() for line in raw_output.splitlines() if line.strip()),
                "",
            )
            first_line_match = re.fullmatch(r"(?:PASSAGE\s+)?([A-W])[.):]?", first_line)
            if first_line_match and first_line_match.group(1) in allowed:
                print(
                    "Bedrock returned extra lines; using its first selected label "
                    f"{first_line_match.group(1)!r}."
                )
                return first_line_match.group(1)
            print(f"Unexpected Bedrock output on attempt {attempt}/3: {raw_output!r}")

        raise RuntimeError(
            f"Bedrock model {self.llm!r} did not return one of {allowed} after 3 attempts"
        )

    def truncate(self, text: str, length: int) -> str:
        # Bedrock does not expose model tokenizers. This deterministic word limit
        # keeps this backend GPU-free and makes the approximation explicit.
        return " ".join(text.split()[:length])

    def heapify(self, arr, n, i, query, attack_prompt="none", attack_position="back"):
        if self.num_child * i + 1 < n:
            end = min(self.num_child * (i + 1) + 1, n)
            inds = [i] + list(range(self.num_child * i + 1, end))
            docs = [arr[index] for index in inds]
            output = self.compare(query, docs, attack_prompt, attack_position)
            best_ind = self.CHARACTERS.index(output)
            largest = inds[best_ind] if best_ind < len(inds) else i
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.heapify(arr, n, largest, query, attack_prompt, attack_position)

    def heap_sort(self, arr, query, k, attack_prompt="none", attack_position="back"):
        n = len(arr)
        for i in range(n // self.num_child, -1, -1):
            self.heapify(arr, n, i, query, attack_prompt, attack_position)
        for ranked, i in enumerate(range(n - 1, 0, -1), start=1):
            arr[i], arr[0] = arr[0], arr[i]
            if ranked == k:
                break
            self.heapify(arr, i, 0, query, attack_prompt, attack_position)

    def rerank(
        self,
        query: str,
        ranking: List[SearchResult],
        attack_prompt: str = "none",
        attack_position: str = "back",
    ) -> List[SearchResult]:
        if self.method != "heapsort":
            raise NotImplementedError(
                "The Bedrock backend currently supports setwise heapsort only."
            )

        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.heap_sort(ranking, query, self.k, attack_prompt, attack_position)
        ranking = list(reversed(ranking))

        results = []
        top_doc_ids = set()
        for rank, doc in enumerate(ranking[: self.k], start=1):
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
        next_rank = len(results) + 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-next_rank, text=None))
                next_rank += 1
        return results
