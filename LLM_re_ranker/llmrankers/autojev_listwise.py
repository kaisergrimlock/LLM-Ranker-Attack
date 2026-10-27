"""Listwise reranking through the Jev/AutoJev systemone HTTP API.

The API exposes choice probabilities rather than a permutation endpoint.  We
therefore submit each sliding window as one multi-choice question and order
the window by the returned probability that each passage is the best choice.
This keeps the repository's listwise window/step protocol while avoiding any
local model dependency.
"""

import copy
import json
from typing import List
from urllib.request import Request, urlopen

from .rankers import LlmRanker, SearchResult


class AutoJevListwiseRanker(LlmRanker):
    def __init__(self, endpoint="http://127.0.0.1:8000/v1/systemone",
                 model="autojev", window_size=3, step_size=1,
                 num_repeat=1, timeout=180):
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if step_size < 1:
            raise ValueError("step_size must be at least 1")
        if num_repeat < 1:
            raise ValueError("num_repeat must be at least 1")
        self.endpoint = endpoint
        self.model = model
        self.window_size = window_size
        self.step_size = step_size
        self.num_repeat = num_repeat
        self.timeout = timeout
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def truncate(self, text, length):
        return " ".join(str(text).split()[:length])

    def _rank_window(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        criteria = {
            str(index): (
                f"Passage {index + 1} is the single most relevant passage and "
                "best answers the query."
            )
            for index in range(len(documents))
        }
        passages = "\n\n".join(
            f"Passage {index + 1}: {document.text}"
            for index, document in enumerate(documents)
        )
        body = {
            "model": self.model,
            "state": f"Query: {query}\n\n{passages}",
            "questions": {
                "ranking": {
                    "type": "choice",
                    "instructions": (
                        "Compare all passages together. Select the passage that "
                        "should appear first in a relevance ranking."
                    ),
                    "criteria": criteria,
                }
            },
        }
        request = Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        answer = payload["answers"]["ranking"]
        probabilities = answer.get("probabilities", {})
        if not probabilities:
            raise ValueError("AutoJev listwise response did not contain probabilities")
        scores = [float(probabilities.get(str(index), 0.0))
                  for index in range(len(documents))]
        self.total_compare += 1
        order = sorted(range(len(documents)), key=lambda index: scores[index], reverse=True)
        for index, score in enumerate(scores):
            documents[index].score = score
        return [documents[index] for index in order]

    def rerank(self, query: str, ranking: List[SearchResult], **_kwargs) -> List[SearchResult]:
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        current = copy.deepcopy(ranking)

        for _ in range(self.num_repeat):
            end = len(current)
            while end > 0:
                start = max(0, end - self.window_size)
                if end - start > 1:
                    current[start:end] = self._rank_window(query, current[start:end])
                if start == 0:
                    break
                end -= self.step_size

        for index, document in enumerate(current):
            document.score = -index
        return current
