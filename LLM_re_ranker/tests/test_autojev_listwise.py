import json
import unittest
from unittest.mock import patch

from llmrankers.autojev_listwise import AutoJevListwiseRanker
from llmrankers.rankers import SearchResult


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class AutoJevListwiseTests(unittest.TestCase):
    def test_window_probabilities_define_order(self):
        payload = {
            "answers": {
                "ranking": {
                    "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2}
                }
            }
        }
        ranker = AutoJevListwiseRanker(window_size=3)
        documents = [SearchResult(str(i), 0.0, f"passage {i}") for i in range(3)]
        with patch("llmrankers.autojev_listwise.urlopen", return_value=FakeResponse(payload)):
            result = ranker._rank_window("query", documents)
        self.assertEqual([document.docid for document in result], ["1", "2", "0"])
        self.assertEqual(ranker.total_compare, 1)

    def test_sliding_windows_make_expected_number_of_requests(self):
        payload = {
            "answers": {
                "ranking": {
                    "probabilities": {"0": 0.5, "1": 0.3, "2": 0.2}
                }
            }
        }
        ranker = AutoJevListwiseRanker(window_size=3, step_size=1)
        documents = [SearchResult(str(i), 0.0, f"passage {i}") for i in range(5)]
        with patch("llmrankers.autojev_listwise.urlopen", return_value=FakeResponse(payload)) as request:
            result = ranker.rerank("query", documents)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(len(result), 5)
        self.assertEqual([document.score for document in result], [0, -1, -2, -3, -4])


if __name__ == "__main__":
    unittest.main()
