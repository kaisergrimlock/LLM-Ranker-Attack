import sys
import tempfile
import unittest
from pathlib import Path


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION_ROOT))

from ndcg_at_10 import build_report, ndcg_at_k, read_trec_run


class NdcgAtTenTests(unittest.TestCase):
    def test_perfect_ranking_scores_one(self):
        qrels = {"d1": 3, "d2": 2, "d3": 0}
        self.assertAlmostEqual(1.0, ndcg_at_k(["d1", "d2", "d3"], qrels, 10))

    def test_attack_report_includes_delta_and_zero_relevance_movement(self):
        qrels = {"q1": {"relevant": 3, "zero": 0}}
        clean = {"q1": ["relevant", "zero"]}
        attacked = {"q1": ["zero", "relevant"]}

        report = build_report(clean, qrels, 10, attacked)

        self.assertLess(report["attacked"]["mean_ndcg"], report["clean"]["mean_ndcg"])
        self.assertLess(report["comparison"]["mean_delta"], 0)
        self.assertEqual(
            {"promoted": 1, "demoted": 0, "unchanged": 0},
            report["comparison"]["zero_relevance_movement"],
        )

    def test_unjudged_run_queries_are_reported_and_ignored(self):
        report = build_report(
            {"judged": ["d1"], "unjudged": ["d2"]},
            {"judged": {"d1": 1}},
            10,
        )
        self.assertEqual(1, report["clean"]["judged_query_count"])
        self.assertEqual(1, report["clean"]["unjudged_run_query_count"])
        self.assertAlmostEqual(1.0, report["clean"]["mean_ndcg"])

    def test_comparison_rejects_different_query_sets(self):
        with self.assertRaisesRegex(ValueError, "identical query IDs"):
            build_report(
                {"q1": ["d1"]},
                {"q1": {"d1": 1}, "q2": {"d2": 1}},
                10,
                {"q2": ["d2"]},
            )

    def test_trec_run_is_sorted_by_declared_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.txt"
            path.write_text(
                "q1 Q0 d2 2 -2 tag\nq1 Q0 d1 1 -1 tag\n", encoding="utf-8"
            )
            self.assertEqual({"q1": ["d1", "d2"]}, read_trec_run(path))


if __name__ == "__main__":
    unittest.main()
