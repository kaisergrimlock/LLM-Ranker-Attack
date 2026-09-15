"""Evaluate Task 2 TREC runs and compute vulnerability deltas."""
from __future__ import annotations
import argparse, csv, math, re
from pathlib import Path
import ir_datasets

def ndcg(ranking, qrels, k=10):
    gains = [qrels.get(doc, 0) for doc in ranking[:k]]
    dcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(gains))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg else None

def load_run(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 6:
            out.setdefault(parts[0], []).append((int(parts[4]), parts[2]))
    return {qid: [doc for _, doc in sorted(rows)] for qid, rows in out.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("LLM_re_ranker/outputs/task2_ndcg_dl19"))
    ap.add_argument("--output", type=Path, default=Path("Results/task2_ndcg_vulnerability.csv"))
    ap.add_argument("--dataset", default="msmarco-passage/trec-dl-2019")
    args = ap.parse_args()
    ds = ir_datasets.load(args.dataset)
    qrels = {}
    for row in ds.qrels_iter():
        qrels.setdefault(str(row.query_id), {})[str(row.doc_id)] = int(row.relevance)
    rows = []
    for path in sorted(args.run_dir.glob("*.txt")):
        m = re.match(r"(.+)\.hits100\.(none|so|sd|qi)\.txt$", path.name)
        if not m: continue
        model, condition = m.groups(); run = load_run(path)
        scores = [ndcg(run[q], qrels[q]) for q in run if q in qrels]
        scores = [x for x in scores if x is not None]
        rows.append({"model": model, "condition": {"none":"clean","so":"doh","sd":"dch","qi":"qi"}[condition], "ndcg@10": sum(scores)/len(scores) if scores else "", "queries": len(scores), "run": str(path)})
    by_model = {(r["model"], r["condition"]): r for r in rows}
    for r in rows:
        if r["condition"] != "clean":
            base = by_model.get((r["model"], "clean"), {}).get("ndcg@10", "")
            if base != "":
                delta = float(base) - float(r["ndcg@10"]); r["delta_ndcg@10"] = delta; r["relative_degradation_%"] = 100 * delta / float(base) if base else ""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model","condition","ndcg@10","delta_ndcg@10","relative_degradation_%","queries","run"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
if __name__ == "__main__": main()
