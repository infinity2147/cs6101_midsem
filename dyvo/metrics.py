"""Evaluation metrics (nDCG@k, R@k, MRR@10) – matches ir_measures / trec_eval definitions."""
import argparse
import json
import math

import numpy as np

from .data import read_qrels, read_run


def _dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def per_query(run, qrels, measures=("nDCG@10", "nDCG@20", "R@100", "R@1000", "RR@10")):
    out = {}
    for qid, rels in qrels.items():
        ranked = [d for d, _ in sorted(run.get(qid, {}).items(), key=lambda x: (-x[1], x[0]))]
        n_rel = sum(1 for r in rels.values() if r > 0)
        res = {}
        for m in measures:
            name, k = m.split("@")
            k = int(k)
            top = ranked[:k]
            if name == "nDCG":
                ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
                idcg = _dcg(ideal)
                res[m] = _dcg([rels.get(d, 0) for d in top]) / idcg if idcg > 0 else 0.0
            elif name == "R":
                res[m] = sum(1 for d in top if rels.get(d, 0) > 0) / max(n_rel, 1)
            elif name == "RR":
                res[m] = next((1 / (i + 1) for i, d in enumerate(top) if rels.get(d, 0) > 0), 0.0)
        out[qid] = res
    return out


def aggregate(pq):
    ms = next(iter(pq.values())).keys()
    return {m: 100 * float(np.mean([v[m] for v in pq.values()])) for m in ms}


def evaluate(run, qrels, **kw):
    return aggregate(per_query(run, qrels, **kw))


def paired_ttest(a, b, measure):
    """two-sided paired t-test between per-query dicts a and b (same qids)."""
    from scipy import stats
    qids = sorted(set(a) & set(b))
    x = np.array([a[q][measure] for q in qids])
    y = np.array([b[q][measure] for q in qids])
    return float(stats.ttest_rel(x, y).pvalue)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--qrels", required=True)
    a = ap.parse_args()
    print(json.dumps(evaluate(read_run(a.run), read_qrels(a.qrels)), indent=1))
