"""Paired t-tests behind every p-value quoted in the report (per-query nDCG@10).

  python -m dyvo.significance --runs RUNS --work WORK [--out report/results/significance.json]
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

from .analysis import seen_entities
from .data import read_jsonl, read_qrels
from .evaluate import mask_entities, retrieve
from .metrics import per_query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="report/results/significance.json")
    a = ap.parse_args()
    R, W = a.runs, a.work
    L = lambda n, t="test": json.load(open(f"{R}/{n}/metrics_{t}.json"))["per_query"]
    lsr, dyvo, gate, noisy, learned = (L("lsr_w"), L("dyvo_link_w2v"), L("dyvo_linkdense_gate"),
                                       L("dyvo_linkdense_w2v"), L("dyvo_link_learned"))
    qrels = read_qrels(f"{W}/qrels_test.txt")
    link = {r["id"]: set(r["entities"]) for r in read_jsonl(f"{W}/queries_test.ent_link.jsonl")}
    dl = {r["id"]: set(r["entities"]) for r in read_jsonl(f"{W}/corpus.ent_link.jsonl")}
    seen = seen_entities(W, "link", f"{R}/dyvo_link_w2v")
    allq = list(qrels)
    unseen = [q for q in allq if link[q] and any(e not in seen for e in link[q])]
    consistent = [q for q in allq if link[q] & dl[next(iter(qrels[q]))]]
    inconsistent = [q for q in allq if link[q] and not link[q] & dl[next(iter(qrels[q]))]]

    # static-vocabulary re-scoring of the trained DyVo model (Ext-B), from the saved sparse reps
    from scipy import sparse
    from . import WORD_VOCAB_SIZE
    from .entity_store import EntityStore
    st = EntityStore(W, ["corpus", "queries_train10k", "queries_test", "queries_val"], source="link")
    keep = np.array([WORD_VOCAB_SIZE + st.row2id[e] for e in seen if e in st.row2id])
    D = sparse.load_npz(f"{R}/dyvo_link_w2v/reps_docs.npz")
    Q = sparse.load_npz(f"{R}/dyvo_link_w2v/reps_queries_test.npz")
    corpus = [x["id"] for x in read_jsonl(f"{W}/corpus_test.jsonl")]
    qids = [x["id"] for x in read_jsonl(f"{W}/queries_test.jsonl")]
    static = per_query(retrieve(mask_entities(Q, keep), mask_entities(D, keep), corpus, qids)[0], qrels)

    def t(x, y, qs, name):
        xs = np.array([x[q]["nDCG@10"] for q in qs])
        ys = np.array([y[q]["nDCG@10"] for q in qs])
        r = {"comparison": name, "n": len(qs), "delta_nDCG@10": round(100 * (ys.mean() - xs.mean()), 2),
             "p": float(stats.ttest_rel(ys, xs).pvalue)}
        print(f"{name:62s} n={r['n']:4d}  Δ={r['delta_nDCG@10']:+6.2f}  p={r['p']:.4f}")
        return r

    out = [t(lsr, dyvo, allq, "DyVo(link) vs LSR-w, all queries"),
           t(lsr, dyvo, consistent, "DyVo(link) vs LSR-w, consistent linking"),
           t(lsr, dyvo, inconsistent, "DyVo(link) vs LSR-w, inconsistent linking"),
           t(lsr, gate, inconsistent, "DyVo-Gate vs LSR-w, inconsistent linking"),
           t(lsr, noisy, allq, "DyVo(link+dense) vs LSR-w, all"),
           t(dyvo, noisy, allq, "DyVo(link+dense) vs DyVo(link), all"),
           t(noisy, gate, allq, "DyVo-Gate vs DyVo(link+dense), all"),
           t(lsr, gate, allq, "DyVo-Gate vs LSR-w, all"),
           t(lsr, dyvo, unseen, "DyVo(link) vs LSR-w, >=1 unseen entity"),
           t(static, dyvo, unseen, "dynamic vs static vocabulary, >=1 unseen entity"),
           t(learned, dyvo, unseen, "frozen W2V vs learned table, >=1 unseen entity"),
           t(learned, dyvo, allq, "frozen W2V vs learned table, all"),
           t(lsr, learned, allq, "learned table vs LSR-w, all")]
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
