"""Data-preparation steps that were run once for the mid-sem experiments.

  python -m dyvo.prep stage0 --work W       # queries_s0.jsonl (train questions 10001-14000) +
                                            # BM25 hard-negative triples with hard labels (stage 0)
  python -m dyvo.prep eval_corpus --work W  # corpus_test.jsonl: all 2,067 dev paragraphs + 6,000
                                            # random train paragraphs (seed 7) = 8,067 docs
"""
import argparse
import json
import os
import random

from .bm25 import BM25
from .data import read_jsonl, read_qrels, write_jsonl, write_run
from .teacher import build_triples


def stage0(work):
    train = read_jsonl(os.path.join(work, "queries_train.jsonl"))
    qs = train[10000:14000]                      # disjoint from queries_train10k (the first 10,000)
    write_jsonl(os.path.join(work, "queries_s0.jsonl"), qs)
    bm25 = BM25(read_jsonl(os.path.join(work, "corpus.jsonl")))
    run = bm25.search(qs, k=50)
    write_run(os.path.join(work, "run_bm25_s0.trec"), run, tag="bm25")
    triples = build_triples(qs, read_qrels(os.path.join(work, "qrels_train.txt")), run, seed=1)
    with open(os.path.join(work, "train_triples_s0_hard.jsonl"), "w") as f:
        for q, p, n in triples:   # hard labels: softmax([10, 0]) ~ [1, 0] -> KL == pairwise CE
            f.write(json.dumps({"qid": q, "pos": p, "neg": n, "pos_score": 10.0, "neg_score": 0.0}) + "\n")
    print(f"stage-0: {len(qs)} queries, {len(triples)} triples")


def eval_corpus(work, n_distractors=6000, seed=7):
    corpus = read_jsonl(os.path.join(work, "corpus.jsonl"))
    dev = [d for d in corpus if d["id"].startswith("d")]
    tr = [d for d in corpus if d["id"].startswith("t")]
    random.Random(seed).shuffle(tr)
    sub = dev + tr[:n_distractors]
    write_jsonl(os.path.join(work, "corpus_test.jsonl"), sub)
    bm25 = BM25(sub)
    write_run(os.path.join(work, "run_bm25_test.trec"),
              bm25.search(read_jsonl(os.path.join(work, "queries_test.jsonl"))), tag="bm25")
    print(f"eval corpus: {len(dev)} dev + {n_distractors} distractors = {len(sub)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage0", "eval_corpus"])
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    stage0(a.work) if a.cmd == "stage0" else eval_corpus(a.work)
