"""BM25 over a scipy sparse term-document matrix (baseline + hard-negative mining)."""
import argparse
import re
from collections import Counter

import numpy as np
from scipy import sparse

from .data import doc_text, read_jsonl, write_run

_STOP = set("""a an and are as at be but by for from has have he her his i if in into is it its
of on or she that the their them they this to was were what when where which who whom why
will with how did do does done can could would should than then there these those been being
""".split())
_TOK = re.compile(r"[a-z0-9]+")


def analyze(text):
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs, k1=0.9, b=0.4):  # Anserini defaults
        self.doc_ids = [d["id"] for d in docs]
        self.vocab = {}
        rows, cols, vals = [], [], []
        lens = np.zeros(len(docs))
        for j, d in enumerate(docs):
            tf = Counter(analyze(doc_text(d)))
            lens[j] = sum(tf.values())
            for t, c in tf.items():
                rows.append(self.vocab.setdefault(t, len(self.vocab)))
                cols.append(j)
                vals.append(c)
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(len(self.vocab), len(docs)), dtype=np.float32)
        df = np.diff(tf.indptr)
        n = len(docs)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1 - b + b * lens / lens.mean())
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.col])
        self.mat = sparse.csr_matrix((w * self.idf[tf.row], (tf.row, tf.col)), shape=tf.shape)

    def search(self, queries, k=1000):
        run = {}
        for q in queries:
            ids = [self.vocab[t] for t in analyze(q["text"]) if t in self.vocab]
            if not ids:
                run[q["id"]] = {}
                continue
            scores = np.asarray(self.mat[ids].sum(0)).ravel()
            top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
            top = top[np.argsort(-scores[top])]
            run[q["id"]] = {self.doc_ids[i]: float(scores[i]) for i in top if scores[i] > 0}
        return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1000)
    a = ap.parse_args()
    bm25 = BM25(read_jsonl(a.corpus))
    write_run(a.out, bm25.search(read_jsonl(a.queries), a.k), tag="bm25")
