"""Encode corpus + queries into the joint (word-piece ++ entity) sparse space, retrieve
exhaustively (scipy sparse matmul == inverted-index dot product), evaluate.

Outputs in --out:
  reps_docs.npz / reps_queries.npz   sparse matrices (reusable for test-time vocab changes)
  run_<name>.trec, metrics_<name>.json (aggregate, per-query, sparsity/efficiency stats)
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from scipy import sparse
from transformers import AutoTokenizer

from . import WORD_VOCAB_SIZE, bf16_ok
from .data import doc_text, read_jsonl, read_qrels, write_run
from .metrics import aggregate, per_query
from .train import Collator, build_model, build_store


@torch.inference_mode()
def encode(model, coll, items, is_doc, batch_size=32):
    """items: list of (id, text) -> CSR matrix [n x (V + n_ent)]"""
    n_ent = len(coll.store) if coll.store is not None else 0
    order = sorted(range(len(items)), key=lambda i: len(items[i][1]))
    rows, cols, vals = [], [], []
    t0 = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        tids = [items[i][0] for i in idx]
        texts = [items[i][1] for i in idx]
        batch = coll.docs(texts, tids) if is_doc else coll.queries(texts, tids)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16_ok()):
            w, e = (model.encode_d if is_doc else model.encode_q)(batch)
        w = w.float()
        r, c = w.nonzero(as_tuple=True)
        rows.append(np.asarray(idx)[r.numpy()]); cols.append(c.numpy()); vals.append(w[r, c].numpy())
        if e is not None:
            e = e.float()
            r, c = (e > 0).nonzero(as_tuple=True)
            rows.append(np.asarray(idx)[r.numpy()])
            cols.append(WORD_VOCAB_SIZE + batch["ent_ids"][r, c].numpy())
            vals.append(e[r, c].numpy())
        if (b // batch_size) % 100 == 0:
            print(f"  encoded {b + len(idx)}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    m = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(items), WORD_VOCAB_SIZE + n_ent), dtype=np.float32)
    m.sum_duplicates()
    return m


def retrieve(Q, D, doc_ids, qids, k=1000, chunk=256):
    run = {}
    Dt = D.T.tocsr()
    t0 = time.time()
    for s in range(0, Q.shape[0], chunk):
        S = (Q[s:s + chunk] @ Dt).toarray()
        top = np.argpartition(-S, min(k, S.shape[1] - 1), axis=1)[:, :k]
        for i, row in enumerate(top):
            sc = S[i, row]
            o = np.argsort(-sc)
            run[qids[s + i]] = {doc_ids[row[j]]: float(sc[j]) for j in o if sc[j] > 0}
    return run, (time.time() - t0) / Q.shape[0] * 1000


def stats(Q, D):
    def split(M):
        wc = M[:, :WORD_VOCAB_SIZE]
        ec = M[:, WORD_VOCAB_SIZE:]
        return float(np.diff(wc.indptr).mean()), float(np.diff(ec.tocsr().indptr).mean())
    qw, qe = split(Q)
    dw, de = split(D)
    # SPLADE FLOPS: expected #multiplications per (q,d) = sum_j p_q(j) p_d(j)
    pq = np.asarray((Q > 0).mean(0)).ravel()
    pd = np.asarray((D > 0).mean(0)).ravel()
    return {"q_words": qw, "q_ents": qe, "d_words": dw, "d_ents": de,
            "flops": float((pq * pd).sum()),
            "flops_ent": float((pq[WORD_VOCAB_SIZE:] * pd[WORD_VOCAB_SIZE:]).sum()),
            "index_postings": int(D.nnz)}


def mask_entities(M, keep_cols):
    """zero all entity columns not in keep_cols (test-time vocabulary restriction, Ext-B)."""
    M = M.tocoo()
    keep = (M.col < WORD_VOCAB_SIZE) | np.isin(M.col, keep_cols)
    return sparse.csr_matrix((M.data[keep], (M.row[keep], M.col[keep])), shape=M.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reuse", action="store_true", help="reuse saved doc reps if present")
    ap.add_argument("--name", default=None)
    ap.add_argument("--cands", default=None, help="override candidate source at test time")
    ap.add_argument("--k_dense_q", type=int, default=None)
    ap.add_argument("--k_dense_d", type=int, default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    args = argparse.Namespace(**json.load(open(os.path.join(a.model_dir, "args.json"))))
    name = a.name or a.split
    rep_tag = ""
    if a.cands or a.k_dense_q or a.k_dense_d:   # test-time candidate change (noise study)
        args.cands = a.cands or args.cands
        args.k_dense_q = a.k_dense_q or args.k_dense_q
        args.k_dense_d = a.k_dense_d or args.k_dense_d
        rep_tag = f"_{args.cands}_q{args.k_dense_q}_d{args.k_dense_d}"

    tok = AutoTokenizer.from_pretrained(args.backbone)
    train_store = build_store(argparse.Namespace(**json.load(open(os.path.join(a.model_dir, "args.json")))))
    store = build_store(args)
    model = build_model(args, train_store)
    state = torch.load(os.path.join(a.model_dir, "model.pt"))
    model.load_state_dict(state, strict=False)
    if store is not None and train_store is not None and rep_tag:
        # new candidate source at test time: the (frozen) embedding table must cover the new
        # entities -> rebuild the head's buffer on the new compact vocabulary (dynamic vocab!)
        assert args.ent_emb_mode == "frozen", "vocabulary growth needs frozen external embeddings"
        new = store.embeddings(args.ent_emb, w2v_dir=args.w2v, backbone=args.backbone,
                               table=getattr(args, "ent_table", None))
        for enc in (model.q_enc, model.d_enc):
            enc.head.ent_emb = new
    model.eval()
    coll = Collator(tok, store, d_len=args.d_len)

    cpath = os.path.join(args.data, "corpus_test.jsonl")  # evaluation corpus (subset), if present
    corpus = read_jsonl(cpath if os.path.exists(cpath) else os.path.join(args.data, "corpus.jsonl"))
    queries = read_jsonl(os.path.join(args.data, f"queries_{a.split}.jsonl"))
    qrels = read_qrels(os.path.join(args.data, f"qrels_{a.split}.txt"))
    dpath = os.path.join(a.model_dir, f"reps_docs{rep_tag}.npz")
    t0 = time.time()
    if a.reuse and os.path.exists(dpath):
        D = sparse.load_npz(dpath)
    else:
        D = encode(model, coll, [(d["id"], doc_text(d)) for d in corpus], True)
        sparse.save_npz(dpath, D)
    enc_time = time.time() - t0
    Q = encode(model, coll, [(q["id"], q["text"]) for q in queries], False)
    sparse.save_npz(os.path.join(a.model_dir, f"reps_queries_{name}{rep_tag}.npz"), Q)
    run, ms_per_q = retrieve(Q, D, [d["id"] for d in corpus], [q["id"] for q in queries])
    write_run(os.path.join(a.model_dir, f"run_{name}{rep_tag}.trec"), run, tag=os.path.basename(a.model_dir))
    pq = per_query(run, qrels)
    res = {"model": a.model_dir, "split": a.split, "cands": args.cands, "agg": aggregate(pq),
           "stats": stats(Q, D), "ms_per_query": ms_per_q, "doc_encode_s": enc_time, "per_query": pq}
    json.dump(res, open(os.path.join(a.model_dir, f"metrics_{name}{rep_tag}.json"), "w"))
    print(json.dumps({k: res[k] for k in ("agg", "stats", "ms_per_query")}, indent=1))


if __name__ == "__main__":
    main()
