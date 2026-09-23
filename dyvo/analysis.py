"""Collect results into the report tables + figures.

* Table R1  (paper Table 1 analogue): BM25 / LSR-w / DyVo(linked)            nDCG@10/20, R@100/1k, MRR
* Table R2  (paper Table 2 analogue): candidate source  (link / link+dense)
* Table R3  (paper Table 3 analogue): entity embeddings (Wikipedia2Vec / Token Aggr.)
* Ext-A     noise robustness: ungated vs gated, at train-time noise and 3x test-time noise
* Ext-B     dynamic vocabulary: seen vs unseen-entity queries; frozen external embeddings vs a
            static learned table; test-time restriction of the vocabulary to training entities
* Candidate quality: recall of the gold article entity among a query's candidates
Paired two-sided t-tests vs LSR-w (per-query nDCG@10).
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import sparse

from . import WORD_VOCAB_SIZE
from .data import read_jsonl, read_qrels
from .entity_store import EntityStore
from .evaluate import mask_entities, retrieve
from .metrics import aggregate, paired_ttest, per_query

M = ["nDCG@10", "nDCG@20", "R@100", "R@1000", "RR@10"]


def load(runs, name, tag="test"):
    p = os.path.join(runs, name, f"metrics_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def row(label, res, base=None):
    a = res["agg"]
    cells = [f"{a[m]:.2f}" for m in M]
    if base is not None:
        p = paired_ttest(res["per_query"], base["per_query"], "nDCG@10")
        cells[0] += "†" if p < 0.05 and a["nDCG@10"] > base["agg"]["nDCG@10"] else ""
        cells[0] += "‡" if p < 0.05 and a["nDCG@10"] < base["agg"]["nDCG@10"] else ""
    s = res.get("stats", {})
    eff = f"{s.get('d_words', 0):.0f} / {s.get('d_ents', 0):.1f} | {s.get('q_words', 0):.1f} / {s.get('q_ents', 0):.1f} | {s.get('flops', 0):.2f}" if s else "–"
    return f"| {label} | " + " | ".join(cells) + f" | {eff} |"


HDR = ("| Model | nDCG@10 | nDCG@20 | R@100 | R@1k | MRR@10 | doc words/ents \\| q words/ents \\| FLOPs |\n"
       "|---|---|---|---|---|---|---|")


def bm25_res(work):
    from .data import read_run
    run = read_run(os.path.join(work, "run_bm25_test.trec"))
    pq = per_query(run, read_qrels(os.path.join(work, "qrels_test.txt")))
    return {"agg": aggregate(pq), "per_query": pq}


def training_triples(model_dir):
    """exactly the triples a run was trained on (same selection as dyvo.train)."""
    import random
    args = json.load(open(os.path.join(model_dir, "args.json")))
    triples = read_jsonl(args["triples"])
    triples.sort(key=lambda t: t["qid"])
    random.Random(0).shuffle(triples)
    qids = {q["id"] for q in read_jsonl(os.path.join(args["data"], f"{args['train_queries']}.jsonl"))}
    return [t for t in triples if t["qid"] in qids][args["triple_start"]:args["triple_end"]]


def seen_entities(work, source, model_dir):
    """entity rows appearing in any *training* text's candidates (train queries + their docs)."""
    st = EntityStore(work, ["corpus", "queries_train10k"], source=source)
    triples = training_triples(model_dir)
    train_docs = {t["pos"] for t in triples} | {t["neg"] for t in triples}
    train_q = {t["qid"] for t in triples}
    seen = set()
    for tid, c in st.cands.items():
        if tid in train_docs or tid in train_q:   # doc ids "t12_3"/"d4_5", query ids are hex
            seen.update(e for e, _ in c)
    return seen


def ext_b(runs, work, out):
    lines = []
    link = {r["id"]: r for r in read_jsonl(os.path.join(work, "queries_test.ent_link.jsonl"))}
    seen = seen_entities(work, "link", os.path.join(runs, "dyvo_link_w2v"))
    groups = {"no entity linked": [], "all entities seen in training": [], ">=1 unseen entity": []}
    for qid, r in link.items():
        ents = r["entities"]
        if not ents:
            groups["no entity linked"].append(qid)
        elif all(e in seen for e in ents):
            groups["all entities seen in training"].append(qid)
        else:
            groups[">=1 unseen entity"].append(qid)
    models = [("LSR-w", "lsr_w"), ("DyVo (frozen W2V, dynamic)", "dyvo_link_w2v"),
              ("DyVo (static learned table)", "dyvo_link_learned")]
    lines.append("| Query group | #q | " + " | ".join(m for m, _ in models) + " |")
    lines.append("|---|---|" + "---|" * len(models))
    res = {k: load(runs, v) for _, v in models for k in [v]}
    for g, qids in groups.items():
        cells = []
        for _, v in models:
            r = res[v]
            cells.append(f"{100 * np.mean([r['per_query'][q]['nDCG@10'] for q in qids]):.2f}" if r and qids else "–")
        lines.append(f"| {g} | {len(qids)} | " + " | ".join(cells) + " |")
    # test-time vocabulary restriction on the frozen model (no re-encoding needed)
    d = os.path.join(runs, "dyvo_link_w2v")
    if os.path.exists(os.path.join(d, "reps_docs.npz")):
        st = EntityStore(work, ["corpus", "queries_train10k", "queries_test", "queries_val"], source="link")
        keep = np.array([WORD_VOCAB_SIZE + st.row2id[e] for e in seen if e in st.row2id])
        D = sparse.load_npz(os.path.join(d, "reps_docs.npz"))
        Q = sparse.load_npz(os.path.join(d, "reps_queries_test.npz"))
        cp = os.path.join(work, "corpus_test.jsonl")
        corpus = [x["id"] for x in read_jsonl(cp if os.path.exists(cp) else os.path.join(work, "corpus.jsonl"))]
        qids = [x["id"] for x in read_jsonl(os.path.join(work, "queries_test.jsonl"))]
        qrels = read_qrels(os.path.join(work, "qrels_test.txt"))
        run, _ = retrieve(mask_entities(Q, keep), mask_entities(D, keep), corpus, qids)
        pq = per_query(run, qrels)
        full = res["dyvo_link_w2v"]["per_query"]
        lines.append("")
        lines.append("Test-time vocabulary restriction of the same DyVo model (entities never seen in "
                     "training removed from the index = a static vocabulary):")
        lines.append("")
        lines.append("| Vocabulary | all queries nDCG@10 | >=1-unseen-entity queries nDCG@10 |")
        lines.append("|---|---|---|")
        uq = groups[">=1 unseen entity"]
        for lab, p in [("dynamic (all Wikipedia entities)", full), ("static (training entities only)", pq)]:
            lines.append(f"| {lab} | {100 * np.mean([p[q]['nDCG@10'] for q in p]):.2f} | "
                         f"{100 * np.mean([p[q]['nDCG@10'] for q in uq]):.2f} |")
        lines.append(f"\n{len(seen)} distinct entities seen in training; "
                     f"{sum(1 for e in {x for r in link.values() for x in r['entities']} if e not in seen)} "
                     f"distinct test-query entities unseen.")
    return "\n".join(lines)


def consistency(runs, work):
    """Linking-consistency breakdown: DyVo can only help when the query's linked entity is also a
    candidate of the relevant document; inconsistent linking actively hurts."""
    ql = {r["id"]: set(r["entities"]) for r in read_jsonl(os.path.join(work, "queries_test.ent_link.jsonl"))}
    dl = {r["id"]: set(r["entities"]) for r in read_jsonl(os.path.join(work, "corpus.ent_link.jsonl"))}
    qrels = read_qrels(os.path.join(work, "qrels_test.txt"))
    groups = {"no linked query entity": [], "query entity linked, absent from gold doc": [],
              "query entity also linked in gold doc": []}
    for q, rel in qrels.items():
        g = next(iter(rel))
        k = ("no linked query entity" if not ql[q] else
             "query entity also linked in gold doc" if ql[q] & dl.get(g, set()) else
             "query entity linked, absent from gold doc")
        groups[k].append(q)
    models = [("LSR-w", "lsr_w"), ("DyVo (link)", "dyvo_link_w2v"), ("DyVo (link+dense)", "dyvo_linkdense_w2v"),
              ("DyVo-Gate", "dyvo_linkdense_gate")]
    models = [(l, n, load(runs, n)) for l, n in models if load(runs, n)]
    lines = ["| Query group | #q | " + " | ".join(l for l, _, _ in models) + " |",
             "|---|---|" + "---|" * len(models)]
    for g, qs in groups.items():
        cells = [f"{100 * np.mean([r['per_query'][q]['nDCG@10'] for q in qs]):.2f}" for _, _, r in models]
        lines.append(f"| {g} | {len(qs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cand_quality(work, w2v_dir):
    names = open(os.path.join(w2v_dir, "entities.txt"), encoding="utf-8").read().split("\n")
    idx = {n: i for i, n in enumerate(names)}
    qs = read_jsonl(os.path.join(work, "queries_test.jsonl"))
    link = {r["id"]: r for r in read_jsonl(os.path.join(work, "queries_test.ent_link.jsonl"))}
    dense = {r["id"]: r for r in read_jsonl(os.path.join(work, "queries_test.ent_dense.jsonl"))}
    lines = ["| Candidate source (test queries) | avg #cands | gold-article-entity recall |", "|---|---|---|"]
    gold = {q["id"]: idx.get(q["article"]) for q in qs}
    for lab, fn in [("WikiLinker (REL analogue)", lambda q: link[q]["entities"]),
                    ("Dense W2V top-10 (LaQue analogue)", lambda q: dense[q]["entities"][:10]),
                    ("Dense W2V top-30", lambda q: dense[q]["entities"][:30]),
                    ("Linker ∪ dense top-10", lambda q: set(link[q]["entities"]) | set(dense[q]["entities"][:10]))]:
        c = [fn(q["id"]) for q in qs]
        rec = np.mean([gold[q["id"]] in set(x) for q, x in zip(qs, c)])
        lines.append(f"| {lab} | {np.mean([len(x) for x in c]):.1f} | {100 * rec:.1f}% |")
    return "\n".join(lines)


def plots(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, lab, c in [("lsr_init", "stage-0 LSR init", "#999999"), ("lsr_w", "LSR-w", "#4C72B0"),
                         ("dyvo_link_w2v", "DyVo (linked, W2V)", "#DD8452"),
                         ("dyvo_linkdense_w2v", "DyVo (link+dense)", "#55A868"),
                         ("dyvo_linkdense_gate", "DyVo-Gate (Ext-A)", "#C44E52")]:
        p = os.path.join(runs, name, "train_log.json")
        if not os.path.exists(p):
            continue
        log = json.load(open(p))
        k = np.array([r["kl"] for r in log])
        sm = np.convolve(k, np.ones(25) / 25, mode="valid")
        ax.plot(sm, label=lab, color=c, lw=1.4)
    ax.set_xlabel("step"); ax.set_ylabel("KL to monoT5 (smoothed)"); ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "training_curves.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--w2v", default="/home/user/data/w2v/converted")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = load(a.runs, "lsr_w")
    bm25 = bm25_res(a.work)
    R = {}
    R["R1"] = [HDR, row("BM25", bm25)]
    for lab, n in [("LSR-w stage-0 init (no distillation)", "lsr_init"), ("LSR-w", "lsr_w"),
                   ("DyVo (linked entities, Wikipedia2Vec)", "dyvo_link_w2v")]:
        r = load(a.runs, n)
        if r:
            R["R1"].append(row(lab, r, base if n != "lsr_w" else None))
    R["R2"] = [HDR]
    for lab, n, tag in [("DyVo – linked (precision)", "dyvo_link_w2v", "test"),
                        ("DyVo – link ∪ dense top-10/20 (recall, noisy)", "dyvo_linkdense_w2v", "test"),
                        ("  … same model, 3x test-time noise (top-30/30)", "dyvo_linkdense_w2v", "test_link+dense_q30_d30"),
                        ("DyVo-Gate (Ext-A) – link ∪ dense", "dyvo_linkdense_gate", "test"),
                        ("  … same model, 3x test-time noise (top-30/30)", "dyvo_linkdense_gate", "test_link+dense_q30_d30")]:
        r = load(a.runs, n, tag)
        if r:
            R["R2"].append(row(lab, r, base))
    R["R3"] = [HDR]
    for lab, n in [("Wikipedia2Vec (100d, frozen, projected)", "dyvo_link_w2v"),
                   ("Token Aggr. (DistilBERT input embeddings)", "dyvo_link_tokaggr"),
                   ("Static learned table (Ext-B control)", "dyvo_link_learned")]:
        r = load(a.runs, n)
        if r:
            R["R3"].append(row(lab, r, base))
    text = []
    for k, title in [("R1", "Headline reproduction (paper Table 1 analogue)"),
                     ("R2", "Candidate source & Ext-A noise robustness (paper Table 2 analogue)"),
                     ("R3", "Entity embeddings (paper Table 3 analogue)")]:
        text += [f"### {title}", "", *R[k], ""]
    text += ["### Candidate quality", "", cand_quality(a.work, a.w2v), ""]
    text += ["### Linking consistency (nDCG@10 by query group)", "", consistency(a.runs, a.work), ""]
    try:
        text += ["### Ext-B: dynamic vocabulary and unseen entities", "", ext_b(a.runs, a.work, a.out), ""]
    except Exception as e:  # partial results
        text += [f"(Ext-B pending: {e})"]
    text += ["† / ‡ : significantly better / worse than LSR-w (paired t-test on nDCG@10, p<0.05)."]
    open(os.path.join(a.out, "tables.md"), "w").write("\n".join(text))
    plots(a.runs, a.out)
    summ = {}
    for p in glob.glob(os.path.join(a.runs, "*", "metrics_*.json")):
        r = json.load(open(p))
        summ[os.path.relpath(p, a.runs)] = {"agg": r["agg"], "stats": r.get("stats"), "ms_per_query": r.get("ms_per_query")}
    summ["bm25/metrics_test.json"] = {"agg": bm25["agg"]}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("\n".join(text))


if __name__ == "__main__":
    main()
