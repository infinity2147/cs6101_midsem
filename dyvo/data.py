"""Dataset construction and I/O.

SQuAD-Open-Para: an open, entity-rich Wikipedia paragraph-retrieval benchmark built
from SQuAD v1.1 (the paper's Robust04 / Core18 / CODEC are licensed or on request, so
they cannot be downloaded in our sandbox; `configs/paper_datasets.md` explains how to
run the same code on them).

* corpus  : every paragraph of SQuAD train + dev (20,958 paragraphs, 490 Wikipedia
            articles), text = "<article title>. <paragraph>"   (title+body, as in the
            paper's doc_field=['title','body'])
* test    : N questions sampled from SQuAD *dev*; relevant doc = source paragraph
* train   : questions from SQuAD *train*; dev articles are never seen in training,
            so test queries are about topics/entities unseen during training
            (this is what the zero-shot vocabulary-growth extension exploits).

All files are simple JSONL / TSV so the same pipeline can read converted TREC data.
"""
import argparse
import json
import os
import random


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def read_qrels(path):
    """TREC qrels: qid 0 docid rel"""
    qrels = {}
    with open(path) as f:
        for line in f:
            qid, _, did, rel = line.split()
            qrels.setdefault(qid, {})[did] = int(rel)
    return qrels


def write_qrels(path, qrels):
    with open(path, "w") as f:
        for qid, docs in qrels.items():
            for did, rel in docs.items():
                f.write(f"{qid} 0 {did} {rel}\n")


def write_run(path, run, tag="dyvo"):
    with open(path, "w") as f:
        for qid, docs in run.items():
            ranked = sorted(docs.items(), key=lambda x: -x[1])
            for rank, (did, s) in enumerate(ranked, 1):
                f.write(f"{qid} Q0 {did} {rank} {s:.6f} {tag}\n")


def read_run(path):
    run = {}
    with open(path) as f:
        for line in f:
            qid, _, did, _, s, _ = line.split()
            run.setdefault(qid, {})[did] = float(s)
    return run


def build_squad_open(squad_dir, out_dir, n_test=2000, n_train=None, seed=13):
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    corpus, queries = [], {"train": [], "dev": []}
    for split in ["train", "dev"]:
        data = json.load(open(os.path.join(squad_dir, f"{split}-v1.1.json")))["data"]
        for ai, art in enumerate(data):
            title = art["title"].replace("_", " ")
            for pi, para in enumerate(art["paragraphs"]):
                did = f"{split[0]}{ai}_{pi}"
                corpus.append({"id": did, "title": title, "text": para["context"],
                               "article": title})
                for qa in para["qas"]:
                    queries[split].append({"id": qa["id"], "text": qa["question"].strip(),
                                           "pos": did, "article": title})
    write_jsonl(os.path.join(out_dir, "corpus.jsonl"), corpus)
    rng.shuffle(queries["dev"])
    test = queries["dev"][:n_test]
    val = queries["dev"][n_test:n_test + 500]
    train = queries["train"]
    rng.shuffle(train)
    if n_train:
        train = train[:n_train]
    for name, qs in [("test", test), ("val", val), ("train", train)]:
        write_jsonl(os.path.join(out_dir, f"queries_{name}.jsonl"),
                    [{"id": q["id"], "text": q["text"], "article": q["article"]} for q in qs])
        write_qrels(os.path.join(out_dir, f"qrels_{name}.txt"), {q["id"]: {q["pos"]: 1} for q in qs})
    print(f"corpus={len(corpus)} train={len(train)} val={len(val)} test={len(test)}")


def doc_text(d):
    return f"{d['title']}. {d['text']}" if d.get("title") else d["text"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_test", type=int, default=2000)
    a = ap.parse_args()
    build_squad_open(a.squad_dir, a.out, a.n_test)
