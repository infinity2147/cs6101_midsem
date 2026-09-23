"""Convert the paper's datasets + the authors' released DyVo data into our format.

Needs (outside our sandbox): ir_datasets access to the licensed collections and
`huggingface-cli download lsr42/dyvo_data` (entity tables, candidates, InPars-v2 queries
and monoT5-3B scores). Then, e.g. for Robust04:

  python -m dyvo.paper_data export --irds disks45/nocr/trec-robust-2004 --query_field description \
      --doc_fields title body --out work/robust04
  python -m dyvo.paper_data train --queries dyvo_data/robust04/.../entity_topics-robust04.tsv \
      --qrels .../entity_robust04_qrels.json --ce .../entity_monot5_3b_scores.json --out work/robust04
  python -m dyvo.paper_data cands --queries dyvo_data/robust04/queries_test_train_entities_rel.jsonl \
      --docs dyvo_data/robust04/docs_entities.jsonl --out work/robust04 --tag link
  python -m dyvo.train --data work/robust04 --train_queries queries_train --triples work/robust04/triples.jsonl \
      --cands link --ent_emb table --ent_table dyvo_data/knowledge_base/entity_embs_laque.pt \
      --backbone distilbert-base-uncased --d_len 512 --batch_size 32 ... (see configs/paper_*.sh)

Dataset settings copied from the reference configs (lsr/configs/dataset/*.yaml):
  Robust04  disks45/nocr/trec-robust-2004  docs: title+body  queries: description
  Core18    wapo/v2/trec-core-2018          docs: title+body  queries: description
  CODEC     codec                           docs: title+text  queries: query
"""
import argparse
import json
import os
import random

from .data import write_jsonl, write_qrels


def export(irds, out, query_field, doc_fields):
    import ir_datasets
    ds = ir_datasets.load(irds)
    os.makedirs(out, exist_ok=True)
    write_jsonl(os.path.join(out, "corpus.jsonl"),
                ({"id": d.doc_id, "title": "",
                  "text": " ".join(getattr(d, f) for f in doc_fields if getattr(d, f, None))}
                 for d in ds.docs_iter()))
    qs = [{"id": q.query_id, "text": getattr(q, query_field)} for q in ds.queries_iter()]
    write_jsonl(os.path.join(out, "queries_test.jsonl"), qs)
    write_jsonl(os.path.join(out, "queries_val.jsonl"), [])
    qrels = {}
    for r in ds.qrels_iter():
        qrels.setdefault(r.query_id, {})[r.doc_id] = r.relevance
    write_qrels(os.path.join(out, "qrels_test.txt"), qrels)
    write_qrels(os.path.join(out, "qrels_val.txt"), {})


def train(queries_tsv, qrels_json, ce_json, out, samples_per_query=8, seed=0):
    """Mimics lsr.datasets.TripletIDDistilDataset (train_group_size=2, neg_as_pos=True)."""
    rng = random.Random(seed)
    qs = []
    with open(queries_tsv) as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            qs.append({"id": qid, "text": text})
    qrels = json.load(open(qrels_json))
    ce = json.load(open(ce_json))
    qs = [q for q in qs if q["id"] in ce]
    write_jsonl(os.path.join(out, "queries_train.jsonl"), qs)
    triples = []
    for q in qs:
        cands = list(ce[q["id"]])
        for _ in range(samples_per_query):
            if q["id"] in qrels and rng.random() < 0.6:
                d1 = rng.choice(list(qrels[q["id"]]))
            else:
                d1 = rng.choice(cands)
            d2 = rng.choice(cands)
            s1 = ce[q["id"]].get(d1, max(ce[q["id"]].values()))
            triples.append({"qid": q["id"], "pos": d1, "neg": d2, "pos_score": s1,
                            "neg_score": ce[q["id"]][d2]})
    write_jsonl(os.path.join(out, "triples.jsonl"), triples)
    print(f"{len(qs)} queries, {len(triples)} triples")


def cands(queries_jsonl, docs_jsonl, out, tag):
    """authors' {"id", "entities": [row ids into their embedding table]} -> ours"""
    for src, stems in [(queries_jsonl, ["queries_train", "queries_test", "queries_val"]),
                       (docs_jsonl, ["corpus"])]:
        rows = []
        with open(src) as f:
            for line in f:
                r = json.loads(line)
                ents = list(dict.fromkeys(int(e) for e in r["entities"]))
                rows.append({"id": str(r["id"]), "entities": ents, "scores": [1.0] * len(ents)})
        for stem in stems:
            write_jsonl(os.path.join(out, f"{stem}.ent_{tag}.jsonl"), rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--irds", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--query_field", default="description")
    e.add_argument("--doc_fields", nargs="+", default=["title", "body"])
    t = sub.add_parser("train")
    t.add_argument("--queries", required=True)
    t.add_argument("--qrels", required=True)
    t.add_argument("--ce", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--samples_per_query", type=int, default=8)
    c = sub.add_parser("cands")
    c.add_argument("--queries", required=True)
    c.add_argument("--docs", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--tag", default="link")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.irds, a.out, a.query_field, a.doc_fields)
    elif a.cmd == "train":
        train(a.queries, a.qrels, a.ce, a.out, a.samples_per_query)
    else:
        cands(a.queries, a.docs, a.out, a.tag)
