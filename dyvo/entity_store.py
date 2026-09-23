"""Dynamic entity vocabulary: candidate sets -> compact ids, embeddings and gate features.

Entity token id in the joint vocabulary = WORD_VOCAB_SIZE + compact id (same convention
as the reference code, which offsets entity ids by 30522).

Candidate sources (the paper's Table 2 axis):
  link          – WikiLinker (precision-oriented, REL analogue)
  dense         – DenseEntityRetriever top-k (recall-oriented, LaQue/BM25 analogue)
  link+dense    – union (noisy)
Entity embeddings (the paper's Table 3 axis):
  w2v           – Wikipedia2Vec 100d  (paper: Wikipedia2Vec 300d)
  tokaggr       – mean of the backbone's input word-piece embeddings of the title
                  (paper: "Token Aggr.")
  table         – any precomputed [N x d] tensor, e.g. the authors' LaQue / BLINK / JDS tables
"""
import json
import math
import os

import numpy as np
import torch

from .data import read_jsonl


def load_cands(path):
    return {r["id"]: r for r in read_jsonl(path)} if os.path.exists(path) else {}


class EntityStore:
    def __init__(self, data_dir, files, source="link", k_dense_q=10, k_dense_d=20,
                 max_q=40, max_d=64, n_w2v=2_592_608):
        """files: list of jsonl stems whose candidates to load, e.g. ['corpus', 'queries_test']"""
        self.source = source
        self.max_q, self.max_d = max_q, max_d
        self.n_w2v = n_w2v
        self.cands = {}
        for stem in files:
            link = load_cands(os.path.join(data_dir, f"{stem}.ent_link.jsonl"))
            dense = load_cands(os.path.join(data_dir, f"{stem}.ent_dense.jsonl"))
            is_doc = stem.startswith("corpus")
            k = k_dense_d if is_doc else k_dense_q
            cap = max_d if is_doc else max_q
            for tid in set(link) | set(dense):
                items = {}  # w2v row -> [score, linked, dense]
                if "link" in source and tid in link:
                    for e, s in zip(link[tid]["entities"], link[tid]["scores"]):
                        items[e] = [s, 1.0, 0.0]
                if "dense" in source and tid in dense:
                    for e, s in list(zip(dense[tid]["entities"], dense[tid]["scores"]))[:k]:
                        if e in items:
                            items[e][2] = 1.0
                        else:
                            items[e] = [s, 0.0, 1.0]
                ranked = sorted(items.items(), key=lambda x: -x[1][0])[:cap]
                self.cands[tid] = ranked
        rows = sorted({e for c in self.cands.values() for e, _ in c})
        self.rows = np.array(rows, dtype=np.int64)          # compact id -> w2v row
        self.row2id = {r: i for i, r in enumerate(rows)}

    def __len__(self):
        return len(self.rows)

    def popularity(self, row):
        return 1.0 - math.log1p(row) / math.log1p(self.n_w2v)

    def get(self, tid):
        """-> (compact ids, features[score, linked, dense, popularity])"""
        c = self.cands.get(tid, [])
        ids = [self.row2id[e] for e, _ in c]
        feats = [[f[0], f[1], f[2], self.popularity(e)] for e, f in c]
        return ids, feats

    def embeddings(self, kind, w2v_dir=None, backbone=None, table=None):
        if kind == "table":  # e.g. the paper's dyvo_data/knowledge_base/entity_embs_laque.pt
            return torch.load(table, map_location="cpu")[torch.from_numpy(self.rows)].float()
        if kind == "w2v":
            vecs = np.load(os.path.join(w2v_dir, "entity_vecs.npy"), mmap_mode="r")
            return torch.from_numpy(np.asarray(vecs[self.rows], dtype=np.float32))
        if kind == "tokaggr":
            from transformers import AutoModel, AutoTokenizer
            with open(os.path.join(w2v_dir, "entities.txt"), encoding="utf-8") as f:
                titles = f.read().split("\n")
            names = [titles[r] for r in self.rows]
            tok = AutoTokenizer.from_pretrained(backbone)
            emb = AutoModel.from_pretrained(backbone).get_input_embeddings().weight.detach()
            out = torch.zeros(len(names), emb.size(1))
            for i, n in enumerate(names):
                ids = tok(n, add_special_tokens=False)["input_ids"] or [tok.unk_token_id]
                out[i] = emb[ids].mean(0)
            return out
        raise ValueError(kind)

    def entity_names(self, w2v_dir):
        with open(os.path.join(w2v_dir, "entities.txt"), encoding="utf-8") as f:
            titles = f.read().split("\n")
        return [titles[r] for r in self.rows]

    def save_vocab(self, path, w2v_dir):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"rows": self.rows.tolist(), "names": self.entity_names(w2v_dir)}, f)
