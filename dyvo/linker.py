"""Entity candidate generation for the DyVo head.

The paper feeds DyVo with candidates from REL (entity linker), BM25, LaQue (dense
entity retrieval), Mixtral/GPT-4 (generative). REL's data dumps are not reachable
from our sandbox, so we implement two lightweight analogues on top of Wikipedia2Vec:

* ``WikiLinker``  – *precision-oriented* linker (REL analogue):
    mention detection by longest-match n-gram lookup in an alias table built from all
    2.59M Wikipedia titles (+ "Title (qualifier)", "Place, Region", surname and acronym
    aliases, since anchor-text statistics are unavailable), with
    capitalisation heuristics; disambiguation by
        score(e | m, text) = cos(v_e, v_text) + a * popularity(e) + b * [exact title]
    where v_text is the mean Wikipedia2Vec word vector of the text (words and entities
    live in the same space, cf. Yamada et al. 2016) and popularity is the frequency
    rank in the Wikipedia2Vec vocabulary.
* ``DenseEntityRetriever`` – *recall-oriented* retriever (LaQue / BM25 analogue):
    top-k entities by cosine to v_text. Noisy by design (used for the noise study).

Output rows: {"id", "entities": [w2v_row...], "scores": [...], "mentions": [...]}.
"""
import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict

import numpy as np

from .data import doc_text, read_jsonl
from .w2v import Wikipedia2Vec

_WORD = re.compile(r"\w+", re.UNICODE)
_STOP = set("""a an and are as at be but by for from has have he her his i if in into is it its of
on or she that the their them they this to was were what when where which who whom why will with
how did do does can could would should than then there these those been being about after before
many much most more one two first also other some such not no only our out over so up we you your
""".split())


def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return " ".join(_WORD.findall(s))


class WikiLinker:
    def __init__(self, w2v: Wikipedia2Vec, max_n=6, max_cands=8, a_pop=0.35, b_exact=0.15,
                 min_score=0.45, common_words=1500):
        self.w2v = w2v
        self.max_n = max_n
        self.a_pop, self.b_exact, self.min_score = a_pop, b_exact, min_score
        self.N = len(w2v.entities)
        self.common = set(w2v.words[:common_words]) | _STOP
        self.frequent = set(w2v.words[:8000])
        alias = defaultdict(list)
        paren = re.compile(r"^(.*) \(([^)]*)\)$")
        for r, title in enumerate(w2v.entities):
            if title.startswith(("List of", "Lists of", "Index of", "Outline of")):
                continue
            alias[norm(title)].append((r, 1))
            m = paren.match(title)
            if m:
                alias[norm(m.group(1))].append((r, 0))
            elif ", " in title:
                alias[norm(title.split(", ")[0])].append((r, 0))
            toks = title.split()
            if 2 <= len(toks) <= 3 and all(t[:1].isupper() and t.isalpha() for t in toks):
                alias[norm(toks[-1])].append((r, 0))          # surname-style alias: "Tesla"
            caps = [t for t in toks if t[:1].isupper()]
            if len(caps) >= 2 and len(toks) <= 8 and not m:
                alias["".join(t[0] for t in caps).lower()].append((r, 0))  # acronym: "IPCC"
        # keep the most popular candidates per alias (entities are rank-ordered)
        self.alias = {k: sorted(v)[:max_cands] for k, v in alias.items()}
        ev = np.asarray(w2v.entity_vecs, dtype=np.float32)
        self.ent_norm = np.linalg.norm(ev, axis=1) + 1e-6

    def popularity(self, r):
        return 1.0 - math.log1p(r) / math.log1p(self.N)

    def _eligible(self, toks, i, j):
        span = toks[i:j]
        words = [t for t, _, _ in span]
        low = [w.lower() for w in words]
        if all(w in self.common for w in low) and not any(w.isupper() and len(w) > 1 for w in words):
            return False
        if j - i == 1:
            w = words[0]
            if w.isdigit() or len(w) < 2:
                return False
            if not w[0].isupper():
                return False
            if low[0] in self.common:
                return False
            if i == 0 and not w.isupper() and low[0] in self.frequent:
                return False  # sentence-initial capitalisation of an ordinary word
            return True
        if low[0] in _STOP or low[-1] in _STOP:
            return False
        return any(w[0].isupper() or w[0].isdigit() for w in words)

    def link(self, text):
        toks = [(m.group(), m.start(), m.end()) for m in _WORD.finditer(unicodedata.normalize("NFKC", text))]
        v_text = self.w2v.text_vector([t.lower() for t, _, _ in toks if t.lower() not in _STOP])
        v_norm = np.linalg.norm(v_text) + 1e-6
        out, i = {}, 0
        while i < len(toks):
            matched = False
            for j in range(min(len(toks), i + self.max_n), i, -1):
                key = " ".join(t.lower() for t, _, _ in toks[i:j])
                cands = self.alias.get(key)
                if not cands or not self._eligible(toks, i, j):
                    continue
                best = None
                for r, exact in cands:
                    sim = float(np.dot(self.w2v.entity_vecs[r].astype(np.float32), v_text)
                                / (self.ent_norm[r] * v_norm))
                    s = sim + self.a_pop * self.popularity(r) + self.b_exact * exact
                    if best is None or s > best[1]:
                        best = (r, s)
                if best[1] >= self.min_score:
                    r, s = best
                    if r not in out or out[r][0] < s:
                        out[r] = (s, text[toks[i][1]:toks[j - 1][2]])
                    i, matched = j, True
                    break
            if not matched:
                i += 1
        items = sorted(out.items(), key=lambda x: -x[1][0])
        return {"entities": [r for r, _ in items], "scores": [round(s, 4) for _, (s, _) in items],
                "mentions": [m for _, (_, m) in items]}


class DenseEntityRetriever:
    """top-k Wikipedia2Vec entities by cosine with the mean word vector of the text."""

    def __init__(self, w2v: Wikipedia2Vec, max_entities=1_000_000):
        self.w2v = w2v
        E = np.asarray(w2v.entity_vecs[:max_entities], dtype=np.float32)
        self.E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)

    def retrieve(self, texts, k=20, batch=64):
        res = []
        for b in range(0, len(texts), batch):
            V = np.stack([self.w2v.text_vector([t for t in norm(x).split() if t not in _STOP])
                          for x in texts[b:b + batch]])
            V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-6
            S = V @ self.E.T
            top = np.argpartition(-S, k, axis=1)[:, :k]
            for row, idx in zip(S, top):
                idx = idx[np.argsort(-row[idx])]
                res.append({"entities": idx.tolist(), "scores": [round(float(row[i]), 4) for i in idx]})
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w2v", required=True)
    ap.add_argument("--inputs", nargs="+", required=True, help="jsonl files (corpus / queries)")
    ap.add_argument("--method", choices=["link", "dense"], required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out_suffix", default=None)
    a = ap.parse_args()
    w2v = Wikipedia2Vec(a.w2v)
    tool = WikiLinker(w2v) if a.method == "link" else DenseEntityRetriever(w2v)
    for path in a.inputs:
        rows = read_jsonl(path)
        texts = [doc_text(r) if "title" in r else r["text"] for r in rows]
        if a.method == "link":
            outs = [tool.link(t) for t in texts]
        else:
            outs = tool.retrieve(texts, k=a.k)
        out_path = path.replace(".jsonl", f".ent_{a.out_suffix or a.method}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r, o in zip(rows, outs):
                f.write(json.dumps({"id": r["id"], **o}, ensure_ascii=False) + "\n")
        n = np.mean([len(o["entities"]) for o in outs])
        print(f"{path}: {len(rows)} texts, {n:.2f} entities/text -> {out_path}")


if __name__ == "__main__":
    main()
