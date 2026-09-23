"""Wikipedia2Vec loading.

The pretrained Wikipedia2Vec text file (enwiki_20180420_100d.txt.bz2) interleaves
words and entities (``ENTITY/Title``) sorted by corpus frequency. We split it into

* ``entities.txt`` + ``entity_vecs.npy``  – every entity, in frequency-rank order
* ``words.txt``    + ``word_vecs.npy``    – the top-N words (used for context vectors)

The entity rank doubles as a popularity prior for the entity linker, and the
entity row index is the entity's id in the DyVo vocabulary (token id = 30522 + row).
"""
import argparse
import bz2
import os

import numpy as np


def convert(src, out_dir, max_words=400_000):
    os.makedirs(out_dir, exist_ok=True)
    ent_titles, ent_vecs, words, word_vecs = [], [], [], []
    with bz2.open(src, "rt", encoding="utf-8") as f:
        n, dim = map(int, f.readline().split())
        for i, line in enumerate(f):
            key, _, rest = line.rstrip("\n").partition(" ")
            if key.startswith("ENTITY/"):
                ent_titles.append(key[7:].replace("_", " "))
                ent_vecs.append(np.array(rest.split(" "), dtype=np.float16))
            elif len(words) < max_words:
                words.append(key)
                word_vecs.append(np.array(rest.split(" "), dtype=np.float16))
            if i % 500_000 == 0:
                print(f"{i}/{n} lines, {len(ent_titles)} entities", flush=True)
    np.save(os.path.join(out_dir, "entity_vecs.npy"), np.stack(ent_vecs))
    np.save(os.path.join(out_dir, "word_vecs.npy"), np.stack(word_vecs))
    with open(os.path.join(out_dir, "entities.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ent_titles) + "\n")
    with open(os.path.join(out_dir, "words.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(words) + "\n")
    print(f"done: {len(ent_titles)} entities, {len(words)} words, dim={dim}")


class Wikipedia2Vec:
    """Memory-mapped access to the converted Wikipedia2Vec tables."""

    def __init__(self, path):
        self.entity_vecs = np.load(os.path.join(path, "entity_vecs.npy"), mmap_mode="r")
        self.word_vecs = np.load(os.path.join(path, "word_vecs.npy"), mmap_mode="r")
        with open(os.path.join(path, "entities.txt"), encoding="utf-8") as f:
            self.entities = [l.rstrip("\n") for l in f]
        with open(os.path.join(path, "words.txt"), encoding="utf-8") as f:
            self.words = [l.rstrip("\n") for l in f]
        self.word2id = {w: i for i, w in enumerate(self.words)}
        self.dim = self.entity_vecs.shape[1]

    def text_vector(self, tokens):
        ids = [self.word2id[t] for t in tokens if t in self.word2id]
        if not ids:
            return np.zeros(self.dim, dtype=np.float32)
        return np.asarray(self.word_vecs[ids], dtype=np.float32).mean(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_words", type=int, default=400_000)
    a = ap.parse_args()
    convert(a.src, a.out, a.max_words)
