"""Cross-encoder teacher (monoT5) for KL distillation.

The paper distils from monoT5-3B scores on InParsV2 synthetic queries
(`..._inparsv2_monot53b_distillation...` configs). We use castorini/monoT5-base-msmarco
(same model family and prompt, 13x smaller) so it runs on CPU.
Score = log P("true") - log P("false") given "Query: q Document: d Relevant:".
"""
import argparse
import json
import random
import time

import torch
import sentencepiece as spm
from transformers import T5ForConditionalGeneration

from .data import doc_text, read_jsonl, read_qrels, read_run


class MonoT5:
    def __init__(self, path, device="cpu", max_length=384, bf16=True):
        # SentencePiece directly: transformers>=5 T5Tokenizer(vocab_file=...) mis-tokenizes
        self.sp = spm.SentencePieceProcessor(model_file=f"{path}/spiece.model")
        self.model = T5ForConditionalGeneration.from_pretrained(path).eval().to(device)
        self.device = device
        self.max_length = max_length
        self.bf16 = bf16  # CPU AMX/avx512-bf16 autocast: ~3x faster, scores match fp32 closely
        self.true_id = self.sp.piece_to_id("▁true")    # 1176
        self.false_id = self.sp.piece_to_id("▁false")  # 6136

    @torch.inference_mode()
    def score(self, queries, docs):
        inp = [f"Query: {q} Document: {d} Relevant:" for q, d in zip(queries, docs)]
        ids = [x[: self.max_length - 1] + [1] for x in self.sp.encode(inp)]  # </s> = 1
        L = max(map(len, ids))
        input_ids = torch.tensor([x + [0] * (L - len(x)) for x in ids], device=self.device)
        enc = {"input_ids": input_ids, "attention_mask": (input_ids != 0).long()}
        dec = torch.full((len(inp), 1), self.model.config.decoder_start_token_id, device=self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.bf16):
            logits = self.model(**enc, decoder_input_ids=dec).logits[:, 0].float()
        lp = torch.log_softmax(logits[:, [self.true_id, self.false_id]], dim=-1)
        return (lp[:, 0] - lp[:, 1]).tolist()


def build_triples(queries, qrels, run, n_neg=1, depth=30, seed=0):
    """(qid, pos, neg) with negatives sampled from the BM25 top-`depth` (hard negatives)."""
    rng = random.Random(seed)
    triples = []
    for q in queries:
        pos = [d for d, r in qrels[q["id"]].items() if r > 0]
        cands = [d for d in list(run.get(q["id"], {}))[:depth] if d not in qrels[q["id"]]]
        if not pos or not cands:
            continue
        for neg in rng.sample(cands, min(n_neg, len(cands))):
            triples.append((q["id"], rng.choice(pos), neg))
    return triples


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--bm25_run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_queries", type=int, default=16000)
    ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(3)
    docs = {d["id"]: doc_text(d) for d in read_jsonl(a.corpus)}
    queries = read_jsonl(a.queries)[: a.n_queries]
    qtext = {q["id"]: q["text"] for q in queries}
    triples = build_triples(queries, read_qrels(a.qrels), read_run(a.bm25_run))
    teacher = MonoT5(a.model)
    # process in length-sorted order to minimise padding
    triples.sort(key=lambda t: len(docs[t[1]]) + len(docs[t[2]]))
    import os
    done = set()
    if os.path.exists(a.out):  # resume after interruption
        for l in open(a.out):
            try:
                r = json.loads(l); done.add((r["qid"], r["pos"], r["neg"]))
            except json.JSONDecodeError:
                pass
    triples = [t for t in triples if t not in done]
    print(f"resuming: {len(done)} done, {len(triples)} left", flush=True)
    t0 = time.time()
    with open(a.out, "a") as f:
        for i in range(0, len(triples), a.batch_size // 2):
            chunk = triples[i: i + a.batch_size // 2]
            qs = [qtext[q] for q, _, _ in chunk for _ in range(2)]
            ds = [docs[d] for _, p, n in chunk for d in (p, n)]
            s = teacher.score(qs, ds)
            for j, (q, p, n) in enumerate(chunk):
                f.write(json.dumps({"qid": q, "pos": p, "neg": n,
                                    "pos_score": s[2 * j], "neg_score": s[2 * j + 1]}) + "\n")
            f.flush()
            if (i // (a.batch_size // 2)) % 20 == 0:
                done = i + len(chunk)
                print(f"{done}/{len(triples)} triples, {done / (time.time() - t0):.1f} triples/s", flush=True)
