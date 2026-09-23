"""Training: KL distillation from monoT5 (paper recipe) + optional in-batch negatives.

loss = KL( softmax([T(q,d+), T(q,d-)]) || log_softmax([S(q,d+), S(q,d-)]) )      (paper)
     + lambda_ib * CE over all 2B in-batch documents                             (stage-0 only)
     + L1(doc word weights) * w_t,  w_t = w * min(1, t/T)^2                      (paper: reg_ent=False,
                                                                                 entities not regularised)
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from . import bf16_ok
from .data import doc_text, read_jsonl
from .entity_store import EntityStore
from .model import DyVo, entity_scores_all

BF16 = bf16_ok()


def stems(args):
    return ["corpus", getattr(args, "train_queries", "queries_train10k"), "queries_test", "queries_val"]


class Collator:
    def __init__(self, tok, store, q_len=50, d_len=256, noise_ents=0):
        self.tok, self.store = tok, store
        self.q_len, self.d_len = q_len, d_len
        self.noise_ents = noise_ents   # Ext-A: random distractor entities injected at training time

    def _ents(self, tids, feats_dim=4):
        if self.store is None:
            return {}
        ids, feats = zip(*[self.store.get(t) for t in tids])
        ids, feats = [list(x) for x in ids], [list(x) for x in feats]
        if self.noise_ents:
            for i in range(len(ids)):
                for _ in range(self.noise_ents):
                    e = random.randrange(len(self.store))
                    if e not in ids[i]:
                        ids[i].append(e)
                        feats[i].append([0.3, 0.0, 1.0, 0.5])  # looks like a dense-retrieved cand.
        K = max(1, max(len(x) for x in ids))
        E = torch.zeros(len(ids), K, dtype=torch.long)
        M = torch.zeros(len(ids), K)
        Fe = torch.zeros(len(ids), K, feats_dim)
        for i, (x, f) in enumerate(zip(ids, feats)):
            if x:
                E[i, :len(x)] = torch.tensor(x)
                M[i, :len(x)] = 1
                Fe[i, :len(x)] = torch.tensor(f)
        return {"ent_ids": E, "ent_mask": M, "ent_feats": Fe}

    def encode(self, texts, tids, max_len):
        enc = self.tok(texts, padding=True, truncation=True, max_length=max_len,
                       return_special_tokens_mask=True, return_tensors="pt")
        b = {k: enc[k] for k in ("input_ids", "attention_mask", "special_tokens_mask")}
        b.update(self._ents(tids))
        return b

    def queries(self, texts, tids):
        return self.encode(texts, tids, self.q_len)

    def docs(self, texts, tids):
        return self.encode(texts, tids, self.d_len)


def build_store(args):
    if args.cands == "none":
        return None
    return EntityStore(args.data, stems(args), source=args.cands, k_dense_q=args.k_dense_q,
                       k_dense_d=args.k_dense_d)


def build_model(args, store):
    ent = None
    if store is not None:
        ent = store.embeddings(args.ent_emb, w2v_dir=args.w2v, backbone=args.backbone,
                               table=getattr(args, "ent_table", None))
    return DyVo(args.backbone, ent_emb=ent, entity_weight=args.entity_weight, gate=args.gate,
                ent_emb_mode=args.ent_emb_mode) if ent is not None else DyVo(args.backbone)


def score_pairs(model, qb, db, B):
    qw, qe = model.encode_q(qb)
    dw, de = model.encode_d(db)
    # docs are laid out [p_0, n_0, p_1, n_1, ...]
    s_word = torch.einsum("bv,bnv->bn", qw, dw.view(B, 2, -1))
    s_all = qw @ dw.T                                                  # B x 2B (in-batch)
    if qe is not None:
        s_all = s_all + entity_scores_all(qb["ent_ids"], qe, db["ent_ids"], de)  # B x 2B
        idx = torch.arange(B)
        s_pairs = torch.stack([s_all[idx, 2 * idx], s_all[idx, 2 * idx + 1]], dim=1)
    else:
        s_pairs = s_word
    return s_pairs, s_all, qw, dw, qe, de


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--triples", required=True)
    ap.add_argument("--w2v", default="/home/user/data/w2v/converted")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="LSR-w checkpoint to initialise from")
    ap.add_argument("--cands", default="none", choices=["none", "link", "dense", "link+dense"])
    ap.add_argument("--k_dense_q", type=int, default=10)
    ap.add_argument("--k_dense_d", type=int, default=20)
    ap.add_argument("--ent_emb", default="w2v", choices=["w2v", "tokaggr", "table"])
    ap.add_argument("--ent_table", default=None, help="torch .pt [N x d] entity embeddings (paper's dyvo_data)")
    ap.add_argument("--train_queries", default="queries_train10k")
    ap.add_argument("--ent_emb_mode", default="frozen", choices=["frozen", "learned"])
    ap.add_argument("--entity_weight", type=float, default=0.05)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--noise_ents", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--triple_start", type=int, default=0)
    ap.add_argument("--triple_end", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--l1_d", type=float, default=1e-4)
    ap.add_argument("--l1_q", type=float, default=0.0)
    ap.add_argument("--lambda_ib", type=float, default=0.0)
    ap.add_argument("--d_len", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)

    docs = {d["id"]: doc_text(d) for d in read_jsonl(os.path.join(args.data, "corpus.jsonl"))}
    qtext = {q["id"]: q["text"] for q in read_jsonl(os.path.join(args.data, f"{args.train_queries}.jsonl"))}
    triples = read_jsonl(args.triples)
    triples.sort(key=lambda t: t["qid"])  # undo teacher's length sort; deterministic
    random.Random(0).shuffle(triples)
    triples = [t for t in triples if t["qid"] in qtext][args.triple_start:args.triple_end]

    tok = AutoTokenizer.from_pretrained(args.backbone)
    store = build_store(args)
    model = build_model(args, store)
    if args.init:
        missing, unexpected = model.load_lsr_weights(torch.load(os.path.join(args.init, "model.pt")))
        print(f"init from {args.init}: missing(non-head)={missing} unexpected={unexpected}")
    coll = Collator(tok, store, d_len=args.d_len, noise_ents=args.noise_ents)

    head_p = [p for n, p in model.named_parameters() if ".head." in n and p.requires_grad]
    base_p = [p for n, p in model.named_parameters() if ".head." not in n and p.requires_grad]
    opt = torch.optim.AdamW([{"params": base_p, "lr": args.lr},
                             {"params": head_p, "lr": args.head_lr}], weight_decay=0.01)
    steps = args.epochs * len(triples) // args.batch_size
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    T_reg = max(1, steps // 2)
    print(f"{len(triples)} triples, {steps} steps, entities={0 if store is None else len(store)}", flush=True)

    model.train()
    t0, log = time.time(), []
    step = 0
    for ep in range(args.epochs):
        for i in range(0, len(triples) - args.batch_size + 1, args.batch_size):
            batch = triples[i:i + args.batch_size]
            B = len(batch)
            qb = coll.queries([qtext[t["qid"]] for t in batch], [t["qid"] for t in batch])
            dids = [d for t in batch for d in (t["pos"], t["neg"])]
            db = coll.docs([docs[d] for d in dids], dids)
            teacher = torch.tensor([[t["pos_score"], t["neg_score"]] for t in batch])
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=BF16):
                s_pairs, s_all, qw, dw, qe, de = score_pairs(model, qb, db, B)
            s_pairs, s_all = s_pairs.float(), s_all.float()
            kl = F.kl_div(F.log_softmax(s_pairs, 1), F.softmax(teacher, 1), reduction="batchmean")
            loss = kl
            if args.lambda_ib > 0:
                ib = F.cross_entropy(s_all, torch.arange(B) * 2)
                loss = loss + args.lambda_ib * ib
            w_t = min(1.0, (step + 1) / T_reg) ** 2
            reg_d = dw.abs().sum(1).mean() * args.l1_d * w_t
            reg_q = qw.abs().sum(1).mean() * args.l1_q * w_t
            loss = loss + reg_d + reg_q
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "loss": loss.item(), "kl": kl.item(), "reg_d": reg_d.item(),
                   "q_nnz": (qw > 0).sum(1).float().mean().item(),
                   "d_nnz": (dw > 0).sum(1).float().mean().item(),
                   "acc": (s_pairs[:, 0] > s_pairs[:, 1]).float().mean().item()}
            if qe is not None:
                rec["q_ent_nnz"] = (qe > 0).sum(1).float().mean().item()
                rec["d_ent_nnz"] = (de > 0).sum(1).float().mean().item()
                rec["ent_weight"] = model.d_enc.head.ent_weight.item()
            log.append(rec)
            if step % args.log_every == 0 or step == steps:
                recent = log[-args.log_every:]
                avg = {k: round(sum(r[k] for r in recent) / len(recent), 4) for k in recent[0]}
                avg["step"] = step
                avg["s/step"] = round((time.time() - t0) / step, 2)
                print(json.dumps(avg), flush=True)
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    json.dump(log, open(os.path.join(args.out, "train_log.json"), "w"))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
