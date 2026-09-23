# Running the paper's headline experiment (Robust04 / Core18 / CODEC) on a GPU

Our sandbox could not reach the licensed TREC collections or the HuggingFace hub. With access
to them, the same code reproduces the paper's Table 1 as follows.

## Settings copied from the reference configs (`lsr/configs/`)

| | paper (reference repo) | our flags |
|---|---|---|
| backbone | `distilbert-base-uncased` | `--backbone distilbert-base-uncased` |
| query / doc encoder | `mlp_emlm` / `mlm_emlm` | fixed in `dyvo/model.py` |
| entity weight init | `entity_weight: 0.05` | `--entity_weight 0.05` |
| loss | `EntityDistilKLLoss`, `reg_ent: False` | KL + doc L1 on word pieces only |
| L1 (doc) | 1e-3 / 1e-4 / 1e-5, `T: 50000` quadratic warm-up | `--l1_d`, warm-up = steps/2 |
| teacher | monoT5-3B scores on InPars-v2 queries | `dyvo/paper_data.py train --ce ...` |
| init | MS MARCO LSR (`dyvo_data/dyvo_init`) | `--init <dir with model.pt>` (convert safetensors) |
| query length / doc length | 50 / 512 | `--d_len 512` |
| steps / LR | 100k, lr 5e-7, warm-up ratio 0.1 | set `--epochs`, `--lr` |
| queries | Robust04 & Core18: `description`; CODEC: `query` | `paper_data export --query_field` |
| docs | `title + body` (CODEC `title + text`) | `--doc_fields` |

## Steps

```bash
pip install ir_datasets huggingface_hub
huggingface-cli download lsr42/dyvo_data --local-dir dyvo_data

D=work/robust04
python -m dyvo.paper_data export --irds disks45/nocr/trec-robust-2004 --query_field description --doc_fields title body --out $D
python -m dyvo.paper_data train  --queries dyvo_data/robust04/<inparsv2 topics>.tsv --qrels <...qrels.json> --ce <...monot5_3b_scores.json> --out $D
python -m dyvo.paper_data cands  --queries dyvo_data/robust04/queries_test_train_entities_rel.jsonl --docs dyvo_data/robust04/docs_entities.jsonl --out $D --tag link

COMMON="--data $D --train_queries queries_train --triples $D/triples.jsonl --backbone distilbert-base-uncased --d_len 512 --batch_size 32 --l1_d 1e-5"
python -m dyvo.train $COMMON --out runs/robust04_lsrw
python -m dyvo.train $COMMON --out runs/robust04_dyvo --cands link --ent_emb table --ent_table dyvo_data/knowledge_base/entity_embs_laque.pt
python -m dyvo.evaluate --model_dir runs/robust04_lsrw
python -m dyvo.evaluate --model_dir runs/robust04_dyvo
```

Repeat with `wapo/v2/trec-core-2018` and `codec` (`--query_field query --doc_fields title text`).
Robust04 has 528k documents: use a GPU for encoding. The exhaustive scipy retrieval needs about 2 GB RAM.
`dyvo/model.py` and `train.py` use `torch.autocast("cpu")`; on a GPU change that to `"cuda"` and move the batches with `.to("cuda")`.

## Paper targets (Table 1, DistilBERT, REL candidates, LaQue embeddings)

Values collected from paper excerpts; double-check them against the PDF before citing.

| Dataset | model | nDCG@10 | nDCG@20 | R@1k |
|---|---|---|---|---|
| Robust04 | LSR-w | 49.13 | – | 66.86 |
| Robust04 | DyVo (REL) | 51.19 | – | 68.56 |
| Core18 | LSR-w | 40.99 | 38.73 | 63.22 |
| CODEC (reg 1e-3) | BM25 | 37.70 | 35.28 | 61.25 |
| CODEC (reg 1e-3) | LSR-w | 39.10 | 35.32 | 57.58 |
| CODEC (reg 1e-3) | DyVo (REL) | 42.67 | 38.32 | 59.81 |

Best reported (Table 3, GPT-4 candidates + Wikipedia2Vec): Robust04 54.04, Core18 44.15,
CODEC 56.30 nDCG@10.
