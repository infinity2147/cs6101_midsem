# DyVo: Dynamic Vocabularies for Learned Sparse Retrieval with Entities — CS6101 course project

Re-implementation, reproduction and extensions of
**DyVo** (Nguyen, Chatterjee, MacAvaney, Mackie, Dalton, Yates — EMNLP 2024).
Reference code: <https://github.com/thongnt99/DyVo>.

➡️ **Mid-semester report:** [`report/MIDSEM_REPORT.md`](report/MIDSEM_REPORT.md)
(results tables: [`report/results/tables.md`](report/results/tables.md))

## What DyVo does (one paragraph)

Learned sparse retrievers (LSR, e.g. SPLADE) score `q·d` over the 30,522 BERT word pieces, so
entities get shredded ("Nikola Tesla" → `nikola`, `te`, `##sla`) and cannot be updated without
retraining. DyVo adds a **Dynamic Vocabulary head**: for each text it takes a small set of
*candidate* Wikipedia entities (from an entity linker / retriever / LLM), and scores each one with
the frozen, external entity embedding `E_e` as

```
w(e) = max_t log(1 + relu( α · <h_t, P·E_e> ))      (h_t = contextual token states)
```

The entity weights are appended to the word-piece weights, giving one joint sparse vector that
still works with an inverted index. Because `E_e` comes from outside the model, you can add new
entities without retraining.

## Repository layout

| Path | What |
|---|---|
| `dyvo/model.py` | MLP query encoder, MLM (SPLADE) doc encoder, **DyVo head**, gated head (Ext-A), learned-table control (Ext-B) |
| `dyvo/train.py` | KL distillation from monoT5 + L1 sparsity (quadratic warm-up), entity-aware collator |
| `dyvo/evaluate.py` | encodes into the joint sparse space, exhaustive sparse retrieval, metrics, FLOPs/sparsity |
| `dyvo/linker.py` | Wikipedia2Vec-based **entity linker** (REL analogue) and **dense entity retriever** (LaQue analogue) |
| `dyvo/entity_store.py` | dynamic entity vocabulary: candidates → ids, embeddings (W2V / Token-Aggr / any table), gate features |
| `dyvo/teacher.py` | monoT5 cross-encoder teacher (bf16 on CPU) |
| `dyvo/data.py`, `dyvo/bm25.py`, `dyvo/metrics.py` | SQuAD-Open-Para benchmark, BM25 baseline, nDCG/R/MRR (checked against `ir_measures`) |
| `dyvo/paper_data.py` | converters for the **paper's datasets** (Robust04 / Core18 / CODEC + authors' `lsr42/dyvo_data`) |
| `dyvo/analysis.py` | builds the result tables, significance tests, Ext-B breakdown and figures |
| `scripts/run_squad_experiments.sh` | whole CPU pipeline end to end |
| `configs/paper_datasets.md` | how to run the paper's headline experiment on a GPU |

## Quick start

```bash
pip install -r requirements.txt
# downloads: DistilBERT + monoT5-base (legacy HF S3), SQuAD v1.1, Wikipedia2Vec enwiki_20180420_100d
bash scripts/download.sh
python -m dyvo.w2v --src data/w2v/enwiki_20180420_100d.txt.bz2 --out data/w2v/converted
bash scripts/run_squad_experiments.sh        # ~5-6 h on 4 CPU cores (bf16), much faster on a GPU
```
