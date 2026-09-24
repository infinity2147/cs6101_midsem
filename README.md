# DyVo: Dynamic Vocabularies for Learned Sparse Retrieval with Entities — CS6101 course project

Re-implementation, reproduction and extensions of
**DyVo** (Nguyen, Chatterjee, MacAvaney, Mackie, Dalton, Yates — EMNLP 2024).
Reference code: <https://github.com/thongnt99/DyVo>.

**Mid-semester report:** [`report/MIDSEM_REPORT.md`](report/MIDSEM_REPORT.md) (PDF: [`report/MIDSEM_REPORT.pdf`](report/MIDSEM_REPORT.pdf), web version: `report/midsem_report.html`; rebuild the PDF with `python scripts/build_pdf.py`)
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
| `tests/test_core.py` | unit tests: batched entity scoring, pooling speed-up, DyVo head and gate match the reference maths |
| `dyvo/analysis.py` | builds the result tables, significance tests, Ext-B breakdown and figures |
| `dyvo/prep.py`, `dyvo/significance.py` | stage-0 / evaluation-corpus preparation; every p-value in the report |
| `scripts/reproduce_midsem.sh` | **the exact end-to-end sequence behind the mid-sem results** (calls `run_cpu_budget.sh`) |
| `scripts/restore_artifacts.py` | unpack `artifacts/` to recompute all tables and p-values without training |
| `artifacts/` | **all derived data, per-model metrics (per query), encoded indexes, runs and logs**, see `artifacts/README.md` |
| `scripts/run_squad_experiments.sh` | longer full-budget variant of the pipeline |
| `configs/paper_datasets.md` | how to run the paper's headline experiment on a GPU |

## Quick start

```bash
pip install -r requirements.txt
# downloads: DistilBERT + monoT5-base (legacy HF S3), SQuAD v1.1, Wikipedia2Vec enwiki_20180420_100d
bash scripts/download.sh
python -m dyvo.w2v --src data/w2v/enwiki_20180420_100d.txt.bz2 --out data/w2v/converted
bash scripts/reproduce_midsem.sh             # exactly what produced the mid-sem numbers (~5-6 h, 4 CPU cores)

# or, without re-training: recompute every table / p-value from the committed artifacts
python scripts/restore_artifacts.py --work work/squad --runs work/runs
python -m dyvo.analysis --runs work/runs --work work/squad --w2v data/w2v/converted --out report/results
python -m dyvo.significance --runs work/runs --work work/squad

python -m pytest -q tests                    # unit tests: our implementation == reference maths
```
