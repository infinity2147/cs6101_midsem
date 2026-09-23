# Artifacts: every derived file behind the mid-sem results

Everything here was produced by the code in `dyvo/` and `scripts/`, in the order given in
`scripts/reproduce_midsem.sh`. All `.gz` files are gzip-compressed and can be read with `gunzip -c`.

## `data/`: derived datasets (inputs to training and evaluation)
| File | Produced by | Content |
|---|---|---|
| `corpus.jsonl.gz` | `dyvo.data` | 20,963 SQuAD v1.1 paragraphs (`id`, `title`, `text`); ids `t<art>_<para>` = train, `d…` = dev |
| `corpus_test.jsonl.gz` | `dyvo.prep eval_corpus` | evaluation index: 2,067 dev paragraphs + 6,000 distractors (seed 7) |
| `queries_{train,train10k,s0,val,test}.jsonl.gz`, `qrels_*.txt.gz` | `dyvo.data`, `dyvo.prep stage0` | queries and TREC qrels (one relevant paragraph each) |
| `*.ent_link.jsonl.gz` | `dyvo.linker --method link` | linked entities per text (Wikipedia2Vec row ids, scores, mentions) |
| `*.ent_dense.jsonl.gz` | `dyvo.linker --method dense --k 30` | top-30 dense entity candidates per text |
| `train_triples_monot5.jsonl.gz` | `dyvo.teacher` | **5,192** (q, d+, d−) triples with monoT5-base scores (stage 1 uses the first 2,400 after a seeded shuffle) |
| `train_triples_s0_hard.jsonl.gz` | `dyvo.prep stage0` | 3,994 hard-label triples for stage 0 (first 2,400 used) |
| `run_bm25_{test,s0,train}.trec.gz` | `dyvo.bm25` | BM25 runs (test: top-1000 on the 8,067-doc eval index; train/s0: top-50 for negatives) |

Wikipedia2Vec row ids index `entities.txt`, which `python -m dyvo.w2v` creates from
`enwiki_20180420_100d.txt.bz2` (rank order = frequency order).

## `runs/<model>/`: one folder per trained model
| File | Content |
|---|---|
| `args.json` | exact training arguments (paths point to the machine that ran it) |
| `train_log.json` | per-step loss, KL, L1, #non-zeros (words / entities), pairwise accuracy, α |
| `metrics_test*.json` | aggregate metrics, **per-query** metrics (all 2,000 queries, incl. R@1000), sparsity/FLOPs stats, latency |
| `reps_docs*.npz`, `reps_queries_test*.npz` | the encoded sparse index and queries (scipy CSR, columns 0-30521 = word pieces, 30522+ = entities). The Ext-B static-vocabulary result is re-scored from these without re-encoding |
| `run_test*.top100.trec.gz` | ranked runs, top-100 per query (the full top-1000 runs were ~20 MB each compressed; per-query R@1000 is kept in `metrics_test.json`) |

Models: `lsr_init` (stage 0), `lsr_w`, `dyvo_link_w2v`, `dyvo_linkdense_w2v`, `dyvo_linkdense_gate` (Ext-A),
`dyvo_link_learned` (Ext-B control). The `_link+dense_q30_d30` files are the 3× test-time-noise evaluations.

Recompute every table, the figure and all p-values from these files alone:
```bash
python scripts/restore_artifacts.py --work work/squad --runs work/runs   # un-gzips into place
python -m dyvo.analysis --runs work/runs --work work/squad --w2v data/w2v/converted --out report/results
python -m dyvo.significance --runs work/runs --work work/squad
```
(`analysis` needs the converted Wikipedia2Vec only for the candidate-quality table.)

## `logs/`
Raw logs of the actual runs: `pipeline.log` (stage 0 through the last evaluation, with timestamps),
teacher, linker, Wikipedia2Vec conversion, and the first stage-0 attempt that was killed when the
container restarted.

## Not included: model checkpoints
Each fine-tuned checkpoint (`model.pt`) is about 510 MB, over GitHub's 100 MB per-file limit, and
all six together are about 3 GB. They are fully determined by `args.json` plus the data above (seed 42), and
the saved `reps_*.npz` hold their encoded outputs, so every reported number can be recomputed
without them.
