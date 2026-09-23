#!/usr/bin/env bash
# The exact sequence that produced every number in report/ (mid-sem), end to end.
# Paths are configurable: DATA (downloads), WORK (derived data), RUNS (checkpoints + outputs).
# Runtime: ~5-6 h on a 4-core CPU (fp32); bf16 is used automatically on CPUs with AMX/AVX512-BF16.
#
# Every derived file this script writes is also committed under artifacts/, so any step can be
# skipped by copying from there (see artifacts/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
export DATA=${DATA:-$PWD/data} WORK=${WORK:-$PWD/work/squad} RUNS=${RUNS:-$PWD/work/runs}
mkdir -p "$WORK" "$RUNS"
W2V=$DATA/w2v/converted
log() { echo "[$(date +%H:%M:%S)] $*"; }

# 1. downloads + Wikipedia2Vec conversion (2.59M entities, 400k words)
[ -f "$DATA/squad/dev-v1.1.json" ] || DATA=$DATA bash scripts/download.sh
[ -f "$W2V/entity_vecs.npy" ] || python -m dyvo.w2v --src "$DATA/w2v/enwiki_20180420_100d.txt.bz2" --out "$W2V"

# 2. SQuAD-Open-Para benchmark, BM25, stage-0 data, evaluation corpus
if [ ! -f "$WORK/corpus.jsonl" ]; then
  python -m dyvo.data --squad_dir "$DATA/squad" --out "$WORK"          # 20,963 paragraphs; 2,000 test q
  head -10000 "$WORK/queries_train.jsonl" > "$WORK/queries_train10k.jsonl"
fi
[ -f "$WORK/run_bm25_train.trec" ] || python -m dyvo.bm25 --corpus "$WORK/corpus.jsonl" \
    --queries "$WORK/queries_train10k.jsonl" --out "$WORK/run_bm25_train.trec" --k 50
[ -f "$WORK/train_triples_s0_hard.jsonl" ] || python -m dyvo.prep stage0 --work "$WORK"
[ -f "$WORK/corpus_test.jsonl" ] || python -m dyvo.prep eval_corpus --work "$WORK"   # + BM25 test run

# 3. entity candidates (linker = REL analogue, dense = LaQue analogue)
INPUTS="$WORK/queries_test.jsonl $WORK/queries_val.jsonl $WORK/queries_train10k.jsonl $WORK/corpus.jsonl"
[ -f "$WORK/corpus.ent_link.jsonl" ]  || python -m dyvo.linker --w2v "$W2V" --method link  --inputs $INPUTS
[ -f "$WORK/corpus.ent_dense.jsonl" ] || python -m dyvo.linker --w2v "$W2V" --method dense --k 30 --inputs $INPUTS

# 4. monoT5 teacher scores. What was actually run (the container restarted mid-way):
#    (a) length-sorted scoring of the 9,990 (q, d+, BM25 d-) triples, interrupted after 3,792;
#    (b) --max_new 1400: a random (seed 0) subset of the remaining triples -> 5,192 in total.
#    The resulting file is artifacts/data/train_triples_monot5.jsonl.gz; copy it to reuse exactly.
T=$WORK/train_triples_monot5.jsonl
if [ ! -f "$T" ]; then
  if [ -f artifacts/data/train_triples_monot5.jsonl.gz ]; then gunzip -c artifacts/data/train_triples_monot5.jsonl.gz > "$T"
  else
    TEACH="python -m dyvo.teacher --model $DATA/models/monot5-base --corpus $WORK/corpus.jsonl \
      --queries $WORK/queries_train.jsonl --qrels $WORK/qrels_train.txt --bm25_run $WORK/run_bm25_train.trec \
      --out $T --n_queries 10000 --batch_size 32"
    # approximation of (a)+(b) when the committed file is not used (same count, random subset)
    $TEACH --max_new 3792 && $TEACH --max_new 1400
  fi
fi

# 5. training + evaluation of all variants, analysis tables and figure
WORK=$WORK RUNS=$RUNS DATA=$DATA bash scripts/run_cpu_budget.sh
python -m dyvo.significance --runs "$RUNS" --work "$WORK" --out report/results/significance.json
log "done: report/results/{tables.md,significance.json,where_dyvo_helps.png}"
