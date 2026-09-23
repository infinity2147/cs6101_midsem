#!/usr/bin/env bash
# Full CPU-scale reproduction + extensions on SQuAD-Open-Para (see report/MIDSEM_REPORT.md).
# Total ~5-6 h on a 4-core CPU (bf16/AMX); ~20 min on one GPU.
set -euo pipefail
DATA=${DATA:-/home/user/data}
WORK=${WORK:-/home/user/work/squad}
RUNS=${RUNS:-/home/user/work/runs}
BB=${BB:-$DATA/models/distilbert-base-uncased}
W2V=$DATA/w2v/converted
T=${THREADS:-4}
L1=${L1:-5e-4}
S0_END=${S0_END:-4000}        # triples [0, S0_END) -> stage 0 (LSR-w "pre-training", MS MARCO analogue)
mkdir -p "$WORK" "$RUNS"

# ---- 0. data, BM25, teacher, entity candidates -------------------------------------------
if [ ! -f "$WORK/corpus.jsonl" ]; then
  python -m dyvo.data --squad_dir "$DATA/squad" --out "$WORK"
  head -10000 "$WORK/queries_train.jsonl" > "$WORK/queries_train10k.jsonl"
fi
[ -f "$WORK/run_bm25_test.trec" ] || python -m dyvo.bm25 --corpus "$WORK/corpus.jsonl" --queries "$WORK/queries_test.jsonl" --out "$WORK/run_bm25_test.trec"
[ -f "$WORK/run_bm25_train.trec" ] || python -m dyvo.bm25 --corpus "$WORK/corpus.jsonl" --queries "$WORK/queries_train10k.jsonl" --out "$WORK/run_bm25_train.trec" --k 50
[ -f "$WORK/train_triples_monot5.jsonl" ] || python -m dyvo.teacher --model "$DATA/models/monot5-base" \
  --corpus "$WORK/corpus.jsonl" --queries "$WORK/queries_train10k.jsonl" --qrels "$WORK/qrels_train.txt" \
  --bm25_run "$WORK/run_bm25_train.trec" --out "$WORK/train_triples_monot5.jsonl" --n_queries 10000
INPUTS="$WORK/queries_test.jsonl $WORK/queries_val.jsonl $WORK/queries_train10k.jsonl $WORK/corpus.jsonl"
[ -f "$WORK/corpus.ent_link.jsonl" ]  || python -m dyvo.linker --w2v "$W2V" --method link  --inputs $INPUTS
[ -f "$WORK/corpus.ent_dense.jsonl" ] || python -m dyvo.linker --w2v "$W2V" --method dense --k 30 --inputs $INPUTS

TRAIN="python -m dyvo.train --data $WORK --triples $WORK/train_triples_monot5.jsonl --backbone $BB --w2v $W2V --threads $T --l1_d $L1"
EVAL="python -m dyvo.evaluate --threads $T"

# ---- 1. stage 0: LSR-w initialisation (paper: DyVo & LSR-w start from an MS MARCO LSR) ---------
[ -f "$RUNS/lsr_init/model.pt" ] || $TRAIN --out "$RUNS/lsr_init" --triple_end $S0_END --lambda_ib 1.0 --lr 3e-5

# ---- 2. stage 1: all variants from the same init, same triples, same steps, KL only (paper) ----
S1="--init $RUNS/lsr_init --triple_start $S0_END"
run() { name=$1; shift; [ -f "$RUNS/$name/model.pt" ] || $TRAIN --out "$RUNS/$name" $S1 "$@";
        [ -f "$RUNS/$name/metrics_test.json" ] || $EVAL --model_dir "$RUNS/$name"; }

run lsr_w                                                        # Table 1: LSR-w
run dyvo_link_w2v       --cands link                             # Table 1: DyVo (linked entities)
run dyvo_linkdense_w2v  --cands link+dense                       # Table 2: noisy recall-oriented cands
run dyvo_linkdense_gate --cands link+dense --gate --noise_ents 4 # Ext-A: noise-robust gating
run dyvo_link_learned   --cands link --ent_emb_mode learned      # Ext-B control: static learned vocab
run dyvo_link_tokaggr   --cands link --ent_emb tokaggr           # Table 3: entity embedding ablation

# ---- 3. test-time analyses (no re-training) ------------------------------------------------
# Ext-A: inject more noise at test time (dense k=30 per query/doc)
for m in dyvo_linkdense_w2v dyvo_linkdense_gate; do
  $EVAL --model_dir "$RUNS/$m" --cands link+dense --k_dense_q 30 --k_dense_d 30
done
python -m dyvo.analysis --runs "$RUNS" --work "$WORK" --out report/results
