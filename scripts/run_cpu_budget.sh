#!/usr/bin/env bash
# Budget version of run_squad_experiments.sh actually executed for the mid-sem report on a
# 4-core CPU *without* bf16 (fp32): 300-step stage-0, 300-step stage-1 per variant, d_len 192,
# 8k-paragraph evaluation corpus. Resumable: finished steps are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-$PWD/data}; WORK=${WORK:-$PWD/work/squad}; RUNS=${RUNS:-$PWD/work/runs}
BB=$DATA/models/distilbert-base-uncased
COMMON="--data $WORK --w2v $DATA/w2v/converted --backbone $BB --d_len 192 --l1_d 5e-4 --batch_size 8"
log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ ! -f $RUNS/lsr_init/model.pt ]; then
  log "stage 0"
  python -m dyvo.train $COMMON --triples $WORK/train_triples_s0_hard.jsonl --train_queries queries_s0 \
    --out $RUNS/lsr_init --lambda_ib 1.0 --lr 3e-5 --triple_end 2400 --threads ${S0_THREADS:-2} || exit 1
fi
S1="$COMMON --triples $WORK/train_triples_monot5.jsonl --init $RUNS/lsr_init --triple_end 2400 --threads 4"
run() { name=$1; shift
  [ -f $RUNS/$name/model.pt ] || { log "train $name"; python -m dyvo.train $S1 --out $RUNS/$name "$@" || return 1; }
  [ -f $RUNS/$name/metrics_test.json ] || { log "eval $name"; python -m dyvo.evaluate --model_dir $RUNS/$name --threads 4; } ; }
[ -f $RUNS/lsr_init/metrics_test.json ] || python -m dyvo.evaluate --model_dir $RUNS/lsr_init --threads 4
run lsr_w
run dyvo_link_w2v       --cands link
run dyvo_linkdense_w2v  --cands link+dense
run dyvo_linkdense_gate --cands link+dense --gate --noise_ents 4
for m in dyvo_linkdense_w2v dyvo_linkdense_gate; do
  [ -f $RUNS/$m/metrics_test_link+dense_q30_d30.json ] || python -m dyvo.evaluate --model_dir $RUNS/$m --cands link+dense --k_dense_q 30 --k_dense_d 30
done
python -m dyvo.analysis --runs $RUNS --work $WORK --w2v $DATA/w2v/converted --out report/results
log "CORE DONE"
# optional extras if time permits
run dyvo_link_learned   --cands link --ent_emb_mode learned
python -m dyvo.analysis --runs $RUNS --work $WORK --w2v $DATA/w2v/converted --out report/results
log "ALL DONE"
