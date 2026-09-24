# DyVo headline reproduction: LSR-w vs DyVo (REL + LaQue), L1 = 1e-5

**Status: BLOCKED. None of the six headline runs could be trained on this machine.**
This report records exactly what was done, what is ready, what is missing, and the evidence for each
blocker. Every claim below points to a log in `logs/headline/`. No number in this report is a reproduced
retrieval result: the "ours" columns are empty on purpose.

Scope followed: prompt `PROMPT - DyVo Headline Reproduction Stage` (priority1 + priority2 only; no extensions,
no LLM calls, no Table 2/3, no L1 sweep). The separate SQuAD re-implementation (`infinity2147/cs6101_midsem`) is
**not** used as a substitute for the paper's datasets (prompt §20.10).

---

## 1. Exact environment

| Item | Value | Evidence |
|---|---|---|
| Repository | `Parav29/DyVo-CS6101` @ `158b65b` (`master`), clean working tree before this stage | `git rev-parse HEAD` |
| OS | Ubuntu 24.04.4 LTS, Linux 6.18 x86-64 (cloud container) | `uname`, `/etc/os-release` |
| **GPU** | **none**: no `nvidia-smi`, no CUDA driver or toolkit | Step 0 inventory |
| CPU | 4 × Intel Xeon @ 2.1 GHz | `lscpu` |
| **RAM** | **16.9 GB** (15.7 GiB) | `free` |
| **Free disk** | **~9–12 GB** (252 GB volume, shared) | `df -h /` |
| Python | 3.10.20 (`.venv`, same minor version as the build machine) | `.venv/bin/python --version` |
| Packages | torch 2.2.2 (+cu121 wheel from PyPI, running on CPU), transformers 4.40.2, hydra 1.3.2, ir-datasets 0.6.3, ir-measures 0.4.3, wandb 0.16.6: the lock-file pins | `logs/headline/cpu_venv_install.log` |
| CUDA | not available (`torch.cuda.is_available() == False`) | |
| Network | Egress policy blocks `huggingface.co`, `*.hf.co`, `hf-mirror.com`, `download.pytorch.org`, `trec.nist.gov`, `mirror.ir-datasets.com`. PyPI and `raw.githubusercontent.com` are reachable | `logs/headline/hf_reachability.log`, `03_irds_access.log`, `00_setup_env_real.log` |

### Dataset / asset availability

| Resource | Needed by | Status | Evidence |
|---|---|---|---|
| `lsr42/dyvo_data`: `dyvo_init/*/model.safetensors` (MS MARCO LSR init) | all 6 runs | **missing**: HF hub blocked (403) | `hf_reachability.log` |
| `dyvo_data/knowledge_base/entity_embs_laque.pt` (16.2 GB) | 3 DyVo runs | **missing**: HF blocked; would not fit (16.2 GB > free disk; ≈ total RAM) | `01_download_dryrun.log` |
| MonoT5-3B scores, InPars-v2 / Mixtral training topics and qrels (`*_monot5_3b_scores.json`, `*_topics.tsv`, `*_qrels.json`) | all 6 runs | **missing**: HF blocked | `02_check_data_priority{1,2}.log` |
| REL query candidates `*/queries_test_train_entities_rel.jsonl` | DyVo runs | **present** (downloaded on the build machine, in git) | `02_check_data_priority1.log` |
| REL document candidates `*/docs_entities.jsonl` (123 / 431 / 516 MB) | DyVo runs | **missing**: HF blocked | same |
| CODEC queries + qrels (`ir_datasets` `codec`) | priority1 | **available** (GitHub) | `03_irds_access.log` |
| CODEC documents (`comets_documents.jsonl`, 729k docs) | priority1 | **missing**: distributed on request by the CODEC authors | `03_irds_access.log` |
| Robust04 topics, qrels, TREC Disks 4 & 5 | priority2 | **missing**: NIST blocked; corpus is licensed | `03_irds_access.log` |
| Core 2018 topics, qrels, Washington Post v2 | priority2 | **missing**: NIST blocked; corpus is licensed | `03_irds_access.log` |

---

## 2. Exact experiment configuration (verified, not assumed)

The six runs were composed with the real entry point and the wrapper's overrides
(`python -m lsr.train +experiment=<name> training_arguments.fp16=True training_arguments.per_device_train_batch_size=16 --cfg job`).
The fully resolved configs are saved in `logs/headline/resolved_configs/`; the table is extracted from them
(`logs/headline/resolved_settings.md`).

| Run | Batch | LR | Steps | Warm-up | fp16 | q / d len | Seed | L1 doc / query | `reg_ent` | Loss norm | λ_ent q / d | Entity table | Query / doc candidates | Init | Training data |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LSR-w CODEC | 16 | 5e-7 | 100k | 0.1 | ✓ | 50 / 512 | 42 | 1e-5 / 0 | – | std | – | – | – | dyvo_init safetensors | `codec_mistral_topics.tsv` + `codec_monot5_3b_scores.json` |
| DyVo CODEC | 16 | 5e-7 | 100k | 0.1 | ✓ | 50 / 512 | 42 | 1e-5 / 0 | False | std | 0.05 / 0.05 | `entity_embs_laque.pt` (768-d) | REL / `docs_entities.jsonl` | same | same |
| LSR-w Robust04 | 16 | 5e-7 | 100k | 0.1 | ✓ | 50 / 512 | 42 | 1e-5 / 0 | – | std | – | – | – | same (see fix F3) | `robust04_inparsv2_topics.tsv` + `robust04_monot5_3b_scores.json` |
| DyVo Robust04 | 16 | 5e-7 | 100k | 0.1 | ✓ | 50 / 512 | 42 | 1e-5 / 0 | False | std | 0.05 / 0.05 | LaQue | REL / `docs_entities.jsonl` | same | same |
| LSR-w Core18 | 16 | 5e-7 | 100k | 0.1 | ✓ | 50 / 512 | 42 | 1e-5 / 0 | – | std | – | – | – | same (see fix F3) | `wapo_inparsev2_topics.tsv` + `wapo_monot5_3b_scores.json` |
| DyVo Core18 | 16 | 5e-7 | 100k | 0.1 | ✓ | 50 / 512 | 42 | 1e-5 / 0 | False | std | 0.05 / 0.05 | LaQue | REL / `docs_entities.jsonl` | same | same |

All runs use backbone `distilbert-base-uncased`, an MLP query encoder, an MLM (SPLADE) document encoder, frozen
entity embeddings, and the released MonoT5-3B teacher. The test set is the **full collection** for all six
(`test_dataset.num_documents = -1`), with the paper's query fields (`query` for CODEC, `description` for
Robust04/Core18). The in-training eval (`num_documents = 100000`, qrels documents only) is used only for
monitoring and must not be reported. Metrics: nDCG@10, nDCG@20, R@1000. This matches the prompt's paper-faithful
list (§7) item by item.

Release behaviours kept on purpose (NOTES.md; prompt §8): λ_ent inside the ReLU; no attention mask in the entity
max-pool; the DyVo collator truncates queries at `d_max_length`; `normalize: "std"` = L2-normalized teacher scores.

---

## 3. Paper vs reproduced results

`results/results.csv` (from `scripts/big_pc/04_collect_results.py`): **0 / 46 experiments finished; all six
headline rows are `not run`.**

| Dataset | Model | Paper nDCG@10 | Ours | Δ | Paper nDCG@20 | Ours | Δ | Paper R@1000 | Ours | Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CODEC | LSR-w | 52.61 | not run | – | 49.22 | not run | – | 69.07 | not run | – |
| CODEC | DyVo (REL) | 53.40 | not run | – | 51.15 | not run | – | 70.60 | not run | – |
| Robust04 | LSR-w | 49.13 | not run | – | 46.34 | not run | – | 66.86 | not run | – |
| Robust04 | DyVo (REL) | 51.19 | not run | – | 47.65 | not run | – | 68.56 | not run | – |
| Core18 | LSR-w | 40.99 | not run | – | 38.73 | not run | – | 63.22 | not run | – |
| Core18 | DyVo (REL) | 43.72 | not run | – | 40.56 | not run | – | 63.56 | not run | – |

(Paper numbers are read from `scripts/big_pc/experiments.py::PAPER`, not re-typed.)

## 4. Main-effect comparison (DyVo − LSR-w)

| Dataset | Paper Δ nDCG@10 | Paper Δ nDCG@20 | Paper Δ R@1000 | Our Δ | Interpretation |
|---|---:|---:|---:|---|---|
| CODEC | +0.79 | +1.93 | +1.53 | not run | blocked by unavailable data (and no GPU) |
| Robust04 | +2.06 | +1.31 | +1.70 | not run | blocked by unavailable data (and no GPU) |
| Core18 | +2.73 | +1.83 | +0.34 | not run | blocked by unavailable data (and no GPU) |

Note for whoever runs it: the paper's CODEC nDCG@10 gain (+0.79) is small. With only 42 CODEC test queries, a
single-seed run could land on either side of it. Report the per-query paired difference (and ideally a second seed)
before calling the CODEC direction reproduced or not.

## 5. Discrepancy analysis

There are no reproduced numbers to compare, so there is nothing to diagnose on the retrieval side. The
discrepancies found are **pipeline defects that would have broken or silently corrupted the runs**. All were fixed
minimally and marked `DYVO-REPRO`:

| # | Found | Paper description | Released code / repo behaviour | Fix (this stage) | Evidence |
|---|---|---|---|---|---|
| F1 | `requirements-lock.txt` cannot be installed on Linux: `googleapis-common-protos==1.75.3` requires `protobuf>=6.33.5` and `opentelemetry-*==1.44.0` require `protobuf>=5`, but the lock pins `protobuf==4.25.9` (needed by `wandb 0.16.6`). **`00_setup_env.sh` would fail the same way on the GPU machine.** | – | inconsistent `pip freeze` from the build machine | removed the 7 unused pins (commented in the lock). The remaining 80 pins install and `pip check` reports *No broken requirements*, so nothing in the project needs them | `logs/headline/cpu_venv_install.log` |
| F2 | `02_check_data.py` reported `ok ir_datasets:codec has local data` and `ok …wapo…` when only topics/qrels were present: it tested "folder not empty". A missing corpus was also only a WARN | – | – | the check now reads one real document and raises an **ERROR** if it can't. CODEC and Core18 now correctly FAIL here | `logs/headline/02_check_data_priority{1,2}.log` |
| F3 | LSR-w Robust04 and Core18 (L1 1e-5) configs load `dyvo_init/*/pytorch_model.bin`, but the release contains **only** `model.safetensors` (full repo listing `logs/hf_listing.tsv`). These two headline runs would crash at start | paper: every model starts from the same MS MARCO LSR | CODEC LSR-w and all DyVo configs already use `model.safetensors` | the two headline configs now point to `model.safetensors`. `DualSparseEncoder` loads either format, and `tests/test_checkpoint_load.py` already shows the LSR-w encoders load these files. The 4 priority3 LSR-w configs (L1 1e-3/1e-4, Robust04/Core18) have the same issue and were left for that stage | `logs/headline/02_check_data_priority2.log` (the `.bin` FAILs are gone) |

After the fixes, `bash scripts/verify.sh` stays green on Linux: 73/73 configs compose, 61 tests pass, the
overfit check passes, and both smoke runs complete (`logs/headline/verify_after_fixes.log`).

## 6. Reproduction verdict

**Blocked by unavailable data and hardware: not reproduced, not refuted.** Nothing was trained, so this stage
makes no claim about whether DyVo improves over LSR-w.

The blockers, all external to the code:
1. **Released assets** (`lsr42/dyvo_data`: init checkpoints, LaQue table, teacher scores, training topics,
   document entities) can't be downloaded: HF is blocked by this environment's network policy.
2. **Corpora**: CODEC documents are on request; Robust04 (TREC Disks 4 & 5) and Washington Post v2 are licensed
   NIST data, and NIST itself is unreachable from here.
3. **Hardware**: no GPU. RAM (16.9 GB) is about equal to the LaQue table alone (16.2 GB), before the model,
   optimizer and corpus text. Free disk (~9–12 GB) is below the 18.1 GB asset download. The handoff estimates
   ~8–15 A100-hours per run. Reducing batch size, document length, steps, corpus size or model to fit would be a
   different experiment (prompt §9, §20.4), so it was not done.

## 7. What is ready, and the exact path to finish

Ready and verified: the code (`verify.sh` green on Linux), the six resolved configs, the corrected lock file, the
data checker (which now catches missing corpora), the priority runner, the results collector, and the REL query
candidates.

On a machine with an A100-class GPU (≥ 24 GB), ≥ 32 GB RAM (64 GB recommended), ≥ 30 GB free disk, HF access, the
CODEC documents and the licensed NIST corpora:
```bash
bash scripts/big_pc/00_setup_env.sh && source .venv-gpu/bin/activate     # works now that F1 is fixed
bash scripts/verify.sh
python scripts/big_pc/01_list_and_download_data.py --datasets codec,robust04,wapo --embeddings laque
# place comets_documents.jsonl under ~/.ir_datasets/codec/v1/ (and Disks 4&5 / WaPo v2 for priority2)
python scripts/big_pc/02_check_data.py --experiments priority1                   # must end with 0 errors
bash scripts/big_pc/run_priority.sh --groups priority1
python scripts/big_pc/04_collect_results.py --recompute
python scripts/big_pc/02_check_data.py --experiments priority2 && bash scripts/big_pc/run_priority.sh --groups priority2
python scripts/big_pc/04_collect_results.py --recompute
```
Then fill sections 3–6 of this report from `results/results.csv`, checking each log against the list in prompt §12
(init loaded without unexpected missing keys, no NaN, `query/doc #entities` > 0, batch 16, lr 5e-7, 100k steps,
d_len 512, correct candidate and LaQue paths, full-corpus test).

## 8. Files produced in this stage
- `HEADLINE_REPRODUCTION_REPORT.md` (this file)
- `results/results.csv`: collector output, all headline rows `not run`
- `logs/headline/`: environment setup (dry-run and real), CPU venv install, verify runs (before and after fixes),
  download dry-run, HF reachability, `ir_datasets` access test, data checks for priority1/2, priority dry-runs,
  collector log, `resolved_configs/*.yaml` and `resolved_settings.md`
- Code and config changes (all marked `DYVO-REPRO`): `requirements-lock.txt` (F1), `scripts/big_pc/02_check_data.py` (F2),
  two LSR-w headline configs (F3)
