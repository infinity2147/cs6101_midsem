# DyVo headline reproduction stage (for `Parav29/DyVo-CS6101`)

Work done against `Parav29/DyVo-CS6101` @ `158b65b` following the "DyVo Headline Reproduction Stage" prompt
(LSR-w vs DyVo REL + LaQue, L1 = 1e-5, priority1 + priority2). It could not be pushed there (no write access),
so the full change set is kept here.

**Verdict: BLOCKED.** No run could be trained: no GPU, RAM ≈ LaQue table size, and the released `lsr42/dyvo_data`
assets, the CODEC documents and the licensed NIST corpora were unavailable in this environment. Read
`HEADLINE_REPRODUCTION_REPORT.md` for the environment, the verified configs, the evidence and the exact steps to finish.

| File | What |
|---|---|
| `HEADLINE_REPRODUCTION_REPORT.md` | the stage report (a copy of the file added to the DyVo repo) |
| `results.csv` | `04_collect_results.py` output: 0/46 runs finished |
| `0001-…patch` | the commit for `Parav29/DyVo-CS6101` (3 fixes + report + `logs/headline/` evidence, 28 files) |
| `dyvo-headline-reproduction.bundle` | the same commit as a git bundle (branch `claude/dyvo-headline-reproduction`) |

Apply in a clone of `Parav29/DyVo-CS6101`:
```bash
git checkout -b claude/dyvo-headline-reproduction 158b65b
git am /path/to/0001-Headline-reproduction-stage-environment-data-audit-3.patch
# or: git fetch /path/to/dyvo-headline-reproduction.bundle claude/dyvo-headline-reproduction
```
