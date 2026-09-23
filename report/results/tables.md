### Headline reproduction (paper Table 1 analogue)

| Model | nDCG@10 | nDCG@20 | R@100 | R@1k | MRR@10 | doc words/ents \| q words/ents \| FLOPs |
|---|---|---|---|---|---|---|
| BM25 | 84.25 | 84.81 | 97.95 | 99.15 | 81.47 | – |
| LSR-w stage-0 init (no distillation) | 80.43† | 81.22 | 97.60 | 99.25 | 76.81 | 156 / 0.0 | 11.2 / 0.0 | 4.38 |
| LSR-w | 78.41 | 79.34 | 97.35 | 98.85 | 74.82 | 98 / 0.0 | 9.3 / 0.0 | 2.14 |
| DyVo (linked entities, Wikipedia2Vec) | 77.75 | 78.73 | 97.65 | 99.10 | 73.90 | 102 / 7.4 | 10.6 / 0.7 | 3.21 |

### Candidate source & Ext-A noise robustness (paper Table 2 analogue)

| Model | nDCG@10 | nDCG@20 | R@100 | R@1k | MRR@10 | doc words/ents \| q words/ents \| FLOPs |
|---|---|---|---|---|---|---|
| DyVo – linked (precision) | 77.75 | 78.73 | 97.65 | 99.10 | 73.90 | 102 / 7.4 | 10.6 / 0.7 | 3.21 |
| DyVo – link ∪ dense top-10/20 (recall, noisy) | 76.40‡ | 77.31 | 96.90 | 98.80 | 72.36 | 88 / 10.3 | 9.5 / 3.1 | 2.22 |
|   … same model, 3x test-time noise (top-30/30) | 76.44‡ | 77.35 | 96.90 | 98.80 | 72.41 | 88 / 14.0 | 9.5 / 9.6 | 2.31 |
| DyVo-Gate (Ext-A) – link ∪ dense | 78.48 | 79.27 | 97.65 | 98.90 | 74.61 | 110 / 0.2 | 9.0 / 10.6 | 2.02 |
|   … same model, 3x test-time noise (top-30/30) | 78.48 | 79.27 | 97.65 | 98.90 | 74.61 | 110 / 0.2 | 9.0 / 30.4 | 2.02 |

### Entity embeddings (paper Table 3 analogue)

| Model | nDCG@10 | nDCG@20 | R@100 | R@1k | MRR@10 | doc words/ents \| q words/ents \| FLOPs |
|---|---|---|---|---|---|---|
| Wikipedia2Vec (100d, frozen, projected) | 77.75 | 78.73 | 97.65 | 99.10 | 73.90 | 102 / 7.4 | 10.6 / 0.7 | 3.21 |

### Candidate quality

| Candidate source (test queries) | avg #cands | gold-article-entity recall |
|---|---|---|
| WikiLinker (REL analogue) | 0.7 | 16.9% |
| Dense W2V top-10 (LaQue analogue) | 10.0 | 4.8% |
| Dense W2V top-30 | 30.0 | 7.6% |
| Linker ∪ dense top-10 | 10.7 | 20.0% |

### Linking consistency (nDCG@10 by query group)

| Query group | #q | LSR-w | DyVo (link) | DyVo (link+dense) | DyVo-Gate |
|---|---|---|---|---|---|
| no linked query entity | 900 | 73.12 | 72.09 | 71.23 | 73.91 |
| query entity linked, absent from gold doc | 174 | 74.44 | 65.94 | 72.51 | 76.21 |
| query entity also linked in gold doc | 926 | 84.30 | 85.46 | 82.16 | 83.34 |

### Ext-B: dynamic vocabulary and unseen entities

| Query group | #q | LSR-w | DyVo (frozen W2V, dynamic) | DyVo (static learned table) |
|---|---|---|---|---|
| no entity linked | 900 | 73.12 | 72.09 | – |
| all entities seen in training | 646 | 80.24 | 78.33 | – |
| >=1 unseen entity | 454 | 86.29 | 88.13 | – |

Test-time vocabulary restriction of the same DyVo model (entities never seen in training removed from the index = a static vocabulary):

| Vocabulary | all queries nDCG@10 | >=1-unseen-entity queries nDCG@10 |
|---|---|---|
| dynamic (all Wikipedia entities) | 77.75 | 88.13 |
| static (training entities only) | 77.30 | 86.16 |

On >=1-unseen-entity queries: dynamic vs static p=0.0017; DyVo vs LSR-w p=0.0094 (paired t-test).

16686 distinct entities seen in training; 420 distinct test-query entities unseen.

† / ‡ : significantly better / worse than LSR-w (paired t-test on nDCG@10, p<0.05).