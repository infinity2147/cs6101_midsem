"""Correctness checks for the parts where we deviate in *form* (not maths) from the reference.
run: python -m pytest -q tests   (no pretrained weights needed)"""
import numpy as np
import torch
from scipy import sparse

from dyvo import WORD_VOCAB_SIZE
from dyvo.evaluate import mask_entities
from dyvo.metrics import evaluate
from dyvo.model import DyVoHead, entity_scores_all


def test_pool_then_activate_equals_reference():
    """log1p(relu) is monotone -> max_t f(x_t) == f(max_t x_t) (our MLM speed-up)."""
    x = torch.randn(3, 17, 50)
    ref = torch.log1p(torch.relu(x)).max(1).values
    ours = torch.log1p(torch.relu(x.max(1).values))
    assert torch.allclose(ref, ours)


def test_entity_scores_match_reference_sparse_dot():
    """batched version == reference lsr.losses.entity_distil_kl_loss.sparse_dot_product."""
    torch.manual_seed(0)
    qi, di = torch.randint(0, 6, (4, 5)), torch.randint(0, 6, (7, 9))
    qw, dw = torch.rand(4, 5), torch.rand(7, 9)
    ours = entity_scores_all(qi, qw, di, dw)
    for i in range(4):
        for j in range(7):
            ref = ((qi[i][:, None] == di[j][None, :]).float() * qw[i][:, None] * dw[j][None, :]).sum()
            assert torch.allclose(ours[i, j], ref, atol=1e-5)


def test_dyvo_head_matches_formula_and_masks():
    torch.manual_seed(0)
    E = torch.randn(10, 8)
    head = DyVoHead(hidden=8, ent_emb=E, entity_weight=0.5)   # d == hidden -> no projection
    h = torch.randn(2, 6, 8)
    attn = torch.ones(2, 6)
    attn[1, 4:] = 0
    ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[1., 1, 1], [1, 1, 0]])
    w = head(h, attn, ids, mask)
    for b in range(2):
        for k in range(3):
            L = int(attn[b].sum())
            ref = torch.log1p(torch.relu(0.5 * (h[b, :L] @ E[ids[b, k]]))).max() * mask[b, k]
            assert torch.allclose(w[b, k], ref, atol=1e-5)


def test_gate_starts_open_and_is_bounded():
    head = DyVoHead(hidden=8, ent_emb=torch.randn(5, 8), gate=True)
    h, attn = torch.randn(1, 4, 8), torch.ones(1, 4)
    ids, mask = torch.tensor([[0, 1]]), torch.ones(1, 2)
    feats = torch.rand(1, 2, 4)
    g = head(h, attn, ids, mask, feats)
    head.gate = None
    ng = head(h, attn, ids, mask)
    ratio = (g / ng.clamp_min(1e-9))[ng > 0]
    assert torch.allclose(ratio, torch.full_like(ratio, torch.sigmoid(torch.tensor(2.0)).item()), atol=1e-5)


def test_mask_entities_keeps_words():
    M = sparse.csr_matrix(np.array([[1, 0, 2, 3]], dtype=np.float32))
    M.resize((1, WORD_VOCAB_SIZE + 2))
    M = sparse.csr_matrix((np.array([1., 2., 3.]), (np.zeros(3), [5, WORD_VOCAB_SIZE, WORD_VOCAB_SIZE + 1])),
                          shape=(1, WORD_VOCAB_SIZE + 2))
    out = mask_entities(M, np.array([WORD_VOCAB_SIZE + 1]))
    assert out[0, 5] == 1 and out[0, WORD_VOCAB_SIZE] == 0 and out[0, WORD_VOCAB_SIZE + 1] == 3


def test_metrics():
    run = {"q": {"a": 3.0, "b": 2.0, "c": 1.0}}
    qrels = {"q": {"b": 1}}
    r = evaluate(run, qrels)
    assert abs(r["RR@10"] - 50) < 1e-9 and abs(r["nDCG@10"] - 100 / np.log2(3)) < 1e-9
