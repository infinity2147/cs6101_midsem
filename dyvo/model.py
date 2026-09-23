"""DyVo model: LSR encoders + Dynamic Vocabulary (entity) head.

Faithful to the reference implementation (thongnt99/DyVo, lsr/models/{mlp_emlm,mlm_emlm}.py):

* query encoder  ("qmlp")  : w(t) = log1p(relu(Linear(h_t)))            – no expansion
* doc encoder    ("dmlm")  : w(v) = max_t log1p(relu(MLM_logits[t, v]))  – SPLADE-style
* DyVo head (both sides)   : w(e) = max_t log1p(relu(alpha * <h_t, P(E_e)>)), e in candidates(x)
      E_e   frozen external entity embedding (Wikipedia2Vec / token-aggregation / ...)
      P     Linear(ent_dim -> hidden) when dims differ (trainable)
      alpha trainable scalar initialised to 0.05 (config `entity_weight: 0.05`)
* score(q, d) = <w_q, w_d> over the joint vocabulary  [30522 word pieces] ++ [entities]

Differences from the reference (documented in the report):
  - padding positions are masked before the entity max-pool (reference does not mask).
Extensions (off by default):
  - ``gate``: Ext-A noise-robust candidate gating  w(e) <- w(e) * sigmoid(MLP(feat(e)))
  - ``ent_emb_mode="learned"``: Ext-B control – a *static* learned entity table instead of
    frozen external embeddings (what a fixed-vocabulary model would do).
"""
import torch
from torch import nn
from transformers import AutoModel, AutoModelForMaskedLM

from . import WORD_VOCAB_SIZE


class DyVoHead(nn.Module):
    N_FEATS = 4  # [retriever score, from-linker flag, from-dense flag, popularity]

    def __init__(self, hidden, ent_emb, entity_weight=0.05, gate=False, ent_emb_mode="frozen"):
        super().__init__()
        n, d = ent_emb.shape
        if ent_emb_mode == "frozen":
            self.register_buffer("ent_emb", ent_emb.float(), persistent=False)
            self.table = None
        else:  # learned: a static, trainable entity table (random init)
            self.table = nn.Embedding(n, d)
            nn.init.normal_(self.table.weight, std=ent_emb.float().std().item())
        self.proj = nn.Linear(d, hidden) if d != hidden else nn.Identity()
        self.ent_weight = nn.Parameter(torch.tensor(float(entity_weight)))
        self.gate = None
        if gate:
            self.gate = nn.Sequential(nn.Linear(self.N_FEATS + 1, 16), nn.GELU(), nn.Linear(16, 1))
            nn.init.zeros_(self.gate[2].weight)
            nn.init.constant_(self.gate[2].bias, 2.0)  # start (almost) open: sigmoid(2)=0.88

    def embed(self, ent_ids):
        E = self.table(ent_ids) if self.table is not None else self.ent_emb[ent_ids]
        return self.proj(E)

    def forward(self, hidden, attn_mask, ent_ids, ent_mask, ent_feats=None):
        # hidden B x L x H ; ent_ids/ent_mask B x K
        E = self.embed(ent_ids)                                         # B x K x H
        logits = torch.einsum("blh,bkh->blk", hidden, E) * self.ent_weight
        logits = logits.masked_fill(~attn_mask.bool().unsqueeze(-1), -1e4)
        pooled = logits.max(dim=1).values                               # B x K
        w = torch.log1p(torch.relu(pooled))
        if self.gate is not None:
            feats = torch.cat([ent_feats, pooled.detach().unsqueeze(-1)], dim=-1)
            w = w * torch.sigmoid(self.gate(feats).squeeze(-1))
        return w * ent_mask


class SparseEncoder(nn.Module):
    """kind='mlp' (query side, no expansion) or 'mlm' (doc side, SPLADE max-pooling)."""

    def __init__(self, backbone, kind, ent_emb=None, **head_kw):
        super().__init__()
        self.kind = kind
        if kind == "mlm":
            self.lm = AutoModelForMaskedLM.from_pretrained(backbone)
        else:
            self.lm = AutoModel.from_pretrained(backbone)
            self.linear = nn.Linear(self.lm.config.hidden_size, 1)
        self.head = DyVoHead(self.lm.config.hidden_size, ent_emb, **head_kw) if ent_emb is not None else None

    def forward(self, input_ids, attention_mask, special_tokens_mask, ent_ids=None, ent_mask=None,
                ent_feats=None):
        keep = attention_mask * (1 - special_tokens_mask)
        if self.kind == "mlm":
            out = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = out.hidden_states[-1]
            # log1p(relu(.)) is monotone, so max_t log1p(relu(x_t)) == log1p(relu(max_t x_t)):
            # pool first, then activate -> avoids 3 elementwise passes over the B x L x V tensor
            logits = out.logits.masked_fill(~keep.bool().unsqueeze(-1), -1e4)
            word = torch.log1p(torch.relu(logits.max(dim=1).values.float()))  # B x V
        else:
            hidden = self.lm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            tw = torch.log1p(torch.relu(self.linear(hidden).squeeze(-1).float())) * keep
            word = torch.zeros(input_ids.size(0), WORD_VOCAB_SIZE, device=tw.device, dtype=tw.dtype)
            word = word.scatter_add(1, input_ids, tw)                     # duplicates add up
        ent = None
        if self.head is not None and ent_ids is not None:
            ent = self.head(hidden, attention_mask, ent_ids, ent_mask, ent_feats).float()
        return word, ent


def entity_scores_all(q_ids, q_w, d_ids, d_w):
    """entity part of <q_i, d_j> for all pairs: sum_{k,l} [q_ids[i,k] == d_ids[j,l]] q_w[i,k] d_w[j,l]
    (the reference's `sparse_dot_product`, batched). q_*: Bq x Kq, d_*: Bd x Kd; padding weight 0."""
    match = (q_ids[:, None, :, None] == d_ids[None, :, None, :]).to(q_w.dtype)
    return torch.einsum("ik,ijkl,jl->ij", q_w, match, d_w)


class DyVo(nn.Module):
    def __init__(self, backbone, ent_emb=None, **head_kw):
        super().__init__()
        self.q_enc = SparseEncoder(backbone, "mlp", ent_emb, **head_kw)
        self.d_enc = SparseEncoder(backbone, "mlm", ent_emb, **head_kw)

    @property
    def uses_entities(self):
        return self.q_enc.head is not None

    def encode_q(self, batch):
        return self.q_enc(**batch)

    def encode_d(self, batch):
        return self.d_enc(**batch)

    def load_lsr_weights(self, state_dict):
        """initialise from an LSR-w checkpoint (paper: DyVo is initialised from a pretrained LSR)."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        return [m for m in missing if ".head." not in m], unexpected
