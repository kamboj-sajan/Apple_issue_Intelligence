"""GraphSAGE link prediction between questions (PyTorch Geometric ``SAGEConv``).

Encoder: ``h0 = W x + type_embedding``; K mean-aggregation ``SAGEConv`` layers with ReLU and dropout;
the node representation is ``[h0 || h_K]`` (a skip connection, so the model can fall back on text
similarity when the neighbourhood is uninformative).

Decoders (chosen on validation):
* ``mlp``: ``MLP([z_q * z_c, |z_q - z_c|])``
* ``dot``: ``a * cos(z_q, z_c) + b`` with learned ``a``, ``b``

Training: full-batch message passing on the graph as of the validation boundary; binary
cross-entropy on positive pairs (duplicate / body-link labels whose query precedes the boundary)
against random and hard negatives; Adam; early stopping on validation MRR.

New questions at inference: a question posted at time ``t`` is added with its text embedding and
the edges that exist at ``t`` (its tags, entities, and similarity/temporal edges to *earlier*
questions), and is embedded with the trained weights. No retraining is needed (inductive).
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from ml.evaluation.ranking import ranks_of_relevant
from ml.models.link_data import NODE_TYPES, LinkGraph, to_ns
from ml.retrieval.related import EvalSet, RetrievalIndex

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.2
    decoder: str = "mlp"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 40
    batch_pos: int = 256
    n_random: int = 4
    n_hard: int = 4
    eval_every: int = 2
    patience: int = 3
    seed: int = 0
    loss: str = "bce"  # bce | listwise (softmax over each positive and its negatives)
    text_residual: bool = False  # add a learned multiple of the raw text cosine to the score
    extra: dict = field(default_factory=dict)


class GraphSAGELink(nn.Module):
    def __init__(self, in_dim: int, cfg: TrainConfig):
        super().__init__()
        h = cfg.hidden
        self.proj = nn.Linear(in_dim, h)
        self.type_emb = nn.Embedding(len(NODE_TYPES), h)
        self.convs = nn.ModuleList(SAGEConv(h, h, aggr="mean") for _ in range(cfg.layers))
        self.dropout = cfg.dropout
        self.decoder = cfg.decoder
        self.text_residual = cfg.text_residual
        if cfg.text_residual:
            self.text_scale = nn.Parameter(torch.tensor(10.0))
        if cfg.decoder == "mlp":
            self.dec = nn.Sequential(nn.Linear(4 * h, h), nn.ReLU(), nn.Linear(h, 1))
        else:
            self.scale = nn.Parameter(torch.tensor(10.0))
            self.bias = nn.Parameter(torch.tensor(0.0))

    def encode(self, x, node_type, edge_index):
        h0 = self.proj(x) + self.type_emb(node_type)
        h = h0
        for conv in self.convs:
            h = F.dropout(F.relu(conv(h, edge_index)), self.dropout, self.training)
        return torch.cat([h0, h], dim=-1)

    def score(self, zq: torch.Tensor, zc: torch.Tensor, xq=None, xc=None) -> torch.Tensor:
        if zq.dim() == 1:
            zq = zq.expand_as(zc)
        if self.decoder == "mlp":
            s = self.dec(torch.cat([zq * zc, (zq - zc).abs()], dim=-1)).squeeze(-1)
        else:
            s = self.scale * F.cosine_similarity(zq, zc, dim=-1) + self.bias
        if self.text_residual:
            if xq.dim() == 1:
                xq = xq.expand_as(xc)
            s = s + self.text_scale * F.cosine_similarity(xq, xc, dim=-1)
        return s


# --------------------------------------------------------------------------- negatives


def sample_negatives(
    index: RetrievalIndex,
    query_pos: int,
    exclude: set[int],
    primary_category: np.ndarray,
    rng: np.random.Generator,
    n_random: int,
    n_hard: int,
    bm25_top: int = 50,
    max_cos: float = 0.8,
) -> tuple[list[int], list[int]]:
    """Random earlier questions plus *hard* negatives: high BM25 overlap with the query but a different
    primary category and embedding cosine < ``max_cos``. The cosine cap and the exclusion of every
    labelled pair reduce the chance of sampling an unlabelled true duplicate."""
    rand: list[int] = []
    tries = 0
    while len(rand) < n_random and query_pos > len(exclude) and tries < 50 * n_random:
        c = int(rng.integers(0, query_pos))
        tries += 1
        if c not in exclude and c not in rand:
            rand.append(c)
    s = index.bm25.scores(index.tokens[query_pos])[:query_pos]
    top = np.argsort(-s)[:bm25_top]
    cos = index.emb[top] @ index.emb[query_pos] if index.emb is not None else np.zeros(len(top))
    ok = [int(c) for c, cs in zip(top, cos, strict=True) if c not in exclude and cs < max_cos]
    hard = [c for c in ok if primary_category[c] != primary_category[query_pos]][:n_hard]
    if len(hard) < n_hard:  # relax the category condition rather than return too few
        hard += [c for c in ok if c not in hard][: n_hard - len(hard)]
    return rand, hard


def build_training_pairs(
    index: RetrievalIndex,
    pairs: pd.DataFrame,
    all_pairs: pd.DataFrame,
    cfg: TrainConfig,
    seed: int,
) -> pd.DataFrame:
    """One row per (query, candidate, label) with negatives sampled for every positive."""
    rng = np.random.default_rng(seed)
    pos = index.pos
    related: dict[int, set[int]] = {}
    for q, t in zip(all_pairs["query_id"], all_pairs["target_id"], strict=True):
        if q in pos.index and t in pos.index:
            related.setdefault(int(pos[q]), set()).add(int(pos[t]))
            related.setdefault(int(pos[t]), set()).add(int(pos[q]))
    cat = index.q["primary_category"].fillna("").to_numpy()
    rows = []
    for q, t in zip(pairs["query_id"], pairs["target_id"], strict=True):
        qp, tp = int(pos[q]), int(pos[t])
        rows.append((qp, tp, 1.0, "positive"))
        rand, hard = sample_negatives(index, qp, related.get(qp, set()) | {qp}, cat, rng, cfg.n_random, cfg.n_hard)
        rows += [(qp, c, 0.0, "random") for c in rand] + [(qp, c, 0.0, "hard") for c in hard]
    return pd.DataFrame(rows, columns=["query_pos", "cand_pos", "label", "kind"])


# --------------------------------------------------------------------------- training / inference


class Scorer:
    """Scores all earlier candidates for a query using embeddings computed from the graph as of a time."""

    def __init__(self, model: GraphSAGELink, graph: LinkGraph, index: RetrievalIndex):
        self.model, self.graph, self.index = model, graph, index
        self.pos2node = torch.from_numpy(graph.question_node.loc[index.q["question_id"]].to_numpy())
        self.times = to_ns(index.q["created_at"])
        self._cache: tuple[int, torch.Tensor] | None = None

    @torch.no_grad()
    def embeddings(self, t_ns: int) -> torch.Tensor:
        if self._cache is None or self._cache[0] != t_ns:
            self.model.eval()
            z = self.model.encode(self.graph.x, self.graph.node_type, self.graph.edge_index_as_of(t_ns))
            self._cache = (t_ns, z)
        return self._cache[1]

    @torch.no_grad()
    def scores(self, query_pos: int, t_ns: int | None = None) -> np.ndarray:
        """Scores of candidates [0, query_pos). Default time: the query's own creation time (strict)."""
        z = self.embeddings(int(self.times[query_pos]) if t_ns is None else t_ns)
        qn, cn = self.pos2node[query_pos], self.pos2node[:query_pos]
        x = self.graph.x
        return self.model.score(z[qn], z[cn], x[qn], x[cn]).numpy()


def evaluate_mrr(scorer: Scorer, es: EvalSet, weekly: bool = True) -> float:
    """Mean reciprocal rank. ``weekly=True`` embeds all queries of a week with the graph as of the end
    of that week (fast; used for validation). Test evaluation uses the strict per-query time."""
    q = es.queries
    times = scorer.times[q["query_pos"].to_numpy()]
    if weekly:
        # integer ns at the end of each query's week (UTC)
        key = to_ns(pd.DatetimeIndex(pd.to_datetime(times, utc=True)).tz_localize(None).to_period("W-SUN").end_time)
    else:
        key = times
    rr = []
    for k in np.unique(key):
        for r in q[key == k].itertuples(index=False):
            s = scorer.scores(r.query_pos, int(k))
            rr.append(1.0 / ranks_of_relevant(s, r.relevant_pos).min())
    return float(np.mean(rr))


def _listwise_loss(logits: torch.Tensor, y: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
    """Softmax cross-entropy per (positive + its negatives) block: -log p(positive | block)."""
    _, inv = torch.unique(block, return_inverse=True)
    n = int(inv.max()) + 1
    mx = torch.full((n,), -torch.inf).scatter_reduce(0, inv, logits, "amax").detach()
    lse = torch.zeros(n).scatter_add(0, inv, torch.exp(logits - mx[inv])).log() + mx
    pos = torch.zeros(n).scatter_add(0, inv, logits * y)
    return (lse - pos).mean()


def train_model(
    graph: LinkGraph,
    index: RetrievalIndex,
    train_rows: pd.DataFrame,
    train_cutoff_ns: int,
    val_set: EvalSet,
    cfg: TrainConfig,
    on_eval: Callable[[int, float, float], None] | None = None,
) -> tuple[GraphSAGELink, list[dict]]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = GraphSAGELink(graph.x.shape[1], cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ei = graph.edge_index_as_of(train_cutoff_ns)
    pos2node = torch.from_numpy(graph.question_node.loc[index.q["question_id"]].to_numpy())
    qn = pos2node[torch.from_numpy(train_rows["query_pos"].to_numpy())]
    cn = pos2node[torch.from_numpy(train_rows["cand_pos"].to_numpy())]
    y = torch.from_numpy(train_rows["label"].to_numpy()).float()
    # build_training_pairs emits each positive followed by its own negatives; keep them together.
    pos_rows = np.nonzero(train_rows["label"].to_numpy() == 1)[0]
    blocks = np.split(np.arange(len(train_rows)), pos_rows[1:])
    block_id = torch.from_numpy(np.repeat(np.arange(len(blocks)), [len(b) for b in blocks]))

    history, best, best_state, bad = [], -1.0, None, 0
    scorer = Scorer(model, graph, index)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = rng.permutation(len(blocks))
        losses = []
        for i in range(0, len(order), cfg.batch_pos):
            idx = torch.from_numpy(np.concatenate([blocks[j] for j in order[i : i + cfg.batch_pos]]))
            z = model.encode(graph.x, graph.node_type, ei)
            logits = model.score(z[qn[idx]], z[cn[idx]], graph.x[qn[idx]], graph.x[cn[idx]])
            if cfg.loss == "listwise":
                loss = _listwise_loss(logits, y[idx], block_id[idx])
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        rec = {"epoch": epoch, "loss": float(np.mean(losses))}
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            scorer._cache = None
            rec["val_mrr"] = evaluate_mrr(scorer, val_set, weekly=True)
            if rec["val_mrr"] > best + 1e-4:
                best, best_state, bad = rec["val_mrr"], copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            if on_eval:
                on_eval(epoch, rec["loss"], rec["val_mrr"])
            log.info("epoch %d loss %.4f val_mrr %.4f", epoch, rec["loss"], rec["val_mrr"])
        history.append(rec)
        if bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
