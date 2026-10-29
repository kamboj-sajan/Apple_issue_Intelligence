"""Reproduction of supervised GraphSAGE-mean on PPI (Hamilton, Ying & Leskovec, NeurIPS 2017).

Target: Table 1, PPI, supervised micro-F1 = 0.598 for GraphSAGE-mean.

The ``faithful`` model follows the official TensorFlow implementation
(github.com/williamleif/GraphSAGE, ``supervised_train.py`` / ``supervised_models.py`` /
``aggregators.py`` / ``minibatch.py``):

* K = 2; each layer computes ``[W_self h_v || W_neigh mean(h_N(v))]`` (concat), no bias,
  ReLU on the hidden layer and identity on the last; output L2-normalised; linear classifier;
  sigmoid cross-entropy (``--sigmoid``); Adam, lr 0.01, batch 512, 10 epochs, no dropout or weight
  decay; gradients clipped to [-5, 5].
* Fixed-size neighbour tables of ``max_degree`` = 128 per node (sampled without replacement if the
  degree is larger, with replacement if smaller; isolated nodes point to a zero "padding" node).
  Each batch shuffles the table columns and keeps the first S columns.
* Following Algorithm 2 of the paper (and the official ``sample()``), the target's direct
  neighbours are sampled with S_2 = 10 and the second hop with S_1 = 25.
* Features are standardised with statistics from training nodes; test evaluation also samples.
* Micro-F1 with threshold 0.5 on sigmoid outputs, over the 2 test graphs.

The ``pyg_fullbatch`` variant uses PyTorch Geometric's ``SAGEConv`` (mean aggregation, which *sums*
the self and neighbour transforms) on full neighbourhoods, for comparison with the framework
implementation. PyG's ``NeighborLoader`` needs ``pyg-lib``/``torch-sparse``, which have no wheels for
this torch build, which is why the faithful sampler is implemented here directly.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

PAPER_MICRO_F1 = {"GraphSAGE-GCN": 0.500, "GraphSAGE-mean": 0.598, "GraphSAGE-LSTM": 0.612, "GraphSAGE-pool": 0.600}


@dataclass
class PPIConfig:
    variant: str = "faithful"  # faithful | pyg_fullbatch
    epochs: int = 10
    lr: float = 0.01
    batch_size: int = 512
    dim: int = 128
    samples_hop1: int = 10  # S_2 in the paper: direct neighbours of the target
    samples_hop2: int = 25  # S_1 in the paper: neighbours of those neighbours
    max_degree: int = 128
    grad_clip: float = 5.0
    seed: int = 0


# ----------------------------------------------------------------------------- data


def load_ppi(root: Path):
    from torch_geometric.datasets import PPI

    return [PPI(str(root), split=s) for s in ("train", "val", "test")]


def merge_graphs(dataset) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Concatenate graphs of a split into one disjoint union: features, labels, adjacency lists."""
    xs, ys, adj, offset = [], [], [], 0
    for g in dataset:
        n = g.num_nodes
        xs.append(g.x.numpy())
        ys.append(g.y.numpy())
        nbrs = [[] for _ in range(n)]
        for s, d in g.edge_index.t().numpy():
            if s != d:
                nbrs[s].append(d + offset)
        adj.extend(np.unique(np.asarray(v, dtype=np.int64)) for v in nbrs)
        offset += n
    return np.vstack(xs), np.vstack(ys), adj


def neighbour_table(adj: list[np.ndarray], max_degree: int, pad: int, rng: np.random.Generator) -> np.ndarray:
    table = np.full((len(adj) + 1, max_degree), pad, dtype=np.int64)  # last row: padding node -> itself
    for i, nb in enumerate(adj):
        if len(nb) == 0:
            continue
        table[i] = rng.choice(nb, max_degree, replace=len(nb) < max_degree)
    return table


# ----------------------------------------------------------------------------- faithful model


class MeanAggregator(nn.Module):
    def __init__(self, d_in: int, d_out: int, act: bool):
        super().__init__()
        self.self_w = nn.Linear(d_in, d_out, bias=False)
        self.neigh_w = nn.Linear(d_in, d_out, bias=False)
        nn.init.xavier_uniform_(self.self_w.weight)
        nn.init.xavier_uniform_(self.neigh_w.weight)
        self.act = act

    def forward(self, self_vecs: torch.Tensor, neigh_vecs: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.self_w(self_vecs), self.neigh_w(neigh_vecs.mean(dim=-2))], dim=-1)
        return torch.relu(out) if self.act else out


class SampledGraphSAGE(nn.Module):
    def __init__(self, d_in: int, dim: int, n_classes: int):
        super().__init__()
        self.agg1 = MeanAggregator(d_in, dim, act=True)
        self.agg2 = MeanAggregator(2 * dim, dim, act=False)
        self.pred = nn.Linear(2 * dim, n_classes)
        nn.init.xavier_uniform_(self.pred.weight)
        nn.init.zeros_(self.pred.bias)

    def forward(self, x_t, x_1, x_2):
        # x_t [B,d], x_1 [B,S1,d] direct neighbours, x_2 [B,S1,S2,d] their neighbours
        h_t = self.agg1(x_t, x_1)
        h_1 = self.agg1(x_1, x_2)
        h = self.agg2(h_t, h_1)
        return self.pred(nn.functional.normalize(h, dim=-1))


def sample_batch(nodes: np.ndarray, table: np.ndarray, s1: int, s2: int, rng: np.random.Generator):
    cols1 = rng.permutation(table.shape[1])[:s1]
    hop1 = table[nodes][:, cols1]  # [B, s1]
    cols2 = rng.permutation(table.shape[1])[:s2]
    hop2 = table[hop1][:, :, cols2]  # [B, s1, s2]
    return hop1, hop2


def _forward(model, feats: torch.Tensor, nodes, table, cfg, rng):
    hop1, hop2 = sample_batch(nodes, table, cfg.samples_hop1, cfg.samples_hop2, rng)
    return model(feats[nodes], feats[hop1], feats[hop2])


def run_faithful(data, cfg: PPIConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    (xtr, ytr, atr), (xva, yva, ava), (xte, yte, ate) = data
    mu, sd = xtr.mean(0), xtr.std(0)
    sd[sd == 0] = 1.0

    def prep(x, adj):
        f = np.vstack([(x - mu) / sd, np.zeros((1, x.shape[1]))])  # last row = padding node
        return torch.tensor(f, dtype=torch.float32), neighbour_table(adj, cfg.max_degree, len(x), rng)

    ftr, ttr = prep(xtr, atr)
    fva, tva = prep(xva, ava)
    fte, tte = prep(xte, ate)
    model = SampledGraphSAGE(xtr.shape[1], cfg.dim, ytr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(xtr))
        losses = []
        for i in range(0, len(order), cfg.batch_size):
            b = order[i : i + cfg.batch_size]
            logits = _forward(model, ftr, b, ttr, cfg, rng)
            loss = loss_fn(logits, ytr_t[b])
            opt.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.grad.clamp_(-cfg.grad_clip, cfg.grad_clip)
            opt.step()
            losses.append(loss.item())
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "val_micro_f1": _eval_sampled(model, fva, tva, yva, cfg, rng),
            }
        )
    return {"test_micro_f1": _eval_sampled(model, fte, tte, yte, cfg, rng), "history": history}


@torch.no_grad()
def _eval_sampled(model, feats, table, y, cfg, rng) -> float:
    model.eval()
    preds = []
    nodes = np.arange(len(y))
    for i in range(0, len(nodes), 256):  # validate_batch_size = 256
        preds.append(torch.sigmoid(_forward(model, feats, nodes[i : i + 256], table, cfg, rng)).numpy())
    return float(f1_score(y, (np.vstack(preds) > 0.5).astype(int), average="micro"))


# ----------------------------------------------------------------------------- PyG full-batch variant


def run_pyg_fullbatch(datasets, cfg: PPIConfig) -> dict:
    from torch_geometric.nn import SAGEConv

    torch.manual_seed(cfg.seed)
    train, val, test = datasets
    x_all = torch.cat([g.x for g in train])
    mu, sd = x_all.mean(0), x_all.std(0).clamp_min(1e-6)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = SAGEConv(train.num_features, 2 * cfg.dim, aggr="mean")
            self.c2 = SAGEConv(2 * cfg.dim, 2 * cfg.dim, aggr="mean")
            self.pred = nn.Linear(2 * cfg.dim, train.num_classes)

        def forward(self, x, ei):
            h = torch.relu(self.c1((x - mu) / sd, ei))
            return self.pred(nn.functional.normalize(self.c2(h, ei), dim=-1))

    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        for g in train:  # one graph per step (no sampling)
            loss = loss_fn(model(g.x, g.edge_index), g.y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg.grad_clip)
            opt.step()
        history.append({"epoch": epoch + 1, "val_micro_f1": _eval_pyg(model, val)})
    return {"test_micro_f1": _eval_pyg(model, test), "history": history}


@torch.no_grad()
def _eval_pyg(model, ds) -> float:
    model.eval()
    ys, ps = [], []
    for g in ds:
        ps.append((torch.sigmoid(model(g.x, g.edge_index)) > 0.5).int().numpy())
        ys.append(g.y.numpy())
    return float(f1_score(np.vstack(ys), np.vstack(ps), average="micro"))


# ----------------------------------------------------------------------------- driver


def environment() -> dict:
    import sklearn
    import torch_geometric

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
    }


def run(root: Path, out_dir: Path, configs: list[PPIConfig]) -> dict:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    datasets = load_ppi(root)
    merged = [merge_graphs(ds) for ds in datasets]
    data_info = {
        s: {"graphs": len(ds), "nodes": int(sum(g.num_nodes for g in ds))}
        for s, ds in zip(("train", "val", "test"), datasets, strict=True)
    }
    data_info["features"], data_info["classes"] = datasets[0].num_features, datasets[0].num_classes
    runs = []
    for cfg in configs:
        t0 = time.time()
        res = run_faithful(merged, cfg) if cfg.variant == "faithful" else run_pyg_fullbatch(datasets, cfg)
        runs.append({"config": asdict(cfg), **res, "seconds": round(time.time() - t0, 1)})
    summary = {}
    for key in sorted({(r["config"]["variant"], r["config"]["epochs"]) for r in runs}):
        scores = [r["test_micro_f1"] for r in runs if (r["config"]["variant"], r["config"]["epochs"]) == key]
        summary[f"{key[0]}_{key[1]}ep"] = {
            "n_seeds": len(scores),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "per_seed": scores,
        }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "paper": "Hamilton, Ying, Leskovec. Inductive Representation "
        "Learning on Large Graphs. NeurIPS 2017 (arXiv:1706.02216), Table 1",
        "paper_micro_f1": PAPER_MICRO_F1,
        "dataset": data_info,
        "environment": environment(),
        "summary": summary,
        "runs": runs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ppi_results.json").write_text(json.dumps(report, indent=2))
    return report
