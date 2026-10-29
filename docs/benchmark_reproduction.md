# GraphSAGE Benchmark Reproduction (PPI)

This reproduction checks that our GraphSAGE implementation and training setup behave as published
**before** we use GraphSAGE on Ask Different. It is not evidence about Ask Different performance.

## Target

* **Paper:** W. L. Hamilton, R. Ying, J. Leskovec. *Inductive Representation Learning on Large Graphs.*
  NeurIPS 2017. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216).
* **Official code:** <https://github.com/williamleif/GraphSAGE> (TensorFlow 1.x).
* **Experiment:** Table 1, PPI, *supervised* micro-F1. Paper values: GraphSAGE-GCN 0.500,
  **GraphSAGE-mean 0.598**, GraphSAGE-LSTM 0.612, GraphSAGE-pool 0.600 (checked against the paper PDF).
* **Task:** inductive multi-label node classification. Train on 20 graphs, validate on 2, test on 2
  unseen graphs. 50 features, 121 labels.

## What we matched

Settings taken from the official `supervised_train.py`, `supervised_models.py`, `aggregators.py`
and `minibatch.py`, not from memory:

| Setting | Official | Ours (`faithful`) |
|---------|----------|-------------------|
| Data | 20/2/2 PPI graphs, 44,906 train nodes | same (PyG `PPI`, identical split and sizes) |
| Layers | K = 2 | 2 |
| Aggregator | mean, `[W_self h_v ‖ W_neigh mean(h_N)]` (concat), no bias | same |
| Hidden dims | 128 + 128 (concat → 256) | same |
| Activations | ReLU hidden, identity last; L2-normalised output | same |
| Classifier / loss | linear layer, sigmoid cross-entropy (`--sigmoid`) | same |
| Sampling | 128-slot neighbour table per node; direct neighbours S₂ = 10, second hop S₁ = 25 (paper Alg. 2) | same; also used at test time |
| Optimiser | Adam, lr 0.01, batch 512, grad clip ±5 | same |
| Epochs | 10 (default) | 10 |
| Dropout / weight decay | 0 / 0 | 0 / 0 |
| Features | standardised with training-node statistics | same |
| Metric | micro-F1, sigmoid > 0.5, on test graphs | same |

## Results (`aii ppi-reproduce`, CPU, 2026-10-29)

| Configuration | Seeds | Test micro-F1 (mean ± sd) | Per seed |
|---------------|-------|---------------------------|----------|
| Paper (GraphSAGE-mean) | – | **0.598** | – |
| Ours, faithful, 10 epochs | 5 | **0.580 ± 0.018** | 0.596, 0.592, 0.589, 0.573, 0.552 |
| Ours, faithful, 50 epochs | 3 | 0.631 ± 0.012 | 0.618, 0.634, 0.641 |
| PyG `SAGEConv`, full neighbourhoods, 10 epochs | 3 | 0.657 ± 0.023 | 0.631, 0.664, 0.676 |

**Reading.** The faithful configuration reproduces the published number within about one standard
deviation: the mean is 0.018 below 0.598, and the paper's value sits just above our best seed.
The variance across seeds (0.552–0.596) is large compared with the gaps between aggregators in
Table 1, so single-run comparisons on PPI are unreliable. Longer training and full neighbourhoods both
improve on the paper. This matches the paper's footnote 6, which notes that later work reached much
higher PPI scores with tuned training.

## Deviations and their likely effect

* **Learning rate:** the paper selected learning rates from {0.01, 0.001, 0.0001} on validation per
  model. We used the official default of 0.01 only. This could explain part of the gap below 0.598.
* **Framework:** PyTorch 2.15 instead of TensorFlow 1.x (different initialisation streams and
  numerics). Xavier-uniform initialisation as in the official layers.
* **Sampling implementation:** PyG's `NeighborLoader` needs `pyg-lib` or `torch-sparse`, which have no
  wheels for this torch build. We implemented the official fixed-size table sampler directly, which
  is closer to the original than PyG's sampler would be.
* **Classifier bias:** the official `layers.Dense` source wasn't reviewed. We use a bias in the final
  linear layer.
* **PyG variant:** `SAGEConv` *sums* self and neighbour transforms rather than concatenating. It trains
  one graph per step (200 steps) instead of 512-node mini-batches (~880 steps).

## Rerun

```bash
pip install -e ".[ml]"
aii ppi-reproduce                                  # 5 + 3 + 3 runs, ~4 minutes on a laptop CPU
aii ppi-reproduce --seeds 1 --long-seeds 0 --pyg-seeds 0   # single faithful run
```

Results, including per-epoch validation F1, seeds and package versions, are written to
`data/artifacts/benchmarks/ppi_results.json`. The dataset downloads to `data/benchmarks/PPI` via PyG.
