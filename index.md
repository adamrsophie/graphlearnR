# graphlearnR

graphlearnR is a framework for machine learning on graphs and graph
neural networks in R. It is built on the
[torch](https://torch.mlverse.org) package, so models run on the CPU or
a GPU, and nothing calls Python.

It covers the whole workflow: validated graph data with explicit 1-based
indexing, conversion from igraph, edge lists and sparse matrices,
message passing layers, neighbor sampling, leakage safe splits, training
for node, edge, link and graph level tasks, evaluation, heterogeneous
and temporal graphs, graph transformers, and explanations.

## Installation

``` r

# install.packages("remotes")
remotes::install_github("adamrsophie/graphlearnR")

# LibTorch, once per machine
torch::install_torch()
```

## Example

``` r

library(graphlearnR)

g <- dataset_karate()                       # Zachary's karate club
g <- split_nodes(g, seed = 42)              # stratified train, val, test masks

model <- gnn_model("sage", input_dim = 34, hidden_dim = 16, output_dim = 2)
fit <- train_graph_model(model, g, task = "node_classification",
                         epochs = 150, early_stopping = 20, seed = 1)

evaluate_graph_model(fit, g)                # accuracy, F1, AUC, calibration
head(predict(fit, g))                       # one row per node
```

Graph classification and link prediction use the same interface:

``` r

graphs <- dataset_synthetic_graphs(120, seed = 1)
gin <- gnn_model("gin", 1, 32, 2, pool = "sum")
fit_g <- train_graph_model(gin, graphs, "graph_classification", batch_size = 16)

split <- split_edges(dataset_sbm(c(50, 50), seed = 1), seed = 1)
fit_l <- train_graph_model(gnn_model("gcn", 8, 32), split, "link_prediction")
evaluate_graph_model(fit_l)                 # AUC, AP, MRR, Hits@k
```

## Features

| Area | What is included |
|----|----|
| Graph data | [`graph_data()`](https://adamrsophie.github.io/graphlearnR/reference/graph_data.md) with node, edge and graph features and labels, weights, masks, identifiers; undirected pairs stored in both directions; validation with clear errors |
| Conversion | igraph, edge lists with arbitrary identifiers, dense and sparse adjacency matrices, torch tensors on any device |
| Batching | [`graph_batch()`](https://adamrsophie.github.io/graphlearnR/reference/graph_batch.md) and [`unbatch()`](https://adamrsophie.github.io/graphlearnR/reference/graph_batch.md), lazy and on disk datasets, transforms |
| Message passing | vectorized scatter sum, mean, max, min, std, softmax; a general [`message_passing()`](https://adamrsophie.github.io/graphlearnR/reference/message_passing.md) for custom layers |
| Layers | GraphSAGE, GCN, GraphConv, GIN and GINE, GAT, GATv2, transformer convolution, NNConv, RGCN; global and top-k pooling |
| Training | node, edge, link and graph classification and regression; full batch or neighbor sampled; early stopping, schedulers, checkpoints, reproducibility record |
| Splits | stratified node splits, link splits without leakage, edge label, group, scaffold and temporal splits; negative sampling |
| Heterogeneous graphs | typed nodes and relations, reverse relations, typed sampling, relation specific link splits, heterogeneous GNNs |
| Temporal graphs | event streams, causal neighborhoods, snapshots, graph GRU, spatiotemporal forecasting, dynamic link prediction with baselines |
| Graph transformers | Laplacian and random walk encodings, shortest path distances, Graphormer style model |
| Explainability | saliency, integrated gradients, edge gradients, occlusion, GNNExplainer, attention weights, fidelity and stability |
| Datasets | karate club, stochastic block models, synthetic graph, heterogeneous and temporal data; checksum verified downloaders for TU and LINQS datasets |

## Conventions

- **Indices are 1-based** in every interface: `edge_index`, seeds,
  pairs, returned node ids.
  [`to_zero_based()`](https://adamrsophie.github.io/graphlearnR/reference/to_zero_based.md)
  converts at an explicit boundary.
- **Edges point from source to target**; layers aggregate incoming
  edges.
- **Undirected graphs** are given once per pair and stored in both
  directions; `edge_pair` links the two directions, and splits and
  samplers respect it.
- **Functions that draw random numbers take a `seed`** and leave the
  caller’s random number stream untouched.

## Documentation

The [package website](https://adamrsophie.github.io/graphlearnR/) has a
reference for every function and articles on graph data, message
passing, each task, sampling, heterogeneous and temporal graphs, graph
transformers, explainability, and evaluation.

## How it is tested

Every layer is checked for permutation equivariance, against a dense or
reference formula, and with finite difference gradients in double
precision. Pooled models are checked for permutation invariance and
batched forward passes against separate ones. Splits, samplers and
temporal code are checked for leakage. Continuous integration runs
`R CMD check` on Linux, macOS and Windows.

## Known limitations

- Neighbor and negative sampling run in R on the CPU. On graphs that fit
  in memory, full batch training is faster than sampled training.
- The graph transformer uses dense attention within each graph, so
  memory grows with the square of the largest graph.
- There is no compiled code and no sparse matrix multiplication fast
  path; aggregation uses torch scatter operations.
- CPU and Apple MPS results agree in tests on an Apple M2; CUDA has not
  been tested.
- Not yet included: luz and tidymodels integration, Chebyshev, APPNP and
  PNA layers, cluster based samplers, memory based temporal models, and
  OGB loaders.

## Related work

graphlearnR follows the layer designs of [PyTorch
Geometric](https://pyg.org) (Fey and Lenssen 2019). In R,
[igraph](https://r.igraph.org) and
[tidygraph](https://tidygraph.data-imaginist.com) cover graph analysis
and manipulation, and graphlearnR converts from both. It is part of a
series of R packages by the same author with
[symcompR](https://github.com/adamrsophie/symcompR) and
[z3r](https://github.com/adamrsophie/z3r).

## Citation

``` r

citation("graphlearnR")
```

## Contributing

See
[CONTRIBUTING.md](https://adamrsophie.github.io/graphlearnR/CONTRIBUTING.md).
Please note that this project is released with a [Contributor Code of
Conduct](https://adamrsophie.github.io/graphlearnR/CODE_OF_CONDUCT.md).

## License

GPL-3
