# graphlearnR

graphlearnR is a framework for machine learning on graphs and graph
neural networks in R. It is built on the
[torch](https://torch.mlverse.org) package, so models run on the CPU or
a GPU, and nothing calls Python.

It covers the whole workflow: validated graph data with explicit 1-based
indexing, conversion from igraph, edge lists and sparse matrices,
message passing layers, neighbor sampling, leakage safe splits, training
for node, edge, link and graph level tasks, evaluation, heterogeneous
and temporal graphs, graph transformers, molecular graphs, and
explanations.

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
| Layers | GraphSAGE, GCN, GraphConv, GIN and GINE, GAT, GATv2, transformer convolution, NNConv, RGCN, ChebConv, PNA; APPNP propagation; GRAND style graph neural ODE blocks; global, top-k, SAGPool and DiffPool pooling |
| Training | node, edge, link and graph classification and regression; full batch, neighbor sampled, Cluster-GCN or GraphSAINT batches; early stopping, schedulers, checkpoints, reproducibility record |
| Splits | stratified node splits, link splits without leakage, edge label, group, scaffold and temporal splits; negative sampling |
| Heterogeneous graphs | typed nodes and relations, reverse relations, typed sampling, relation specific link splits, heterogeneous GNNs including RGCN and the heterogeneous graph transformer (HGT) with relative temporal encoding |
| Temporal graphs | event streams, causal neighborhoods, snapshots, graph GRU, spatiotemporal forecasting, dynamic link prediction with baselines, temporal graph networks (TGN) with node memory and graph attention |
| Graph transformers | Laplacian and random walk encodings, shortest path distances, Graphormer style model |
| Molecules | graphs from atoms and bonds with element and bond order features, scaffold splits, synthetic molecules; TU loaders for MUTAG and similar sets |
| Explainability | saliency, integrated gradients, edge gradients, occlusion, GNNExplainer, attention weights, explanation subgraphs, fidelity and stability |
| Sibling packages and frameworks | optional: graph diffusion solved by modelcompR, exact graph operators from symcompR, rule constrained labels with z3r, budgeted link selection with optcompR, jaxR aggregation, luz training, `augment()` and `glance()` |
| Datasets | karate club, stochastic block models, synthetic graph, heterogeneous and temporal data; checksum verified downloaders for TU and LINQS datasets; readers for OGB node and graph property datasets |

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
transformers, molecular graphs, explainability, evaluation, and the
sibling packages.

## How it is tested

Every layer is checked for permutation equivariance, against a dense or
reference formula, and with finite difference gradients in double
precision. Pooled models are checked for permutation invariance and
batched forward passes against separate ones. Splits, samplers and
temporal code are checked for leakage. Continuous integration runs
`R CMD check` on Linux, macOS and Windows. A benchmark workflow copies
weights into PyTorch Geometric and checks that SAGEConv and GCNConv give
the same outputs (largest difference 6e-8 in the last run); on one 2
thread CPU runner, single layer forward times were of the same order as
PyG’s. Scripts are in `benchmarks/pyg`.

## Known limitations

- Neighbor and negative sampling run in R on the CPU. On graphs that fit
  in memory, full batch training is faster than sampled training.
- The graph transformer uses dense attention within each graph, so
  memory grows with the square of the largest graph.
- There is no compiled code and no sparse matrix multiplication fast
  path; aggregation uses torch scatter operations.
- CPU and Apple MPS results agree in tests on an Apple M2. CUDA has not
  been tested: no CUDA hardware was available, and GitHub hosted runners
  have no GPU. The device tests skip without an accelerator.
- DiffPool and the graph transformer build dense per graph matrices, so
  their memory grows with the square of the largest graph in a batch.
- graphlearnR does not parse SMILES or compute chemical scaffolds; use a
  chemistry toolkit and pass atoms, bonds and scaffold identifiers in.
- [`graph_diffusion()`](https://adamrsophie.github.io/graphlearnR/reference/graph_diffusion.md)
  solves on R numerics through modelcompR, so it is a simulation tool,
  not a trainable neural ODE layer.
- [`graph_ode_block()`](https://adamrsophie.github.io/graphlearnR/reference/graph_ode_block.md)
  integrates with a fixed number of steps; it is not an adaptive solver.
- TGN has one embedding layer (one hop) and last message aggregation.
- OGB link property datasets (ogbl) ship splits as PyTorch files and are
  not readable; node (ogbn) and graph (ogbg) datasets are.
- The sibling bridges are tested in continuous integration on Linux
  only; the z3r bridge is tested locally, because the Z3 on the runners
  is older than z3r needs.
- There is no parsnip engine, because graph models need the whole graph
  at prediction time.

## Related work

graphlearnR follows the layer designs of [PyTorch
Geometric](https://pyg.org) (Fey and Lenssen 2019). In R,
[igraph](https://r.igraph.org) and
[tidygraph](https://tidygraph.data-imaginist.com) cover graph analysis
and manipulation, and graphlearnR converts from both. It is part of a
series of R packages by the same author with
[modelcompR](https://github.com/adamrsophie/modelcompR),
[symcompR](https://github.com/adamrsophie/symcompR),
[optcompR](https://github.com/adamrsophie/optcompR),
[jaxR](https://github.com/adamrsophie/jaxR) and
[z3r](https://github.com/adamrsophie/z3r); the optional bridges to
modelcompR and symcompR are described in the ecosystem article.

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
