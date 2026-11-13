# Changelog

## graphlearnR 0.1.0

First release.

### Graph data

- [`graph_data()`](https://adamrsophie.github.io/graphlearnR/reference/graph_data.md)
  stores homogeneous graphs with 1-based `edge_index`, node and edge
  features, labels, weights, masks, and identifiers. Undirected input is
  given once per pair and stored in both directions with `edge_pair`.
- Conversions with igraph, edge lists, and dense or sparse adjacency
  matrices;
  [`graph_tensors()`](https://adamrsophie.github.io/graphlearnR/reference/graph_tensors.md)
  moves a graph to torch tensors on any device.
- [`graph_batch()`](https://adamrsophie.github.io/graphlearnR/reference/graph_batch.md)
  and
  [`unbatch()`](https://adamrsophie.github.io/graphlearnR/reference/graph_batch.md)
  combine graphs into one disconnected graph and back.
- Transforms: self loops, coalescing, induced subgraphs, symmetrization,
  feature normalization, degree features.

### Message passing and layers

- Scatter aggregations (sum, mean, max, min, std, softmax) and
  [`message_passing()`](https://adamrsophie.github.io/graphlearnR/reference/message_passing.md)
  for custom layers.
- Layers: GraphSAGE, GCN, GraphConv, GIN and GINE, GAT, GATv2, graph
  transformer convolution, edge conditioned convolution (NNConv), and
  RGCN with basis decomposition.
- Global add, mean, max, and attention pooling, top-k pooling, and
  DiffPool
  ([`dense_diff_pool()`](https://adamrsophie.github.io/graphlearnR/reference/to_dense_adj.md),
  [`to_dense_adj()`](https://adamrsophie.github.io/graphlearnR/reference/to_dense_adj.md),
  [`dense_graph_conv()`](https://adamrsophie.github.io/graphlearnR/reference/to_dense_adj.md),
  [`diffpool_model()`](https://adamrsophie.github.io/graphlearnR/reference/diffpool_model.md)),
  whose auxiliary losses the trainer adds.
- [`hgt_conv()`](https://adamrsophie.github.io/graphlearnR/reference/hgt_conv.md):
  heterogeneous graph transformer layer, also available as
  `hetero_gnn_model(conv = "hgt")`.
- [`gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/gnn_model.md)
  builds node, edge, link, and graph level models with normalization,
  residual connections, and dropout.

### Training and evaluation

- [`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
  trains node, edge, link, and graph classification and regression
  models, full batch or with neighbor sampling, with early stopping,
  schedulers, checkpoints, and a reproducibility record.
- [`predict()`](https://rdrr.io/r/stats/predict.html),
  [`evaluate_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/evaluate_graph_model.md),
  [`save_graph_fit()`](https://adamrsophie.github.io/graphlearnR/reference/save_graph_fit.md),
  and
  [`load_graph_fit()`](https://adamrsophie.github.io/graphlearnR/reference/save_graph_fit.md).
- Leakage-safe splits for nodes, edges, edge labels, and graphs (random,
  stratified, group, scaffold).
- Metrics for classification, calibration, regression, ranking, and
  clustering.

### Sampling and scale

- [`sample_neighbors()`](https://adamrsophie.github.io/graphlearnR/reference/sample_neighbors.md),
  [`k_hop_subgraph()`](https://adamrsophie.github.io/graphlearnR/reference/k_hop_subgraph.md),
  node and graph loaders, negative sampling, and corrupted tails for
  ranking evaluation.

### Heterogeneous, temporal, and transformer models

- [`hetero_graph_data()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_graph_data.md),
  reverse relations, typed sampling, typed link splits,
  [`hetero_conv()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_conv.md),
  and
  [`hetero_gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_gnn_model.md).
- Event based temporal graphs, snapshots, strictly causal temporal
  neighbors, a graph GRU, spatiotemporal forecasting, and dynamic link
  prediction with baselines.
- Laplacian and random walk positional encodings, shortest path
  distances, and a Graphormer style graph transformer.

### Explainability

- Saliency, integrated gradients, edge gradients, occlusion,
  GNNExplainer, attention weights, explanation subgraphs, fidelity, and
  stability.

### Datasets

- Built in: Zachary’s karate club, stochastic block models, and
  synthetic graph, heterogeneous, and temporal datasets.
- Downloaders with checksum verification for TU datasets and the LINQS
  citation graphs (Cora, CiteSeer); nothing third party is bundled.

### Molecules

- [`molecule_graph()`](https://adamrsophie.github.io/graphlearnR/reference/molecule_graph.md)
  builds graphs from atoms and bonds with element and bond order
  features;
  [`dataset_synthetic_molecules()`](https://adamrsophie.github.io/graphlearnR/reference/molecule_graph.md)
  generates small synthetic molecules with scaffold identifiers for
  scaffold splits.

### Sibling packages (optional)

- [`graph_diffusion()`](https://adamrsophie.github.io/graphlearnR/reference/graph_diffusion.md)
  simulates diffusion on a graph with the modelcompR ODE solvers.
- [`graph_operator_symbolic()`](https://adamrsophie.github.io/graphlearnR/reference/graph_operator_symbolic.md)
  returns exact adjacency, Laplacian, and GCN operators as symcompR
  matrices.
