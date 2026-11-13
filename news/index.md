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

- [`cheb_conv()`](https://adamrsophie.github.io/graphlearnR/reference/cheb_conv.md)
  (ChebConv),
  [`pna_conv()`](https://adamrsophie.github.io/graphlearnR/reference/pna_conv.md)
  with
  [`pna_degree()`](https://adamrsophie.github.io/graphlearnR/reference/pna_conv.md)
  (PNA),
  [`appnp()`](https://adamrsophie.github.io/graphlearnR/reference/appnp.md)
  and
  [`appnp_model()`](https://adamrsophie.github.io/graphlearnR/reference/appnp.md)
  (APPNP), and
  [`sag_pool()`](https://adamrsophie.github.io/graphlearnR/reference/topk_pool.md)
  (SAGPool).
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
- [`cluster_loader()`](https://adamrsophie.github.io/graphlearnR/reference/cluster_loader.md)
  (Cluster-GCN) and
  [`saint_loader()`](https://adamrsophie.github.io/graphlearnR/reference/cluster_loader.md)
  (GraphSAINT random walks with loss and edge normalization), used
  through `train_graph_model(loader = )`.

### Heterogeneous, temporal, and transformer models

- [`hetero_graph_data()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_graph_data.md),
  reverse relations, typed sampling, typed link splits,
  [`hetero_conv()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_conv.md),
  and
  [`hetero_gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_gnn_model.md).
- Event based temporal graphs, snapshots, strictly causal temporal
  neighbors, a graph GRU, spatiotemporal forecasting, and dynamic link
  prediction with baselines.
- [`tgn_model()`](https://adamrsophie.github.io/graphlearnR/reference/tgn_model.md)
  and
  [`train_tgn()`](https://adamrsophie.github.io/graphlearnR/reference/tgn_model.md):
  temporal graph networks with node memory and time, graph sum, or graph
  attention embeddings.
- `hgt_conv(temporal_encoding = TRUE)` and the `edge_time` field of
  heterogeneous relations add HGT’s relative temporal encoding.
- [`graph_ode_block()`](https://adamrsophie.github.io/graphlearnR/reference/graph_ode_block.md)
  and
  [`graph_ode_model()`](https://adamrsophie.github.io/graphlearnR/reference/graph_ode_block.md):
  GRAND style continuous depth diffusion with fixed step Runge-Kutta or
  Euler integration.
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
- [`download_ogb()`](https://adamrsophie.github.io/graphlearnR/reference/read_ogb_node.md),
  [`read_ogb_node()`](https://adamrsophie.github.io/graphlearnR/reference/read_ogb_node.md),
  and
  [`read_ogb_graph()`](https://adamrsophie.github.io/graphlearnR/reference/read_ogb_node.md)
  for OGB node and graph property datasets in their CSV layout, with
  official splits.

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
- [`consistent_labels()`](https://adamrsophie.github.io/graphlearnR/reference/consistent_labels.md)
  finds the most probable labeling that obeys edge rules with the z3r
  optimizer.
- [`select_links()`](https://adamrsophie.github.io/graphlearnR/reference/select_links.md)
  chooses links under a budget as a 0-1 program in optcompR.
- [`aggregate_jax()`](https://adamrsophie.github.io/graphlearnR/reference/aggregate_jax.md)
  performs neighborhood aggregation with jaxR arrays.
- [`luz_node_dataloader()`](https://adamrsophie.github.io/graphlearnR/reference/luz_node_dataloader.md)
  and `luz_node_module` train node models with luz; `augment()` and
  `glance()` methods summarize fits for tidymodels.
