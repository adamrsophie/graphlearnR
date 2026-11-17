# Changelog

## graphlearnR 0.1.0

Initial release. graphlearnR provides graph machine learning for R on
top of the `torch` package: graph data structures, message passing
layers, model builders, training and evaluation, sampling for large
graphs, heterogeneous and temporal graphs, graph transformers, and
explainability.

### Graph data

- [`graph_data()`](https://adamrsophie.github.io/graphlearnR/reference/graph_data.md)
  stores homogeneous graphs with 1-based `edge_index`, node and edge
  features, labels, weights, masks, and node identifiers. Undirected
  input is given once per pair and stored in both directions.
- Conversion to and from igraph, edge lists, and dense or sparse
  adjacency matrices;
  [`graph_tensors()`](https://adamrsophie.github.io/graphlearnR/reference/graph_tensors.md)
  moves a graph to torch tensors on any device.
- [`graph_batch()`](https://adamrsophie.github.io/graphlearnR/reference/graph_batch.md)
  and
  [`unbatch()`](https://adamrsophie.github.io/graphlearnR/reference/graph_batch.md)
  combine graphs into one disconnected graph and split them again.
- Transforms for self loops, duplicate edges, induced subgraphs,
  symmetrization, feature normalization, and degree features.
- Validation of graph structure, and a shared feature schema for
  datasets
  ([`graph_schema()`](https://adamrsophie.github.io/graphlearnR/reference/graph_dataset.md)).

### Layers and message passing

- [`message_passing()`](https://adamrsophie.github.io/graphlearnR/reference/message_passing.md)
  and scatter aggregations (sum, mean, max, min, std, softmax) for
  custom layers.
- Convolutions: GraphSAGE, GCN, GraphConv, GIN and GINE, GAT, GATv2,
  graph transformer convolution, edge conditioned convolution (NNConv),
  RGCN with basis decomposition, ChebConv, APPNP, PNA, and EdgeConv.
- Pooling: global add, mean, max, and attention pooling, top-k pooling,
  SAGPool, and DiffPool with its auxiliary losses.
- [`meta_layer()`](https://adamrsophie.github.io/graphlearnR/reference/meta_layer.md)
  composes edge, node, and graph level updates into a graph network
  block.
- [`gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/gnn_model.md)
  builds node, edge, link, and graph level models with normalization,
  residual connections, and dropout.

### Training and evaluation

- [`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
  trains classification and regression models for nodes, edges, links,
  and whole graphs, full batch or on sampled subgraphs, with early
  stopping, learning rate schedulers, checkpoints, gradient clipping,
  class weights, and a reproducibility record of the settings and split
  members.
- [`predict()`](https://rdrr.io/r/stats/predict.html),
  [`evaluate_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/evaluate_graph_model.md),
  [`save_graph_fit()`](https://adamrsophie.github.io/graphlearnR/reference/save_graph_fit.md),
  and
  [`load_graph_fit()`](https://adamrsophie.github.io/graphlearnR/reference/save_graph_fit.md).
- Leakage safe splits for nodes, edges, edge labels, and graphs (random,
  stratified, group, and scaffold). Link splits treat parallel edges and
  both directions of an undirected pair as one unit, and remove held out
  pairs from paired reverse relations.
- Metrics for classification, calibration, regression, link ranking
  (filtered <Hits@K> and MRR), and clustering;
  [`evaluate_embeddings()`](https://adamrsophie.github.io/graphlearnR/reference/evaluate_embeddings.md)
  with a linear probe;
  [`ogb_evaluate()`](https://adamrsophie.github.io/graphlearnR/reference/ogb_evaluate.md)
  with the official OGB metrics.

### Sampling for large graphs

- [`sample_neighbors()`](https://adamrsophie.github.io/graphlearnR/reference/sample_neighbors.md),
  [`k_hop_subgraph()`](https://adamrsophie.github.io/graphlearnR/reference/k_hop_subgraph.md),
  node and graph loaders, negative sampling, and corrupted tails for
  ranking evaluation.
- [`cluster_loader()`](https://adamrsophie.github.io/graphlearnR/reference/cluster_loader.md)
  (Cluster-GCN),
  [`saint_loader()`](https://adamrsophie.github.io/graphlearnR/reference/cluster_loader.md)
  (GraphSAINT random walks with loss and edge normalization), and
  [`ladies_loader()`](https://adamrsophie.github.io/graphlearnR/reference/ladies_loader.md)
  (layer wise importance sampling), used through
  `train_graph_model(loader = )`.

### Heterogeneous graphs

- [`hetero_graph_data()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_graph_data.md)
  with reverse relations, metapaths
  ([`add_metapath()`](https://adamrsophie.github.io/graphlearnR/reference/add_metapath.md)),
  typed neighbor sampling, and typed link splits.
- [`hetero_conv()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_conv.md)
  with sum, mean, or semantic attention across relations, and
  [`hetero_gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_gnn_model.md)
  with per relation SAGE, GraphConv, or GAT layers, RGCN, HGT
  ([`hgt_conv()`](https://adamrsophie.github.io/graphlearnR/reference/hgt_conv.md)),
  and HAN.

### Temporal graphs

- Event based temporal graphs, snapshots, and strictly causal temporal
  neighbors.
- Temporal graph networks
  ([`tgn_model()`](https://adamrsophie.github.io/graphlearnR/reference/tgn_model.md),
  [`train_tgn()`](https://adamrsophie.github.io/graphlearnR/reference/tgn_model.md))
  with node memory and time, graph sum, or multi layer graph attention
  embeddings.
- Dynamic link prediction with a graph GRU or a GRU variant of
  EvolveGCN-O
  ([`evolve_gcn_cell()`](https://adamrsophie.github.io/graphlearnR/reference/evolve_gcn_cell.md)),
  and spatiotemporal forecasting.
- [`predict()`](https://rdrr.io/r/stats/predict.html) for temporal fits
  uses only information available before the prediction time; temporal
  fits can be saved and loaded.
- HGT relative temporal encoding and continuous depth graph neural ODE
  blocks
  ([`graph_ode_block()`](https://adamrsophie.github.io/graphlearnR/reference/graph_ode_block.md),
  [`graph_ode_model()`](https://adamrsophie.github.io/graphlearnR/reference/graph_ode_block.md)).

### Graph transformers

- Laplacian and random walk positional encodings and shortest path
  distances.
- [`graph_transformer_model()`](https://adamrsophie.github.io/graphlearnR/reference/graph_transformer_model.md):
  a Graphormer style transformer with shortest path and edge feature
  attention biases.
- [`gps_layer()`](https://adamrsophie.github.io/graphlearnR/reference/gps_layer.md):
  GPS layers combining local message passing with global attention,
  exact or linear
  ([`linear_attention()`](https://adamrsophie.github.io/graphlearnR/reference/linear_attention.md),
  with positive random feature (Performer style) and elu kernels).

### Explainability

- Saliency, integrated gradients, edge gradients, occlusion,
  GNNExplainer, PGExplainer
  ([`pg_explainer()`](https://adamrsophie.github.io/graphlearnR/reference/pg_explainer.md)),
  SubgraphX
  ([`subgraphx()`](https://adamrsophie.github.io/graphlearnR/reference/subgraphx.md)),
  and attention weights.
- Explanation subgraphs, fidelity, and stability measures.

### Datasets

- Built in: Zachary’s karate club, stochastic block models, and
  synthetic graph, heterogeneous, temporal, spatiotemporal, and
  molecular datasets.
- Downloaders with checksum verification for TU datasets and the LINQS
  citation graphs (Cora, CiteSeer, Pubmed); no third party data is
  bundled.
- Readers for OGB node and graph property datasets with their official
  splits.
- [`molecule_graph()`](https://adamrsophie.github.io/graphlearnR/reference/molecule_graph.md)
  builds molecular graphs from atoms and bonds.

### Optional integrations

- [`graph_diffusion()`](https://adamrsophie.github.io/graphlearnR/reference/graph_diffusion.md)
  simulates diffusion on a graph with the modelcompR ODE solvers.
- [`graph_operator_symbolic()`](https://adamrsophie.github.io/graphlearnR/reference/graph_operator_symbolic.md)
  returns exact adjacency, Laplacian, and GCN operators as symcompR
  matrices.
- [`consistent_labels()`](https://adamrsophie.github.io/graphlearnR/reference/consistent_labels.md)
  finds the most probable labeling that satisfies edge rules with the
  z3r optimizer.
- [`select_links()`](https://adamrsophie.github.io/graphlearnR/reference/select_links.md)
  chooses links under a budget as a 0-1 program in optcompR.
- [`aggregate_jax()`](https://adamrsophie.github.io/graphlearnR/reference/aggregate_jax.md)
  performs neighborhood aggregation with jaxR arrays.
- [`luz_node_dataloader()`](https://adamrsophie.github.io/graphlearnR/reference/luz_node_dataloader.md)
  and `luz_node_module` train node models with luz; `augment()` and
  `glance()` summarize fits for tidymodels.
