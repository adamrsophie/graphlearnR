# graphlearnR 0.1.0

First release.

## Graph data

- `graph_data()` stores homogeneous graphs with 1-based `edge_index`, node and
  edge features, labels, weights, masks, and identifiers. Undirected input is
  given once per pair and stored in both directions with `edge_pair`.
- Conversions with igraph, edge lists, and dense or sparse adjacency matrices;
  `graph_tensors()` moves a graph to torch tensors on any device.
- `graph_batch()` and `unbatch()` combine graphs into one disconnected graph
  and back.
- Transforms: self loops, coalescing, induced subgraphs, symmetrization,
  feature normalization, degree features.

## Message passing and layers

- Scatter aggregations (sum, mean, max, min, std, softmax) and
  `message_passing()` for custom layers.
- Layers: GraphSAGE, GCN, GraphConv, GIN and GINE, GAT, GATv2, graph
  transformer convolution, edge conditioned convolution (NNConv), and RGCN
  with basis decomposition.
- Global add, mean, max, and attention pooling, and top-k pooling.
- `gnn_model()` builds node, edge, link, and graph level models with
  normalization, residual connections, and dropout.

## Training and evaluation

- `train_graph_model()` trains node, edge, link, and graph classification and
  regression models, full batch or with neighbor sampling, with early
  stopping, schedulers, checkpoints, and a reproducibility record.
- `predict()`, `evaluate_graph_model()`, `save_graph_fit()`, and
  `load_graph_fit()`.
- Leakage-safe splits for nodes, edges, edge labels, and graphs (random,
  stratified, group, scaffold).
- Metrics for classification, calibration, regression, ranking, and
  clustering.

## Sampling and scale

- `sample_neighbors()`, `k_hop_subgraph()`, node and graph loaders, negative
  sampling, and corrupted tails for ranking evaluation.

## Heterogeneous, temporal, and transformer models

- `hetero_graph_data()`, reverse relations, typed sampling, typed link splits,
  `hetero_conv()`, and `hetero_gnn_model()`.
- Event based temporal graphs, snapshots, strictly causal temporal neighbors,
  a graph GRU, spatiotemporal forecasting, and dynamic link prediction with
  baselines.
- Laplacian and random walk positional encodings, shortest path distances,
  and a Graphormer style graph transformer.

## Explainability

- Saliency, integrated gradients, edge gradients, occlusion, GNNExplainer,
  attention weights, fidelity, and stability.

## Datasets

- Built in: Zachary's karate club, stochastic block models, and synthetic
  graph, heterogeneous, and temporal datasets.
- Downloaders with checksum verification for TU datasets and the LINQS
  citation graphs (Cora, CiteSeer); nothing third party is bundled.
