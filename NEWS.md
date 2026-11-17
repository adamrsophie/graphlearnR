# graphlearnR 0.1.0

Initial release. graphlearnR provides graph machine learning for R on top of
the `torch` package: graph data structures, message passing layers, model
builders, training and evaluation, sampling for large graphs,
heterogeneous and temporal graphs, graph transformers, and explainability.

## Graph data

- `graph_data()` stores homogeneous graphs with 1-based `edge_index`, node
  and edge features, labels, weights, masks, and node identifiers.
  Undirected input is given once per pair and stored in both directions.
- Conversion to and from igraph, edge lists, and dense or sparse adjacency
  matrices; `graph_tensors()` moves a graph to torch tensors on any device.
- `graph_batch()` and `unbatch()` combine graphs into one disconnected graph
  and split them again.
- Transforms for self loops, duplicate edges, induced subgraphs,
  symmetrization, feature normalization, and degree features.
- Validation of graph structure, and a shared feature schema for datasets
  (`graph_schema()`).

## Layers and message passing

- `message_passing()` and scatter aggregations (sum, mean, max, min, std,
  softmax) for custom layers.
- Convolutions: GraphSAGE, GCN, GraphConv, GIN and GINE, GAT, GATv2, graph
  transformer convolution, edge conditioned convolution (NNConv), RGCN with
  basis decomposition, ChebConv, APPNP, PNA, and EdgeConv.
- Pooling: global add, mean, max, and attention pooling, top-k pooling,
  SAGPool, and DiffPool with its auxiliary losses.
- `meta_layer()` composes edge, node, and graph level updates into a graph
  network block.
- `gnn_model()` builds node, edge, link, and graph level models with
  normalization, residual connections, and dropout.

## Training and evaluation

- `train_graph_model()` trains classification and regression models for
  nodes, edges, links, and whole graphs, full batch or on sampled
  subgraphs, with early stopping, learning rate schedulers, checkpoints,
  gradient clipping, class weights, and a reproducibility record of the
  settings and split members.
- `predict()`, `evaluate_graph_model()`, `save_graph_fit()`, and
  `load_graph_fit()`.
- Leakage safe splits for nodes, edges, edge labels, and graphs (random,
  stratified, group, and scaffold). Link splits treat parallel edges and
  both directions of an undirected pair as one unit, and remove held out
  pairs from paired reverse relations.
- Metrics for classification, calibration, regression, link ranking
  (filtered Hits@K and MRR), and clustering; `evaluate_embeddings()` with a
  linear probe; `ogb_evaluate()` with the official OGB metrics.

## Sampling for large graphs

- `sample_neighbors()`, `k_hop_subgraph()`, node and graph loaders, negative
  sampling, and corrupted tails for ranking evaluation.
- `cluster_loader()` (Cluster-GCN), `saint_loader()` (GraphSAINT random walks
  with loss and edge normalization), and `ladies_loader()` (layer wise
  importance sampling), used through `train_graph_model(loader = )`.

## Heterogeneous graphs

- `hetero_graph_data()` with reverse relations, metapaths
  (`add_metapath()`), typed neighbor sampling, and typed link splits.
- `hetero_conv()` with sum, mean, or semantic attention across relations, and
  `hetero_gnn_model()` with per relation SAGE, GraphConv, or GAT layers, RGCN,
  HGT (`hgt_conv()`), and HAN.

## Temporal graphs

- Event based temporal graphs, snapshots, and strictly causal temporal
  neighbors.
- Temporal graph networks (`tgn_model()`, `train_tgn()`) with node memory and
  time, graph sum, or multi layer graph attention embeddings.
- Dynamic link prediction with a graph GRU or EvolveGCN-O
  (`evolve_gcn_cell()`), and spatiotemporal forecasting.
- `predict()` for temporal fits uses only information available before the
  prediction time; temporal fits can be saved and loaded.
- HGT relative temporal encoding and continuous depth graph neural ODE blocks
  (`graph_ode_block()`, `graph_ode_model()`).

## Graph transformers

- Laplacian and random walk positional encodings and shortest path distances.
- `graph_transformer_model()`: a Graphormer style transformer with
  shortest path and edge feature attention biases.
- `gps_layer()`: GPS layers combining local message passing with global
  attention, exact or linear (`linear_attention()`, with Performer FAVOR+ and
  elu kernels).

## Explainability

- Saliency, integrated gradients, edge gradients, occlusion, GNNExplainer,
  PGExplainer (`pg_explainer()`), SubgraphX (`subgraphx()`), and attention
  weights.
- Explanation subgraphs, fidelity, and stability measures.

## Datasets

- Built in: Zachary's karate club, stochastic block models, and synthetic
  graph, heterogeneous, temporal, spatiotemporal, and molecular datasets.
- Downloaders with checksum verification for TU datasets and the LINQS
  citation graphs (Cora, CiteSeer, Pubmed); no third party data is bundled.
- Readers for OGB node and graph property datasets with their official
  splits.
- `molecule_graph()` builds molecular graphs from atoms and bonds.

## Optional integrations

- `graph_diffusion()` simulates diffusion on a graph with the modelcompR ODE
  solvers.
- `graph_operator_symbolic()` returns exact adjacency, Laplacian, and GCN
  operators as symcompR matrices.
- `consistent_labels()` finds the most probable labeling that satisfies edge
  rules with the z3r optimizer.
- `select_links()` chooses links under a budget as a 0-1 program in optcompR.
- `aggregate_jax()` performs neighborhood aggregation with jaxR arrays.
- `luz_node_dataloader()` and `luz_node_module` train node models with luz;
  `augment()` and `glance()` summarize fits for tidymodels.
