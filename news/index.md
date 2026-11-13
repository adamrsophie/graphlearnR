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

### Models, evaluation and data (additions)

- [`edge_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)
  (EdgeConv of dynamic graph CNNs, max over \[x_i, x_j - x_i\]
  messages).
- HAN: `hetero_gnn_model(conv = "han")` with semantic attention across
  relations (`hetero_conv(aggr = "semantic")`), and
  [`add_metapath()`](https://adamrsophie.github.io/graphlearnR/reference/add_metapath.md)
  to compose relations into metapaths.
- [`gps_layer()`](https://adamrsophie.github.io/graphlearnR/reference/gps_layer.md):
  GPS graph transformer layer (local message passing plus masked global
  attention per graph).
- [`meta_layer()`](https://adamrsophie.github.io/graphlearnR/reference/meta_layer.md)
  composes edge, node and graph level updates (graph network block, as
  PyG’s `MetaLayer`).
- [`evolve_gcn_cell()`](https://adamrsophie.github.io/graphlearnR/reference/evolve_gcn_cell.md)
  (EvolveGCN-O: GCN weights evolve with a GRU across snapshots), usable
  in `train_dynamic_link_model(cell = "evolve")`.
- [`predict()`](https://rdrr.io/r/stats/predict.html) for temporal fits:
  TGN continues from its stored memory and neighbor store, dynamic link
  models score from earlier snapshots, and spatiotemporal models
  forecast from given origins; temporal fits can be saved and loaded.
- Heterogeneous node tasks train on typed neighbor samples with
  `fanouts` and `batch_size`.
- [`ladies_loader()`](https://adamrsophie.github.io/graphlearnR/reference/ladies_loader.md):
  layer wise importance sampled node batches (LADIES, the successor of
  FastGCN) for `train_graph_model(loader = )`.
- [`evaluate_embeddings()`](https://adamrsophie.github.io/graphlearnR/reference/evaluate_embeddings.md)
  (linear probe and clustering), average precision in classification
  metrics, filtered per positive link ranking in
  [`evaluate_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/evaluate_graph_model.md),
  and
  [`ogb_evaluate()`](https://adamrsophie.github.io/graphlearnR/reference/ogb_evaluate.md)
  with the official OGB metrics; the `bce_masked` loss trains multi task
  targets with missing labels.
- Datasets check a shared feature schema
  ([`graph_schema()`](https://adamrsophie.github.io/graphlearnR/reference/graph_dataset.md));
  [`download_pubmed_raw()`](https://adamrsophie.github.io/graphlearnR/reference/download-datasets.md)
  and
  [`read_linqs_pubmed()`](https://adamrsophie.github.io/graphlearnR/reference/download-datasets.md)
  read Pubmed-Diabetes.
- igraph conversion keeps feature names, label matrices, edge masks,
  graph labels, graph attributes and metadata;
  [`unbatch()`](https://adamrsophie.github.io/graphlearnR/reference/graph_batch.md)
  rebuilds heterogeneous batches.
- The reproducibility record keeps exact split members and all training
  settings.
- [`graph_transformer_model()`](https://adamrsophie.github.io/graphlearnR/reference/graph_transformer_model.md)
  caches shortest path distances of recent batches and, with `edge_dim`,
  adds a learned attention bias from edge features (Graphormer’s edge
  encoding for direct edges).
- [`scatter_std()`](https://adamrsophie.github.io/graphlearnR/reference/scatter.md)
  (and the PNA `std` aggregator) now returns 0 for empty and constant
  groups, as PyTorch Geometric’s `StdAggregation`; PNA layers are
  compared with PyG’s `PNAConv` in the benchmark workflow.

### Correctness and safety behavior

- [`split_edges()`](https://adamrsophie.github.io/graphlearnR/reference/split_edges.md)
  and
  [`split_edges_hetero()`](https://adamrsophie.github.io/graphlearnR/reference/split_edges_hetero.md)
  treat parallel duplicates (and, for undirected or same type relations,
  both orientations) as one unit, so a held out pair never stays in the
  training graph.
- Masks are checked for pairwise overlap, including edge masks;
  [`train_spatiotemporal_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_spatiotemporal_model.md)
  purges windows that cross split boundaries.
- Class labels share one level set across batches, splits and
  predictions; metrics check that truth and prediction lengths match.
- TGN updates memory in event order, never draws the positive pair as a
  negative, keeps events with one timestamp in one batch, and checks
  that train, validation and test periods do not overlap.
- Saved fits keep link decoders and heterogeneous link schemas; edge
  predictions for given pairs use the features of the matching edges.
- Explainers restore the model’s training mode and parameter gradient
  flags;
  [`gnn_explainer()`](https://adamrsophie.github.io/graphlearnR/reference/explain.md)
  supports regression and rejects models that ignore edge weights.
- Sub seeds are drawn with
  [`sample.int()`](https://rdrr.io/r/base/sample.html), so seed
  arithmetic never overflows.
- Training errors on mini batch options a task would ignore (`fanouts`,
  `batch_size`, `loader`), and
  [`write_graph_dataset()`](https://adamrsophie.github.io/graphlearnR/reference/graph_dataset.md)
  will not mix files with an existing dataset unless `overwrite = TRUE`.
- Downloads without a recorded checksum warn; OGB archives use https.

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
