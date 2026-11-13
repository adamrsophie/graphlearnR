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

- `cheb_conv()` (ChebConv), `pna_conv()` with `pna_degree()` (PNA), `appnp()`
  and `appnp_model()` (APPNP), and `sag_pool()` (SAGPool).
- Scatter aggregations (sum, mean, max, min, std, softmax) and
  `message_passing()` for custom layers.
- Layers: GraphSAGE, GCN, GraphConv, GIN and GINE, GAT, GATv2, graph
  transformer convolution, edge conditioned convolution (NNConv), and RGCN
  with basis decomposition.
- Global add, mean, max, and attention pooling, top-k pooling, and DiffPool
  (`dense_diff_pool()`, `to_dense_adj()`, `dense_graph_conv()`,
  `diffpool_model()`), whose auxiliary losses the trainer adds.
- `hgt_conv()`: heterogeneous graph transformer layer, also available as
  `hetero_gnn_model(conv = "hgt")`.
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
- `cluster_loader()` (Cluster-GCN) and `saint_loader()` (GraphSAINT random
  walks with loss and edge normalization), used through
  `train_graph_model(loader = )`.

## Heterogeneous, temporal, and transformer models

- `hetero_graph_data()`, reverse relations, typed sampling, typed link splits,
  `hetero_conv()`, and `hetero_gnn_model()`.
- Event based temporal graphs, snapshots, strictly causal temporal neighbors,
  a graph GRU, spatiotemporal forecasting, and dynamic link prediction with
  baselines.
- `tgn_model()` and `train_tgn()`: temporal graph networks with node memory
  and time, graph sum, or graph attention embeddings.
- `hgt_conv(temporal_encoding = TRUE)` and the `edge_time` field of
  heterogeneous relations add HGT's relative temporal encoding.
- `graph_ode_block()` and `graph_ode_model()`: GRAND style continuous depth
  diffusion with fixed step Runge-Kutta or Euler integration.
- Laplacian and random walk positional encodings, shortest path distances,
  and a Graphormer style graph transformer.

## Explainability

- Saliency, integrated gradients, edge gradients, occlusion, GNNExplainer,
  attention weights, explanation subgraphs, fidelity, and stability.

## Datasets

- Built in: Zachary's karate club, stochastic block models, and synthetic
  graph, heterogeneous, and temporal datasets.
- Downloaders with checksum verification for TU datasets and the LINQS
  citation graphs (Cora, CiteSeer); nothing third party is bundled.
- `download_ogb()`, `read_ogb_node()`, and `read_ogb_graph()` for OGB node
  and graph property datasets in their CSV layout, with official splits.

## Molecules

- `molecule_graph()` builds graphs from atoms and bonds with element and
  bond order features; `dataset_synthetic_molecules()` generates small
  synthetic molecules with scaffold identifiers for scaffold splits.

## Models, evaluation and data (additions)

- `meta_layer()` composes edge, node and graph level updates (graph network
  block, as PyG's `MetaLayer`).
- `predict()` for temporal fits: TGN continues from its stored memory and
  neighbor store, dynamic link models score from earlier snapshots, and
  spatiotemporal models forecast from given origins; temporal fits can be
  saved and loaded.
- Heterogeneous node tasks train on typed neighbor samples with `fanouts`
  and `batch_size`.
- `evaluate_embeddings()` (linear probe and clustering), average precision
  in classification metrics, filtered per positive link ranking in
  `evaluate_graph_model()`, and `ogb_evaluate()` with the official OGB
  metrics; the `bce_masked` loss trains multi task targets with missing
  labels.
- Datasets check a shared feature schema (`graph_schema()`);
  `download_pubmed_raw()` and `read_linqs_pubmed()` read Pubmed-Diabetes.
- igraph conversion keeps feature names, label matrices, edge masks, graph
  labels, graph attributes and metadata; `unbatch()` rebuilds heterogeneous
  batches.
- The reproducibility record keeps exact split members and all training
  settings.
- `graph_transformer_model()` caches shortest path distances of recent
  batches.

## Correctness and safety behavior

- `split_edges()` and `split_edges_hetero()` treat parallel duplicates (and,
  for undirected or same type relations, both orientations) as one unit, so
  a held out pair never stays in the training graph.
- Masks are checked for pairwise overlap, including edge masks;
  `train_spatiotemporal_model()` purges windows that cross split boundaries.
- Class labels share one level set across batches, splits and predictions;
  metrics check that truth and prediction lengths match.
- TGN updates memory in event order, never draws the positive pair as a
  negative, keeps events with one timestamp in one batch, and checks that
  train, validation and test periods do not overlap.
- Saved fits keep link decoders and heterogeneous link schemas; edge
  predictions for given pairs use the features of the matching edges.
- Explainers restore the model's training mode and parameter gradient flags;
  `gnn_explainer()` supports regression and rejects models that ignore edge
  weights.
- Sub seeds are drawn with `sample.int()`, so seed arithmetic never overflows.
- Training errors on mini batch options a task would ignore (`fanouts`,
  `batch_size`, `loader`), and `write_graph_dataset()` will not mix files
  with an existing dataset unless `overwrite = TRUE`.
- Downloads without a recorded checksum warn; OGB archives use https.

## Sibling packages (optional)

- `graph_diffusion()` simulates diffusion on a graph with the modelcompR ODE
  solvers.
- `graph_operator_symbolic()` returns exact adjacency, Laplacian, and GCN
  operators as symcompR matrices.
- `consistent_labels()` finds the most probable labeling that obeys edge
  rules with the z3r optimizer.
- `select_links()` chooses links under a budget as a 0-1 program in optcompR.
- `aggregate_jax()` performs neighborhood aggregation with jaxR arrays.
- `luz_node_dataloader()` and `luz_node_module` train node models with luz;
  `augment()` and `glance()` methods summarize fits for tidymodels.
