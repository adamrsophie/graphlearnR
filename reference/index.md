# Package index

## Package

- [`graphlearnR`](https://adamrsophie.github.io/graphlearnR/reference/graphlearnR-package.md)
  [`graphlearnR-package`](https://adamrsophie.github.io/graphlearnR/reference/graphlearnR-package.md)
  : graphlearnR: Graph Machine Learning and Geometric Deep Learning in R

## Graph data

Create, validate, inspect, and index homogeneous graphs.

- [`graph_data()`](https://adamrsophie.github.io/graphlearnR/reference/graph_data.md)
  : Create a graph data object
- [`validate_graph()`](https://adamrsophie.github.io/graphlearnR/reference/validate_graph.md)
  : Validate a graph data object
- [`num_nodes()`](https://adamrsophie.github.io/graphlearnR/reference/graph-queries.md)
  [`num_edges()`](https://adamrsophie.github.io/graphlearnR/reference/graph-queries.md)
  [`num_node_features()`](https://adamrsophie.github.io/graphlearnR/reference/graph-queries.md)
  [`num_edge_features()`](https://adamrsophie.github.io/graphlearnR/reference/graph-queries.md)
  [`is_directed_graph()`](https://adamrsophie.github.io/graphlearnR/reference/graph-queries.md)
  [`has_self_loops()`](https://adamrsophie.github.io/graphlearnR/reference/graph-queries.md)
  [`has_isolated_nodes()`](https://adamrsophie.github.io/graphlearnR/reference/graph-queries.md)
  [`has_duplicate_edges()`](https://adamrsophie.github.io/graphlearnR/reference/graph-queries.md)
  [`degree()`](https://adamrsophie.github.io/graphlearnR/reference/graph-queries.md)
  : Basic graph queries
- [`to_zero_based()`](https://adamrsophie.github.io/graphlearnR/reference/to_zero_based.md)
  [`from_zero_based()`](https://adamrsophie.github.io/graphlearnR/reference/to_zero_based.md)
  : Convert one based indices to zero based

## Conversion

Move graphs between igraph, edge lists, adjacency matrices, and tensors.

- [`from_igraph()`](https://adamrsophie.github.io/graphlearnR/reference/from_igraph.md)
  [`to_igraph()`](https://adamrsophie.github.io/graphlearnR/reference/from_igraph.md)
  : Convert between igraph and graph_data
- [`graph_from_edgelist()`](https://adamrsophie.github.io/graphlearnR/reference/graph_from_edgelist.md)
  [`edgelist_from_graph()`](https://adamrsophie.github.io/graphlearnR/reference/graph_from_edgelist.md)
  : Build a graph from an edge list
- [`graph_from_adjacency()`](https://adamrsophie.github.io/graphlearnR/reference/graph_from_adjacency.md)
  [`adjacency_from_graph()`](https://adamrsophie.github.io/graphlearnR/reference/graph_from_adjacency.md)
  : Build a graph from an adjacency matrix
- [`graph_tensors()`](https://adamrsophie.github.io/graphlearnR/reference/graph_tensors.md)
  : Convert graph data to torch tensors

## Batching and transforms

- [`graph_batch()`](https://adamrsophie.github.io/graphlearnR/reference/graph_batch.md)
  [`unbatch()`](https://adamrsophie.github.io/graphlearnR/reference/graph_batch.md)
  : Batch graphs into one disconnected graph
- [`add_self_loops()`](https://adamrsophie.github.io/graphlearnR/reference/graph-transforms.md)
  [`remove_self_loops()`](https://adamrsophie.github.io/graphlearnR/reference/graph-transforms.md)
  [`coalesce_edges()`](https://adamrsophie.github.io/graphlearnR/reference/graph-transforms.md)
  [`induced_subgraph()`](https://adamrsophie.github.io/graphlearnR/reference/graph-transforms.md)
  [`as_undirected()`](https://adamrsophie.github.io/graphlearnR/reference/graph-transforms.md)
  [`normalize_features()`](https://adamrsophie.github.io/graphlearnR/reference/graph-transforms.md)
  [`add_degree_features()`](https://adamrsophie.github.io/graphlearnR/reference/graph-transforms.md)
  [`compose_transforms()`](https://adamrsophie.github.io/graphlearnR/reference/graph-transforms.md)
  : Graph transformations
- [`to_dense_batch()`](https://adamrsophie.github.io/graphlearnR/reference/to_dense_batch.md)
  [`from_dense_batch()`](https://adamrsophie.github.io/graphlearnR/reference/to_dense_batch.md)
  : Dense batch layout

## Message passing

Scatter aggregation and the general message passing operator.

- [`scatter_sum()`](https://adamrsophie.github.io/graphlearnR/reference/scatter.md)
  [`scatter_count()`](https://adamrsophie.github.io/graphlearnR/reference/scatter.md)
  [`scatter_mean()`](https://adamrsophie.github.io/graphlearnR/reference/scatter.md)
  [`scatter_max()`](https://adamrsophie.github.io/graphlearnR/reference/scatter.md)
  [`scatter_min()`](https://adamrsophie.github.io/graphlearnR/reference/scatter.md)
  [`scatter_std()`](https://adamrsophie.github.io/graphlearnR/reference/scatter.md)
  [`scatter_softmax()`](https://adamrsophie.github.io/graphlearnR/reference/scatter.md)
  [`scatter_aggregate()`](https://adamrsophie.github.io/graphlearnR/reference/scatter.md)
  : Scatter aggregation over groups
- [`message_passing()`](https://adamrsophie.github.io/graphlearnR/reference/message_passing.md)
  : Generic message passing
- [`add_self_loops_tensor()`](https://adamrsophie.github.io/graphlearnR/reference/add_self_loops_tensor.md)
  [`remove_self_loops_tensor()`](https://adamrsophie.github.io/graphlearnR/reference/add_self_loops_tensor.md)
  [`gcn_norm()`](https://adamrsophie.github.io/graphlearnR/reference/add_self_loops_tensor.md)
  : Self loop and normalization helpers

## Layers and pooling

- [`sage_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)
  [`gcn_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)
  [`graph_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)
  [`gin_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)
  [`gat_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)
  [`gatv2_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)
  [`transformer_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)
  [`nn_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)
  [`rgcn_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md)
  : Graph neural network layers
- [`graph_mlp()`](https://adamrsophie.github.io/graphlearnR/reference/graph_mlp.md)
  : Multilayer perceptron
- [`global_add_pool()`](https://adamrsophie.github.io/graphlearnR/reference/global-pooling.md)
  [`global_mean_pool()`](https://adamrsophie.github.io/graphlearnR/reference/global-pooling.md)
  [`global_max_pool()`](https://adamrsophie.github.io/graphlearnR/reference/global-pooling.md)
  : Global pooling (graph readout)
- [`global_attention_pool()`](https://adamrsophie.github.io/graphlearnR/reference/global_attention_pool.md)
  : Global attention pooling
- [`topk_pool()`](https://adamrsophie.github.io/graphlearnR/reference/topk_pool.md)
  : Top-k pooling (hierarchical, experimental)
- [`to_dense_adj()`](https://adamrsophie.github.io/graphlearnR/reference/to_dense_adj.md)
  [`dense_diff_pool()`](https://adamrsophie.github.io/graphlearnR/reference/to_dense_adj.md)
  [`dense_graph_conv()`](https://adamrsophie.github.io/graphlearnR/reference/to_dense_adj.md)
  : Dense adjacency and differentiable pooling
- [`diffpool_model()`](https://adamrsophie.github.io/graphlearnR/reference/diffpool_model.md)
  : DiffPool graph classification model

## Models and decoders

- [`gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/gnn_model.md)
  : Build a graph neural network
- [`link_decoder()`](https://adamrsophie.github.io/graphlearnR/reference/link_decoder.md)
  [`edge_decoder()`](https://adamrsophie.github.io/graphlearnR/reference/link_decoder.md)
  : Link and edge decoders
- [`graph_ode_block()`](https://adamrsophie.github.io/graphlearnR/reference/graph_ode_block.md)
  [`graph_ode_model()`](https://adamrsophie.github.io/graphlearnR/reference/graph_ode_block.md)
  : Continuous depth graph diffusion (graph neural ODE)

## Sampling and loaders

- [`sample_neighbors()`](https://adamrsophie.github.io/graphlearnR/reference/sample_neighbors.md)
  : Multi-hop neighbor sampling
- [`k_hop_subgraph()`](https://adamrsophie.github.io/graphlearnR/reference/k_hop_subgraph.md)
  : K-hop subgraph
- [`node_loader()`](https://adamrsophie.github.io/graphlearnR/reference/node_loader.md)
  [`graph_loader()`](https://adamrsophie.github.io/graphlearnR/reference/node_loader.md)
  [`loader_batches()`](https://adamrsophie.github.io/graphlearnR/reference/node_loader.md)
  : Mini-batch loaders
- [`cluster_loader()`](https://adamrsophie.github.io/graphlearnR/reference/cluster_loader.md)
  [`saint_loader()`](https://adamrsophie.github.io/graphlearnR/reference/cluster_loader.md)
  : Cluster and GraphSAINT subgraph loaders
- [`negative_sampling()`](https://adamrsophie.github.io/graphlearnR/reference/negative_sampling.md)
  : Negative edge sampling
- [`corrupt_tails()`](https://adamrsophie.github.io/graphlearnR/reference/corrupt_tails.md)
  : Corrupted tails for ranking evaluation

## Splits

Leakage safe splits for every task.

- [`split_nodes()`](https://adamrsophie.github.io/graphlearnR/reference/split_nodes.md)
  : Split nodes into training, validation and test sets
- [`split_edges()`](https://adamrsophie.github.io/graphlearnR/reference/split_edges.md)
  : Split edges for link prediction
- [`split_edge_labels()`](https://adamrsophie.github.io/graphlearnR/reference/split_edge_labels.md)
  : Split labeled edges for edge classification or regression
- [`split_graphs()`](https://adamrsophie.github.io/graphlearnR/reference/split_graphs.md)
  : Split graphs for graph level prediction

## Training, prediction, and evaluation

- [`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
  : Train a graph neural network
- [`predict(`*`<graph_fit>`*`)`](https://adamrsophie.github.io/graphlearnR/reference/predict.graph_fit.md)
  : Predict with a trained graph model
- [`evaluate_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/evaluate_graph_model.md)
  : Evaluate a trained graph model
- [`evaluate_predictions()`](https://adamrsophie.github.io/graphlearnR/reference/evaluate_predictions.md)
  : Evaluate predictions for a task
- [`save_graph_fit()`](https://adamrsophie.github.io/graphlearnR/reference/save_graph_fit.md)
  [`load_graph_fit()`](https://adamrsophie.github.io/graphlearnR/reference/save_graph_fit.md)
  : Save and load a trained model
- [`augment(`*`<graph_fit>`*`)`](https://adamrsophie.github.io/graphlearnR/reference/graph_fit-tidy.md)
  [`glance(`*`<graph_fit>`*`)`](https://adamrsophie.github.io/graphlearnR/reference/graph_fit-tidy.md)
  : Tidy summaries of a fitted graph model
- [`luz_node_dataloader()`](https://adamrsophie.github.io/graphlearnR/reference/luz_node_dataloader.md)
  [`luz_node_module()`](https://adamrsophie.github.io/graphlearnR/reference/luz_node_dataloader.md)
  : Training node models with luz
- [`metric_accuracy()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_balanced_accuracy()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_precision()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_recall()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_f1()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_roc_auc()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_average_precision()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_ece()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_brier()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_mse()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_rmse()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_mae()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_r2()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_hits_at_k()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_mrr()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_nmi()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  [`metric_ari()`](https://adamrsophie.github.io/graphlearnR/reference/metrics.md)
  : Evaluation metrics

## Heterogeneous graphs

- [`hetero_graph_data()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_graph_data.md)
  [`validate_hetero_graph()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_graph_data.md)
  : Heterogeneous graph data
- [`edge_types()`](https://adamrsophie.github.io/graphlearnR/reference/edge_types.md)
  : Relation table of a heterogeneous graph
- [`add_reverse_relations()`](https://adamrsophie.github.io/graphlearnR/reference/add_reverse_relations.md)
  : Add reverse relations
- [`to_homogeneous()`](https://adamrsophie.github.io/graphlearnR/reference/to_homogeneous.md)
  : Convert a heterogeneous graph to a homogeneous one
- [`hetero_tensors()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_tensors.md)
  : Heterogeneous graph tensors
- [`hetero_batch()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_batch.md)
  : Batch heterogeneous graphs
- [`sample_hetero_neighbors()`](https://adamrsophie.github.io/graphlearnR/reference/sample_hetero_neighbors.md)
  : Typed neighbor sampling
- [`split_edges_hetero()`](https://adamrsophie.github.io/graphlearnR/reference/split_edges_hetero.md)
  : Split one relation of a heterogeneous graph for link prediction
- [`hetero_conv()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_conv.md)
  : Heterogeneous convolution wrapper
- [`hgt_conv()`](https://adamrsophie.github.io/graphlearnR/reference/hgt_conv.md)
  : Heterogeneous graph transformer convolution
- [`hetero_gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_gnn_model.md)
  : Heterogeneous graph neural network

## Temporal graphs

- [`temporal_graph_data()`](https://adamrsophie.github.io/graphlearnR/reference/temporal_graph_data.md)
  [`validate_temporal_graph()`](https://adamrsophie.github.io/graphlearnR/reference/temporal_graph_data.md)
  : Temporal graph representations
- [`temporal_window()`](https://adamrsophie.github.io/graphlearnR/reference/temporal_window.md)
  : Select events in a time window
- [`split_temporal()`](https://adamrsophie.github.io/graphlearnR/reference/split_temporal.md)
  : Temporal train, validation and test split
- [`temporal_neighbors()`](https://adamrsophie.github.io/graphlearnR/reference/temporal_neighbors.md)
  : Most recent temporal neighbors
- [`snapshot_sequence()`](https://adamrsophie.github.io/graphlearnR/reference/snapshot_sequence.md)
  [`temporal_snapshots()`](https://adamrsophie.github.io/graphlearnR/reference/snapshot_sequence.md)
  : Snapshot sequences
- [`spatiotemporal_signal()`](https://adamrsophie.github.io/graphlearnR/reference/spatiotemporal_signal.md)
  : Spatiotemporal signals on a fixed graph
- [`graph_gru_cell()`](https://adamrsophie.github.io/graphlearnR/reference/graph_gru_cell.md)
  : Recurrent graph convolution cell
- [`temporal_gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/temporal_gnn_model.md)
  : Recurrent temporal GNN
- [`train_spatiotemporal_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_spatiotemporal_model.md)
  : Train a spatiotemporal forecaster
- [`train_dynamic_link_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_dynamic_link_model.md)
  : Dynamic link prediction on snapshots
- [`dynamic_link_baseline()`](https://adamrsophie.github.io/graphlearnR/reference/dynamic_link_baseline.md)
  : Heuristic baseline for dynamic link prediction
- [`tgn_model()`](https://adamrsophie.github.io/graphlearnR/reference/tgn_model.md)
  [`train_tgn()`](https://adamrsophie.github.io/graphlearnR/reference/tgn_model.md)
  : Temporal graph network with node memory

## Graph transformers and encodings

- [`laplacian_pe()`](https://adamrsophie.github.io/graphlearnR/reference/positional-encodings.md)
  [`random_walk_pe()`](https://adamrsophie.github.io/graphlearnR/reference/positional-encodings.md)
  [`shortest_path_distances()`](https://adamrsophie.github.io/graphlearnR/reference/positional-encodings.md)
  [`random_sign_flip()`](https://adamrsophie.github.io/graphlearnR/reference/positional-encodings.md)
  [`add_positional_encoding()`](https://adamrsophie.github.io/graphlearnR/reference/positional-encodings.md)
  : Structural and positional encodings
- [`graph_transformer_layer()`](https://adamrsophie.github.io/graphlearnR/reference/graph_transformer_layer.md)
  : Graph transformer layer with structural attention bias
- [`graph_transformer_model()`](https://adamrsophie.github.io/graphlearnR/reference/graph_transformer_model.md)
  : Graphormer style graph transformer

## Explainability

- [`explain_saliency()`](https://adamrsophie.github.io/graphlearnR/reference/explain.md)
  [`explain_integrated_gradients()`](https://adamrsophie.github.io/graphlearnR/reference/explain.md)
  [`explain_edges()`](https://adamrsophie.github.io/graphlearnR/reference/explain.md)
  [`explain_occlusion()`](https://adamrsophie.github.io/graphlearnR/reference/explain.md)
  [`gnn_explainer()`](https://adamrsophie.github.io/graphlearnR/reference/explain.md)
  : Explain node predictions
- [`explanation_fidelity()`](https://adamrsophie.github.io/graphlearnR/reference/explanation_fidelity.md)
  [`explanation_stability()`](https://adamrsophie.github.io/graphlearnR/reference/explanation_fidelity.md)
  : Fidelity and stability of an explanation
- [`explanation_subgraph()`](https://adamrsophie.github.io/graphlearnR/reference/explanation_subgraph.md)
  : Explanation subgraph
- [`attention_weights()`](https://adamrsophie.github.io/graphlearnR/reference/attention_weights.md)
  : Attention weights of attention layers

## Datasets

- [`graph_dataset()`](https://adamrsophie.github.io/graphlearnR/reference/graph_dataset.md)
  [`graph_dataset_disk()`](https://adamrsophie.github.io/graphlearnR/reference/graph_dataset.md)
  [`write_graph_dataset()`](https://adamrsophie.github.io/graphlearnR/reference/graph_dataset.md)
  : Graph datasets
- [`dataset_karate()`](https://adamrsophie.github.io/graphlearnR/reference/dataset_karate.md)
  : Zachary's karate club
- [`dataset_sbm()`](https://adamrsophie.github.io/graphlearnR/reference/dataset_sbm.md)
  : Stochastic block model node classification data
- [`dataset_synthetic_graphs()`](https://adamrsophie.github.io/graphlearnR/reference/dataset_synthetic_graphs.md)
  : Synthetic graph classification and regression data
- [`dataset_synthetic_hetero()`](https://adamrsophie.github.io/graphlearnR/reference/dataset_synthetic_hetero.md)
  : Synthetic heterogeneous drug and protein graph
- [`dataset_synthetic_temporal()`](https://adamrsophie.github.io/graphlearnR/reference/dataset_synthetic_temporal.md)
  [`dataset_synthetic_signal()`](https://adamrsophie.github.io/graphlearnR/reference/dataset_synthetic_temporal.md)
  : Synthetic temporal data
- [`download_tu_dataset()`](https://adamrsophie.github.io/graphlearnR/reference/download-datasets.md)
  [`read_tu_dataset()`](https://adamrsophie.github.io/graphlearnR/reference/download-datasets.md)
  [`download_planetoid_raw()`](https://adamrsophie.github.io/graphlearnR/reference/download-datasets.md)
  [`read_linqs_citation()`](https://adamrsophie.github.io/graphlearnR/reference/download-datasets.md)
  : Download and read public benchmark datasets
- [`read_ogb_node()`](https://adamrsophie.github.io/graphlearnR/reference/read_ogb_node.md)
  [`read_ogb_graph()`](https://adamrsophie.github.io/graphlearnR/reference/read_ogb_node.md)
  [`download_ogb()`](https://adamrsophie.github.io/graphlearnR/reference/read_ogb_node.md)
  : Open Graph Benchmark datasets

## Molecules

Molecular graphs from atoms and bonds.

- [`molecule_graph()`](https://adamrsophie.github.io/graphlearnR/reference/molecule_graph.md)
  [`dataset_synthetic_molecules()`](https://adamrsophie.github.io/graphlearnR/reference/molecule_graph.md)
  : Molecular graphs from atoms and bonds

## Sibling packages

Optional bridges to modelcompR, symcompR, z3r, optcompR and jaxR.

- [`graph_diffusion()`](https://adamrsophie.github.io/graphlearnR/reference/graph_diffusion.md)
  : Diffusion on a graph, solved with modelcompR
- [`graph_operator_symbolic()`](https://adamrsophie.github.io/graphlearnR/reference/graph_operator_symbolic.md)
  : Exact graph operators with symcompR
- [`consistent_labels()`](https://adamrsophie.github.io/graphlearnR/reference/consistent_labels.md)
  : Rule constrained node labels with z3r
- [`select_links()`](https://adamrsophie.github.io/graphlearnR/reference/select_links.md)
  : Budgeted link selection with optcompR
- [`aggregate_jax()`](https://adamrsophie.github.io/graphlearnR/reference/aggregate_jax.md)
  : Neighborhood aggregation with jaxR
