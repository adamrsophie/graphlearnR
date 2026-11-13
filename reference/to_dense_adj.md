# Dense adjacency and differentiable pooling

`to_dense_adj()` turns a batched `edge_index` into a `B` by `L` by `L`
tensor, `L` the largest graph, with `adj[b, i, j]` the weight of the
edge from local node `j` to local node `i` of graph `b` (rows are
targets, so `adj %*% x` aggregates incoming messages). Repeated edges
add up.

## Usage

``` r
to_dense_adj(edge_index, batch, edge_weight = NULL)

dense_diff_pool(x, adj, s, mask = NULL)

dense_graph_conv(in_features, out_features, normalize = TRUE)
```

## Arguments

- edge_index:

  Long tensor, 2 by E, 1-based.

- batch:

  Graph membership of each node (R vector or tensor).

- edge_weight:

  Optional edge weights.

- x:

  Dense node features, `B` by `L` by `F`.

- adj:

  Dense adjacency, `B` by `L` by `L`.

- s:

  Assignment logits, `B` by `L` by `C`.

- mask:

  Logical `B` by `L` tensor of real (non padding) nodes.

- in_features, out_features:

  Feature sizes.

- normalize:

  Divide aggregated messages by the weighted in degree.

## Value

`to_dense_adj()`: a tensor. `dense_diff_pool()`: a list with `x`, `adj`,
`s` (the softmax assignment), `link_loss` and `entropy_loss`.

## Details

`dense_diff_pool()` is the DiffPool operator (Ying et al. 2018). With
assignment logits `s` (`B` by `L` by `C`), \\S = \mathrm{softmax}(s)\\
over clusters, padding rows set to zero, \$\$X' = S^\top X,\quad A' =
S^\top A S,\$\$ the auxiliary link loss \\\lVert A - S S^\top \rVert_F /
(B L^2)\\ and entropy loss \\\mathrm{mean}\_i(-\sum_c S\_{ic} \log
S\_{ic})\\.

`dense_graph_conv()` is \\W_1 \tilde A X + W_2 X\\ on dense batches,
\\\tilde A\\ row normalized when `normalize = TRUE`; it is the GNN used
on coarsened graphs.

Memory is \\O(B L^2)\\, so these suit batches of small graphs.

## References

Ying, You, Morris, Ren, Hamilton and Leskovec (2018) Hierarchical graph
representation learning with differentiable pooling.

## Examples

``` r
if (torch::torch_is_installed()) {
  b <- graph_batch(dataset_synthetic_graphs(3, seed = 1))
  tt <- graph_tensors(b)
  adj <- to_dense_adj(tt$edge_index, b$batch)
  d <- to_dense_batch(tt$x, b$batch)
  pooled <- dense_diff_pool(d$x, adj, torch::torch_randn(3, d$L, 2), d$mask)
  dim(pooled$adj)
}
#> [1] 3 2 2
```
