# Layer wise importance sampled mini batches (LADIES)

Builds node mini batches by layer dependent importance sampling (Zou et
al. 2019), the successor of FastGCN (Chen, Ma and Xiao 2018). For each
batch, `batch_size` training nodes are the seeds (layer 0). Going from
layer \\l\\ to layer \\l + 1\\, candidates are the sources of edges into
the layer \\l\\ nodes, and `layer_sizes[l + 1]` of them are drawn
without replacement with probability \$\$p(u) \propto \sum\_{v \in S_l}
\hat A\_{vu}^2,\$\$ where \\\hat A = D^{-1/2}(A + I)D^{-1/2}\\ (degrees
are weighted in degrees plus one). Layer \\l + 1\\ is the sampled set
plus the layer \\l\\ nodes. Each kept edge \\u \to v\\ gets the weight
\\\hat A\_{vu} / q(u)\\, where \\q(u) = \min(1, s\\p(u))\\ approximates
the inclusion probability for `s` draws (and is one when every candidate
is kept), normalized to sum to one over the incoming edges of \\v\\ (the
row normalization of LADIES). Self loops are not part of the sampled
operator; the root weight of the layer plays that role.

## Usage

``` r
ladies_loader(
  g,
  batch_size = 64L,
  layer_sizes = c(256L, 256L),
  shuffle = TRUE,
  seed = NULL
)
```

## Arguments

- g:

  A `graph_data` object with `train_mask` (training passes its own masks
  to the loader).

- batch_size:

  Seed nodes per batch.

- layer_sizes:

  Nodes sampled per layer, one entry per model layer.

- shuffle:

  Shuffle the seeds each epoch.

- seed:

  Optional integer seed; batches of epoch \\e\\ use `seed + e`.

## Value

A `ladies_loader`, to pass as `loader` to
[`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
or to
[`loader_batches()`](https://adamrsophie.github.io/graphlearnR/reference/node_loader.md).

## Details

The batch is one `graph_data` subgraph that holds the edges of all
layers together, so it works with any model in
[`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md);
this is a common simplification of the layer by layer bipartite blocks
of the paper. Each node is a target once, in the first layer that
reaches it: layer \\l + 1\\ samples sources only for the nodes newly
added at layer \\l\\, so the incoming weights of every target sum to
exactly one and the same operator is used at every model layer. The
weights of the incoming edges of each target sum to one, so they already
define a weighted mean: use a model that **sums** weighted messages,
such as `gnn_model("graph_conv")`. A mean aggregator (the default of
`"sage"`) divides by the sampled count a second time, and `gcn_conv`
renormalizes the weights again. Validation and prediction run on the
full graph with its own edge weights, so the operator seen in training
approximates, but is not identical to, the one used there. Only the
seeds carry `train_mask`, so the loss uses the seeds only.

## References

Zou et al. (2019) Layer-dependent importance sampling for training deep
and large graph convolutional networks. NeurIPS. Chen, Ma and Xiao
(2018) FastGCN: fast learning with graph convolutional networks via
importance sampling. ICLR.

## Examples

``` r
g <- split_nodes(dataset_karate(), seed = 1)
ld <- ladies_loader(g, batch_size = 8, layer_sizes = c(10, 10), seed = 1)
b <- loader_batches(ld, epoch = 1)[[1]]
c(nodes = b$num_nodes, seeds = sum(b$train_mask))
#> nodes seeds 
#>    19     8 
```
