# Graph neural network layers

Layers are `torch` modules built on
[`message_passing()`](https://adamrsophie.github.io/graphlearnR/reference/message_passing.md).
Every layer takes node features `x` (an N by F tensor, or
`list(x_src, x_dst)` for bipartite relations where noted) and a one
based long `edge_index` (2 by E, row 1 source, row 2 target), and
returns one row per target node. Messages flow from source to target, so
a node aggregates over its **incoming** edges. For undirected graphs,
[`graph_data()`](https://adamrsophie.github.io/graphlearnR/reference/graph_data.md)
stores both directions, so this is the usual neighborhood.

## Usage

``` r
sage_conv(
  in_features,
  out_features,
  aggr = "mean",
  root_weight = TRUE,
  normalize = FALSE,
  bias = TRUE
)

gcn_conv(
  in_features,
  out_features,
  add_self_loops = TRUE,
  normalize = TRUE,
  improved = FALSE,
  bias = TRUE
)

graph_conv(in_features, out_features, aggr = "sum", bias = TRUE)

gin_conv(nn, eps = 0, train_eps = FALSE, edge_dim = NULL, in_features = NULL)

gat_conv(
  in_features,
  out_features,
  heads = 1L,
  concat = TRUE,
  negative_slope = 0.2,
  dropout = 0,
  add_self_loops = TRUE,
  edge_dim = NULL,
  bias = TRUE
)

gatv2_conv(
  in_features,
  out_features,
  heads = 1L,
  concat = TRUE,
  negative_slope = 0.2,
  dropout = 0,
  add_self_loops = TRUE,
  edge_dim = NULL,
  bias = TRUE
)

transformer_conv(
  in_features,
  out_features,
  heads = 1L,
  concat = TRUE,
  dropout = 0,
  edge_dim = NULL,
  root_weight = TRUE,
  bias = TRUE
)

nn_conv(
  in_features,
  out_features,
  edge_dim = NULL,
  edge_nn = NULL,
  aggr = "sum",
  root_weight = TRUE,
  bias = TRUE
)

edge_conv(edge_conv_nn, aggr = "max")

rgcn_conv(
  in_features,
  out_features,
  num_relations,
  num_bases = NULL,
  aggr = "mean",
  root_weight = TRUE,
  bias = TRUE
)
```

## Arguments

- in_features:

  Input feature size. `sage_conv`, `graph_conv`, `gat_conv`,
  `gatv2_conv` and `transformer_conv` also accept `c(source, target)`
  sizes for bipartite input.

- out_features:

  Output feature size (per head for attention layers).

- aggr:

  Aggregation, see
  [`scatter_aggregate()`](https://adamrsophie.github.io/graphlearnR/reference/scatter.md).

- root_weight:

  Add a transformed copy of the target node's own features.

- normalize:

  For `sage_conv`, L2 normalize the output rows; for `gcn_conv`, apply
  the symmetric degree normalization.

- bias:

  Learn an additive bias.

- add_self_loops:

  Add self loops before propagation (never for a pair of source and
  target feature sets, whose index spaces differ).

- improved:

  Self loop weight 2 instead of 1 (`gcn_conv`).

- nn:

  A module applied after aggregation (`gin_conv`), for example an MLP
  from
  [`graph_mlp()`](https://adamrsophie.github.io/graphlearnR/reference/graph_mlp.md).
  For `gin_conv` its input size must equal the node feature size.

- eps:

  Initial \\\epsilon\\.

- train_eps:

  Learn \\\epsilon\\.

- edge_dim:

  Edge feature size. For `gin_conv` this turns on the GINE variant,
  \\\sum_j \mathrm{ReLU}(x_j + W_e e\_{ji})\\; for attention layers edge
  features enter the attention scores.

- heads:

  Number of attention heads.

- concat:

  Concatenate heads (output size `heads * out_features`); otherwise
  average them.

- negative_slope:

  Slope of the LeakyReLU in the attention score.

- dropout:

  Dropout probability on attention coefficients (training only).

- edge_nn:

  Module mapping edge features (E by `edge_dim`) to E by
  `in_features * out_features` (`nn_conv`). When `NULL`, a two layer MLP
  with 32 hidden units is built from `edge_dim`.

- edge_conv_nn:

  For `edge_conv`, the module \\h\_\Theta\\ applied to \\\[x_i \Vert
  x_j - x_i\]\\ (input size twice the node feature size).

- num_relations:

  Number of relation types; `edge_type` holds codes `1:num_relations`.

- num_bases:

  Use basis decomposition with this many bases (`NULL` for a full weight
  per relation).

## Value

An `nn_module`.

## Details

|  |  |  |  |
|----|----|----|----|
| Layer | Update for node \\i\\ | Edge input | Self loops |
| `sage_conv` | \\W_1 \mathrm{aggr}\_{j} w\_{ji} x_j + W_2 x_i\\ | `edge_weight` | none added |
| `gcn_conv` | \\\sum_j \frac{w\_{ji}}{\sqrt{d_j d_i}} W x_j + b\\ | `edge_weight` | added to nodes without one |
| `graph_conv` | \\W_1 \sum_j w\_{ji} x_j + W_2 x_i\\ | `edge_weight` | none added |
| `gin_conv` | \\\mathrm{MLP}((1+\epsilon) x_i + \sum_j x_j)\\ | `edge_attr` (GINE), `edge_weight` | none added |
| `gat_conv` | \\\sum_j \alpha\_{ij} W x_j\\, \\\alpha\\ from \\a^\top \[W x_i \Vert W x_j\]\\ | `edge_attr` | replaced: existing removed, one per node added |
| `gatv2_conv` | \\\sum_j \alpha\_{ij} W_s x_j\\, \\\alpha\\ from \\a^\top \mathrm{LeakyReLU}(W_s x_j + W_t x_i)\\ | `edge_attr` | as `gat_conv` |
| `transformer_conv` | \\W_r x_i + \sum_j \alpha\_{ij} (W_v x_j + W_e e\_{ji})\\ | `edge_attr` | none added |
| `nn_conv` | \\W x_i + \mathrm{aggr}\_j x_j \Theta(e\_{ji})\\ | `edge_attr` (required) | none added |
| `rgcn_conv` | \\W_0 x_i + \sum_r \frac{1}{\|N_r(i)\|}\sum\_{j \in N_r(i)} W_r x_j\\ | `edge_type` (required) | none added |
| `edge_conv` | \\\max_j h\_\Theta(\[x_i \Vert x_j - x_i\])\\ | none | none added |

Normalization, attention and cost per forward pass (N nodes, E edges,
input size F, output size F', H heads, R relations):

|  |  |  |  |
|----|----|----|----|
| Layer | Normalization or attention | Time | Extra memory |
| `sage_conv` | `aggr` (mean by default) over incoming edges; optional L2 row normalization | \\O(EF + NFF')\\ | \\E F\\ messages |
| `gcn_conv` | symmetric \\D^{-1/2}(A+I)D^{-1/2}\\, weighted in degree | \\O(EF' + NFF')\\ | \\E F'\\ messages |
| `graph_conv` | none (sum by default) | \\O(EF + NFF')\\ | \\E F\\ |
| `gin_conv` | none (sum) | \\O(EF)\\ plus the MLP | \\E F\\ |
| `gat_conv`, `gatv2_conv` | softmax of attention scores over the incoming edges of each node, per head | \\O(EHF' + NFHF')\\ | \\E H F'\\ messages and \\E H\\ coefficients |
| `transformer_conv` | scaled dot product softmax over incoming edges, per head | \\O(EHF' + NFHF')\\ | \\E H F'\\ and \\E H\\ |
| `nn_conv` | `aggr` (mean by default) | \\O(E F F')\\ | \\E F F'\\ edge specific weights, the largest |
| `rgcn_conv` | mean per relation | \\O(EF' + RNFF')\\ | \\R F F'\\ weights (or bases) |
| `edge_conv` | `aggr` (max by default) | \\O(E)\\ calls of \\h\_\Theta\\ on 2F inputs | E rows of \\h\_\Theta\\ output |

All layers are permutation equivariant: permuting node rows and
relabeling `edge_index` permutes the output rows the same way.

The formulas follow the PyTorch Geometric layers of the same names
(`SAGEConv`, `GCNConv`, `GraphConv`, `GINConv`/`GINEConv`, `GATConv`,
`GATv2Conv`, `TransformerConv`, `NNConv`, `RGCNConv`, `EdgeConv`) for
the options exposed here; options not listed are not implemented.

Attention layers keep the coefficients of their last forward pass in
`layer$last_attention` (a list with `edge_index` and an E by heads
tensor `alpha`). Attention coefficients describe how the model weighted
neighbors; they are not by themselves measures of feature importance or
causal relevance.

## References

Hamilton, Ying and Leskovec (2017) Inductive representation learning on
large graphs. Kipf and Welling (2017) Semi-supervised classification
with graph convolutional networks. Xu et al. (2019) How powerful are
graph neural networks? Velickovic et al. (2018) Graph attention
networks. Brody, Alon and Yahav (2022) How attentive are graph attention
networks? Shi et al. (2021) Masked label prediction. Gilmer et al.
(2017) Neural message passing for quantum chemistry. Schlichtkrull et
al. (2018) Modeling relational data with graph convolutional networks.
Morris et al. (2019) Weisfeiler and Leman go neural. Wang et al. (2019)
Dynamic graph CNN for learning on point clouds.

## Examples

``` r
if (torch::torch_is_installed()) {
  x <- torch::torch_randn(4, 3)
  ei <- torch::torch_tensor(rbind(c(1L, 2L, 3L), c(2L, 3L, 4L)),
                            dtype = torch::torch_long())
  layer <- sage_conv(3, 5)
  layer(x, ei)$shape
  gat_conv(3, 2, heads = 2)(x, ei)$shape
}
#> [1] 4 4
```
