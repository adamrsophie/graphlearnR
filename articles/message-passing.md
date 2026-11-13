# Message passing and graph neural network layers

``` r

library(graphlearnR)
library(torch)
r_vec <- function(t) as.numeric(as.array(t))
```

A graph neural network layer updates each node from messages sent along
its incoming edges. For node i with incoming neighbors \mathcal{N}(i),

h_i' = \gamma\left(h_i,\\ \bigoplus\_{j \in \mathcal{N}(i)} \phi(h_i,
h_j, e\_{ji})\right),

where \phi builds a message, \bigoplus is a permutation invariant
aggregation such as sum, mean, or max, and \gamma updates the node. This
article shows the pieces graphlearnR uses to compute this and how the
built in layers fit the pattern.

## Scatter aggregation

Aggregation is a *scatter*: row k of a message matrix is added (or
averaged, or maximized) into row `index[k]` of the output. Rows that
receive nothing are zero.

``` r

msg <- torch_tensor(matrix(c(1, 2, 3, 4, 5, 6), ncol = 1))
index <- c(1L, 1L, 2L, 2L, 2L, 4L)
cbind(sum = r_vec(scatter_sum(msg, index, 4)),
      mean = r_vec(scatter_mean(msg, index, 4)),
      max = r_vec(scatter_max(msg, index, 4)),
      count = r_vec(scatter_count(index, 4)))
#>      sum mean max count
#> [1,]   3  1.5   2     2
#> [2,]  12  4.0   5     3
#> [3,]   0  0.0   0     0
#> [4,]   6  6.0   6     1
```

Node 3 receives no messages, so every aggregation returns 0 for it
rather than a missing value or negative infinity.
[`scatter_softmax()`](https://adamrsophie.github.io/graphlearnR/reference/scatter.md)
normalizes scores within each group, which attention layers use.

``` r

r_vec(scatter_softmax(msg, index, 4))
#> [1] 0.26894143 0.73105860 0.09003057 0.24472848 0.66524094 1.00000000
```

All of these are vectorized torch operations, so they run on the GPU and
are differentiable.

## The general operator

[`message_passing()`](https://adamrsophie.github.io/graphlearnR/reference/message_passing.md)
implements the equation above. By default the message is the source
features, optionally scaled by `edge_weight`.

``` r

x <- torch_tensor(matrix(c(1, 10, 100), ncol = 1))
ei <- torch_tensor(rbind(c(1, 2, 3), c(3, 3, 1)), dtype = torch_long())
r_vec(message_passing(x, ei, aggr = "sum"))
#> [1] 100   0  11
r_vec(message_passing(x, ei, aggr = "mean"))
#> [1] 100.0   0.0   5.5
```

Node 3 receives from nodes 1 and 2, node 1 from node 3, and node 2 from
no one.

A custom message function is called with named arguments `x_j` (source
features), `x_i` (target features), `edge_attr`, `edge_weight`, `index`
(the target of each edge), and `num_nodes`, one row per edge. Accept the
ones you need and `...` for the rest. Here is a message that is the
difference between source and target, the core of EdgeConv:

``` r

diff_msg <- function(x_j, x_i, ...) x_j - x_i
r_vec(message_passing(x, ei, message = diff_msg, aggr = "max"))
#> [1]  99   0 -90
```

## Writing a layer

A layer is a torch `nn_module` whose `forward()` calls
[`message_passing()`](https://adamrsophie.github.io/graphlearnR/reference/message_passing.md).
This one implements a mean aggregator with separate weights for the node
and its neighbors:

``` r

my_conv <- nn_module(
  initialize = function(in_features, out_features) {
    self$lin_self <- nn_linear(in_features, out_features)
    self$lin_nbr <- nn_linear(in_features, out_features, bias = FALSE)
  },
  forward = function(x, edge_index) {
    agg <- message_passing(x, edge_index, aggr = "mean")
    self$lin_self(x) + self$lin_nbr(agg)
  }
)

g <- dataset_karate()
tt <- graph_tensors(g)
torch_manual_seed(1)
layer <- my_conv(34, 4)
dim(layer(tt$x, tt$edge_index))
#> [1] 34  4
```

That is the GraphSAGE update, which graphlearnR also provides as
[`sage_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md).

## Built in layers

| layer | update | edge input |
|----|----|----|
| [`sage_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md) | W_1 h_i + W_2 \operatorname{mean}\_{j} h_j | none |
| [`gcn_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md) | \sum_j \frac{1}{\sqrt{\hat d_i \hat d_j}} W h_j with self loops | weights |
| [`graph_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md) | W_1 h_i + W_2 \sum_j w\_{ji} h_j | weights |
| [`gin_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md) | \operatorname{MLP}((1 + \epsilon) h_i + \sum_j h_j) | optional features (GINE) |
| [`gat_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md) | attention weighted sum, multiple heads | optional features |
| [`gatv2_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md) | GATv2 dynamic attention | optional features |
| [`transformer_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md) | scaled dot product attention over neighbors | optional features |
| [`nn_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md) | W_1 h_i + \sum_j \Theta(e\_{ji}) h_j (edge conditioned) | features, required |
| [`rgcn_conv()`](https://adamrsophie.github.io/graphlearnR/reference/gnn-layers.md) | W_0 h_i + \sum_r \operatorname{mean}\_{j \in \mathcal{N}\_r(i)} W_r h_j | relation type, required |

Every layer takes `(x, edge_index, ...)` and returns new node features.

``` r

torch_manual_seed(2)
layers <- list(
  sage = sage_conv(34, 8),
  gcn = gcn_conv(34, 8),
  gin = gin_conv(graph_mlp(c(34, 16, 8))),
  gat = gat_conv(34, 4, heads = 2),
  transformer = transformer_conv(34, 4, heads = 2)
)
sapply(layers, function(l) dim(l(tt$x, tt$edge_index)))
#>      sage gcn gin gat transformer
#> [1,]   34  34  34  34          34
#> [2,]    8   8   8   8           8
```

Attention layers keep the coefficients of their last forward pass.
Within each target node they sum to one.

``` r

gat <- layers$gat
invisible(gat(tt$x, tt$edge_index))
alpha <- as.array(gat$last_attention$alpha)
dst <- as.array(gat$last_attention$edge_index)[2, ]
head(rowsum(alpha, dst))
#>   [,1] [,2]
#> 1    1    1
#> 2    1    1
#> 3    1    1
#> 4    1    1
#> 5    1    1
#> 6    1    1
```

### Bipartite input

Most layers also accept a pair `list(x_src, x_dst)` when sources and
targets are different node sets with different feature sizes. Give the
input size as a pair too. Heterogeneous models are built on this.

``` r

xs <- torch_randn(5, 3)
xd <- torch_randn(2, 7)
bip <- torch_tensor(rbind(c(1, 2, 5), c(1, 2, 2)), dtype = torch_long())
dim(sage_conv(c(3, 7), 4)(list(xs, xd), bip))
#> [1] 2 4
```

### Edge features and relations

``` r

torch_manual_seed(3)
ge <- dataset_karate()
ge$edge_attr <- matrix(runif(num_edges(ge) * 2), ncol = 2)
ge$edge_type <- sample(1:3, num_edges(ge), replace = TRUE)
te <- graph_tensors(ge)
dim(nn_conv(34, 8, edge_dim = 2)(te$x, te$edge_index, edge_attr = te$edge_attr))
#> [1] 34  8
dim(rgcn_conv(34, 8, num_relations = 3, num_bases = 2)(te$x, te$edge_index,
                                                     edge_type = te$edge_type))
#> [1] 34  8
```

## How the layers are tested

The test suite checks every layer for properties that must hold whatever
the weights are:

- **Permutation equivariance.** Relabeling the nodes relabels the output
  in the same way.
- **Agreement with a reference formula.** GCN is compared with the dense
  \hat D^{-1/2} \hat A \hat D^{-1/2} X W, GraphSAGE and GIN with
  explicit neighbor loops, and RGCN with per relation means.
- **Gradients.** Input gradients match central finite differences in
  double precision, and parameter gradients of GCN match the analytic
  formula.

Here is the equivariance check for GraphSAGE:

``` r

torch_manual_seed(4)
layer <- sage_conv(34, 4)
perm <- sample.int(34)
inv <- order(perm)
gp <- g
gp$x <- g$x[perm, ]
gp$edge_index <- matrix(inv[g$edge_index], nrow = 2)
out <- as.array(layer(tt$x, tt$edge_index))
out_p <- as.array(layer(graph_tensors(gp)$x, graph_tensors(gp)$edge_index))
max(abs(out_p - out[perm, ]))
#> [1] 0
```

## Pooling

Graph level models reduce node features to one vector per graph.
[`global_add_pool()`](https://adamrsophie.github.io/graphlearnR/reference/global-pooling.md),
[`global_mean_pool()`](https://adamrsophie.github.io/graphlearnR/reference/global-pooling.md),
and
[`global_max_pool()`](https://adamrsophie.github.io/graphlearnR/reference/global-pooling.md)
take the `batch` vector;
[`global_attention_pool()`](https://adamrsophie.github.io/graphlearnR/reference/global_attention_pool.md)
learns a gate;
[`topk_pool()`](https://adamrsophie.github.io/graphlearnR/reference/topk_pool.md)
keeps the highest scoring fraction of nodes in each graph.

``` r

b <- graph_batch(dataset_synthetic_graphs(4, seed = 1))
tb <- graph_tensors(b)
as.array(global_mean_pool(tb$x, b$batch))
#>      [,1]
#> [1,]    1
#> [2,]    1
#> [3,]    1
#> [4,]    1
```
