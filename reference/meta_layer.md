# Edge, node and graph update block

One block of the graph network of Battaglia et al. (2018), in the form
of PyTorch Geometric's `MetaLayer`: an edge update, then a node update,
then a graph level update, each optional. With edges \\(j \to i)\\, node
features \\x\\, edge features \\e\\ and graph features \\u\\:

## Usage

``` r
meta_layer(edge_model = NULL, node_model = NULL, global_model = NULL)
```

## Arguments

- edge_model, node_model, global_model:

  Update functions or modules, or `NULL` to keep the corresponding
  features unchanged.

## Value

An `nn_module` called as
`block(x, edge_index, edge_attr = NULL, u = NULL, batch = NULL)`,
returning a list with `x`, `edge_attr` and `u`.

## Details

1.  edge update \\e'\_{ji} = \phi^e(x_j, x_i, e\_{ji}, u\_{g(i)})\\;

2.  node update \\x'\_i = \phi^v(x, \mathrm{edge\\index}, e', u,
    \mathrm{batch})\\ (typically an aggregation of the updated incoming
    edges, see
    [`message_passing()`](https://adamrsophie.github.io/graphlearnR/reference/message_passing.md)
    and
    [`scatter_aggregate()`](https://adamrsophie.github.io/graphlearnR/reference/scatter.md));

3.  graph update \\u'\_g = \phi^u(x', \mathrm{edge\\index}, e', u,
    \mathrm{batch})\\ (typically a pooled summary, see
    [global-pooling](https://adamrsophie.github.io/graphlearnR/reference/global-pooling.md)).

Each update is a function or `nn_module` called with named arguments:

- `edge_model(x_src, x_dst, edge_attr, u, batch)`: `x_src` and `x_dst`
  are the source and target node features gathered per edge, `u` the
  graph features gathered per edge (`NULL` without `u`), `batch` the
  graph of each edge. Returns E rows.

- `node_model(x, edge_index, edge_attr, u, batch)`: returns N rows.

- `global_model(x, edge_index, edge_attr, u, batch)`: returns one row
  per graph.

The block is permutation equivariant in nodes and edges (and invariant
for the graph output) whenever the update functions work row wise and
aggregate with permutation invariant operators.

## References

Battaglia et al. (2018) Relational inductive biases, deep learning, and
graph networks. arXiv:1806.01261.

## Examples

``` r
if (torch::torch_is_installed()) {
  edge_mlp <- torch::nn_linear(2 * 3 + 1, 4)
  node_mlp <- torch::nn_linear(3 + 4, 3)
  block <- meta_layer(
    edge_model = function(x_src, x_dst, edge_attr, ...) {
      edge_mlp(torch::torch_cat(list(x_src, x_dst, edge_attr), dim = 2))
    },
    node_model = function(x, edge_index, edge_attr, ...) {
      agg <- scatter_mean(edge_attr, edge_index[2, ], x$size(1))
      node_mlp(torch::torch_cat(list(x, agg), dim = 2))
    },
    global_model = function(x, batch, ...) global_mean_pool(x, batch)
  )
  g <- dataset_karate()
  tt <- graph_tensors(g)
  out <- block(torch::torch_randn(34, 3), tt$edge_index,
               edge_attr = torch::torch_randn(num_edges(g), 1))
  lapply(out, function(t) t$shape)
}
#> $x
#> [1] 34  3
#> 
#> $edge_attr
#> [1] 156   4
#> 
#> $u
#> [1] 1 3
#> 
```
