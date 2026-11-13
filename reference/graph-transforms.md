# Graph transformations

Functions that return a modified copy of a `graph_data` object.

## Usage

``` r
add_self_loops(g, fill_weight = 1)

remove_self_loops(g)

coalesce_edges(g)

induced_subgraph(g, nodes)

as_undirected(g)

normalize_features(g, method = c("standardize", "row"), stats_mask = NULL)

add_degree_features(g, one_hot = FALSE, max_degree = 10L, log = TRUE)

compose_transforms(...)
```

## Arguments

- g:

  A `graph_data` object.

- fill_weight:

  Weight of new self loops.

- nodes:

  Integer node indices to keep.

- method:

  `"standardize"` (zero mean, unit variance per column) or `"row"` (rows
  sum to one, as in Kipf and Welling 2017).

- stats_mask:

  Logical node mask used to compute column statistics.

- one_hot:

  Use one-hot degree columns capped at `max_degree`.

- max_degree:

  Cap for one-hot encoding.

- log:

  Use `log1p(degree)` for the numeric column.

- ...:

  Transform functions, applied left to right.

## Value

A `graph_data` object.

## Details

- `add_self_loops()`: add a loop `(i, i)` to every node that has none.
  New loops get weight `fill_weight` and zero edge features.

- `remove_self_loops()`: drop all loops.

- `coalesce_edges()`: merge duplicate stored edges, summing weights and
  averaging edge features; keeps the first label.

- `induced_subgraph()`: keep `nodes` (in the given order) and the edges
  among them.

- `as_undirected()`: turn a directed graph into an undirected one,
  keeping one entry per unordered pair (attributes of the first
  occurrence).

- `normalize_features()`: standardize or row normalize node features.
  With `stats_mask`, means and scales come from those nodes only (for
  example the training nodes), so no information from held out nodes
  enters preprocessing.

- `add_degree_features()`: append in degree (optionally log scaled) or
  one-hot degree columns to `x`.

- `compose_transforms()`: chain transforms for
  [`graph_dataset()`](https://adamrsophie.github.io/graphlearnR/reference/graph_dataset.md).

## Examples

``` r
g <- graph_data(x = diag(3), edge_index = rbind(c(1, 1, 2), c(2, 2, 3)))
has_duplicate_edges(g)
#> [1] TRUE
g2 <- coalesce_edges(g)
num_edges(g2)
#> [1] 2
num_edges(add_self_loops(g2))
#> [1] 5
```
