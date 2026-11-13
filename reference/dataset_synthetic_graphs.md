# Synthetic graph classification and regression data

Small graphs whose label depends only on structure, so a model must use
message passing to solve the task. Class `"cycle"` graphs are rings;
class `"wheel"` graphs are rings plus a hub joined to every ring node.
Sizes vary from `min_nodes` to `max_nodes`. Node features are a constant
column, so features alone carry no information. For
`task = "regression"`, `graph_y` is the average degree.

## Usage

``` r
dataset_synthetic_graphs(
  n = 100L,
  min_nodes = 5L,
  max_nodes = 12L,
  task = c("classification", "regression"),
  seed = NULL
)
```

## Arguments

- n:

  Number of graphs.

- min_nodes, max_nodes:

  Ring size range.

- task:

  `"classification"` or `"regression"`.

- seed:

  Optional integer seed.

## Value

A list of undirected `graph_data` objects with `graph_y`.

## Examples

``` r
gs <- dataset_synthetic_graphs(6, seed = 1)
vapply(gs, function(g) as.character(g$graph_y), "")
#> [1] "cycle" "cycle" "wheel" "wheel" "cycle" "cycle"
```
