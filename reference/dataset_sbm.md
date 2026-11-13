# Stochastic block model node classification data

Nodes are split into blocks (classes); edges inside a block appear with
probability `p_in` and between blocks with `p_out`. Node features are a
class specific mean plus Gaussian noise, so both structure and features
carry signal; set `feature_signal = 0` for a structure only task.

## Usage

``` r
dataset_sbm(
  block_sizes = c(50, 50, 50),
  p_in = 0.1,
  p_out = 0.01,
  num_features = 8L,
  feature_signal = 1,
  seed = NULL
)
```

## Arguments

- block_sizes:

  Nodes per block.

- p_in, p_out:

  Within and between block edge probabilities.

- num_features:

  Feature columns.

- feature_signal:

  Distance between class means relative to the noise.

- seed:

  Optional integer seed.

## Value

An undirected `graph_data` with factor labels.

## Examples

``` r
g <- dataset_sbm(c(30, 30, 30), seed = 1)
g
#> <graph_data> undirected graph: 90 nodes, 316 edges (158 undirected pairs)
#>   node features [90 x 8]
#>   node labels: factor with 3 levels
```
