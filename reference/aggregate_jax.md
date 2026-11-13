# Neighborhood aggregation with jaxR

Computes the message passing aggregation \\h_i = \mathrm{aggr}\_{j \to
i} w\_{ji} x_j\\ with [jaxR](https://github.com/adamrsophie/jaxR) array
operations (`jnp_take` and `ops_segment_sum`), the same operation that
[`message_passing()`](https://adamrsophie.github.io/graphlearnR/reference/message_passing.md)
performs with torch. Because it is written in jaxR, the result can be
differentiated with `jaxR::grad()`, vectorized with `jaxR::vmap()` and
compiled with `jaxR::jit()` (pass `num_nodes` under `jit()`).

## Usage

``` r
aggregate_jax(
  x,
  edge_index,
  num_nodes = NULL,
  aggr = c("sum", "mean"),
  edge_weight = NULL
)
```

## Arguments

- x:

  Node features: an R matrix or a jaxR array, N by F.

- edge_index:

  Integer matrix, 2 by E, 1-based (source, target).

- num_nodes:

  Number of target nodes; default `nrow(x)`.

- aggr:

  `"sum"` or `"mean"`.

- edge_weight:

  Optional weight per edge.

## Value

A jaxR array, `num_nodes` by F;
[`as.matrix()`](https://rdrr.io/r/base/matrix.html) converts it.

## Details

This is an optional bridge for experiments with a second array backend;
graphlearnR's layers and training remain on torch, and models are not
converted between backends. It needs jaxR and its XLA runtime
(`jaxR::install_pjrt()`).

## Examples

``` r
if (FALSE) { # \dontrun{
x <- matrix(1:6, 3, 2)
ei <- rbind(c(1, 2, 3), c(2, 3, 2))
as.matrix(aggregate_jax(x, ei, aggr = "mean"))
} # }
```
