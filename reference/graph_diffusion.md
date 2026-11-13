# Diffusion on a graph, solved with modelcompR

Solves the graph heat equation \$\$\frac{du}{dt} = -\kappa\\ L\\ u,\quad
u(0) = u_0,\$\$ with \\L = D - A\\ the Laplacian (`"laplacian"`) or
\\I - D^{-1/2} A D^{-1/2}\\ (`"normalized_laplacian"`), where
\\A\_{ij}\\ is the weight of the edge from \\j\\ to \\i\\ and \\D\\
holds the weighted in degrees. For an undirected graph the solution
conserves \\\sum_i u_i\\ under the plain Laplacian and tends to the
mean.

## Usage

``` r
graph_diffusion(
  g,
  u0,
  times,
  rate = 1,
  operator = c("laplacian", "normalized_laplacian"),
  alg = NULL,
  ...
)
```

## Arguments

- g:

  A `graph_data` object.

- u0:

  Initial value per node (numeric vector of length `num_nodes`).

- times:

  Times at which to return the solution; the first is the start.

- rate:

  Diffusion rate \\\kappa\\.

- operator:

  `"laplacian"` or `"normalized_laplacian"`.

- alg:

  A modelcompR algorithm; default
  [`modelcompR::Tsit5()`](https://rdrr.io/pkg/modelcompR/man/ode_algorithms.html).

- ...:

  Passed to modelcompR's [`solve()`](https://rdrr.io/r/base/solve.html)
  (for example `abstol`).

## Value

A matrix with one row per time and one column per node, with the times
in attribute `"times"`.

## Details

The ODE is solved by
[modelcompR](https://github.com/adamrsophie/modelcompR) (`ODEProblem()`
and [`solve()`](https://rdrr.io/r/base/solve.html) with `Tsit5()` by
default); graphlearnR has no integrator of its own. This is an optional
bridge: modelcompR must be installed, and the solve runs on R numerics,
so no gradients flow back to torch. It is a mechanistic simulation tool,
not a neural ODE layer.

## Examples

``` r
if (requireNamespace("modelcompR", quietly = TRUE)) {
  g <- graph_data(edge_index = rbind(1:3, 2:4), num_nodes = 4, directed = FALSE)
  graph_diffusion(g, c(1, 0, 0, 0), times = c(0, 1, 5))
}
#>              1         2         3         4
#> [1,] 1.0000000 0.0000000 0.0000000 0.0000000
#> [2,] 0.5238153 0.3087569 0.1235759 0.0438519
#> [3,] 0.2728255 0.2594370 0.2405384 0.2271991
#> attr(,"times")
#> [1] 0 1 5
```
