# Rule constrained node labels with z3r

Finds the most probable joint labeling of the nodes that satisfies
logical rules on edges, using the Z3 optimizer through
[z3r](https://github.com/adamrsophie/z3r). Each node \\i\\ gets one
label \\c_i\\; the objective is \\\max \sum_i \log p_i(c_i)\\ (log
probabilities rounded to `precision`) subject to: no edge \\(j, i)\\ has
\\(c_j, c_i)\\ in `forbidden`, and nodes in `fixed` keep their label.
This turns independent per node predictions into a consistent
assignment, for example when two adjacent nodes cannot both belong to a
class.

## Usage

``` r
consistent_labels(
  probs,
  edge_index,
  forbidden,
  fixed = NULL,
  precision = 1000,
  timeout = 10000
)
```

## Arguments

- probs:

  Matrix of class probabilities, one row per node and one named column
  per class (for example from the `.prob_` columns of
  [`predict()`](https://rdrr.io/r/stats/predict.html)).

- edge_index:

  Integer matrix, 2 by E, 1-based.

- forbidden:

  Two column matrix or data frame of class names: an edge from a node of
  the first class to a node of the second is not allowed.

- fixed:

  Optional named vector of known labels; names are node indices.

- precision:

  Log probabilities are multiplied by this and rounded.

- timeout:

  Solver timeout in milliseconds.

## Value

A list with `labels` (factor), `status` (`"sat"`, `"unsat"` or
`"unknown"`), `changed` (nodes whose label differs from the most
probable class) and `log_prob` (total log probability of the result).

## Details

This is an optional bridge for small graphs: the problem is a weighted
constraint satisfaction problem with one integer variable per node, and
solving time grows quickly with the number of nodes and rules. It runs
after training and is not differentiable.

## Examples

``` r
if (requireNamespace("z3r", quietly = TRUE)) {
  p <- rbind(c(a = 0.9, b = 0.1), c(a = 0.6, b = 0.4), c(a = 0.2, b = 0.8))
  ei <- rbind(c(1, 2, 2, 3), c(2, 1, 3, 2))
  consistent_labels(p, ei, forbidden = cbind("a", "a"))
}
```
