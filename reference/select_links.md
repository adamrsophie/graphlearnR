# Budgeted link selection with optcompR

Chooses which candidate links to add to a graph: at most `budget` pairs,
each node gaining at most `max_new_degree` new links, maximizing the sum
of predicted scores (for example link probabilities from a trained link
prediction model). The selection is a 0-1 integer program \$\$\max\_{z
\in \\0,1\\^P} \sum_p s_p z_p \quad\text{s.t.}\quad \sum_p z_p \le B,\\
\sum\_{p \ni v} z_p \le d_v,\$\$ built and solved with
[optcompR](https://github.com/adamrsophie/optcompR) (HiGHS by default).
Pairs with a score at or below `min_score` are never chosen.

## Usage

``` r
select_links(pairs, scores, budget, max_new_degree = Inf, min_score = -Inf)
```

## Arguments

- pairs:

  Integer matrix, 2 by P, of candidate node pairs (1-based).

- scores:

  Numeric score per pair.

- budget:

  Maximum number of pairs.

- max_new_degree:

  Maximum new links per node (a scalar or one value per node, indexed by
  node number).

- min_score:

  Pairs scoring at or below this are excluded.

## Value

A list with `selected` (column indices into `pairs`), `pairs` (the
selected pairs), `objective` and `status`.

## Details

This is an optional bridge used after training; integer programs are not
differentiable and are not part of any model.

## Examples

``` r
if (requireNamespace("optcompR", quietly = TRUE)) {
  pairs <- rbind(c(1, 1, 2, 3), c(2, 3, 3, 4))
  select_links(pairs, c(0.9, 0.8, 0.7, 0.6), budget = 3, max_new_degree = 1)
}
#> $selected
#> [1] 1 4
#> 
#> $pairs
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
#> 
#> $objective
#> [1] 1.5
#> 
#> $status
#> [1] "optimal"
#> 
```
