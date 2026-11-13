# Add reverse relations

For every relation `src__rel__dst` without a reverse, adds
`dst__<prefix>rel__src` with flipped edges and the same edge attributes,
and records the pairing in `reverse` and `reverse_of`.

## Usage

``` r
add_reverse_relations(hg, prefix = "rev_")
```

## Arguments

- hg:

  A `hetero_graph_data` object.

- prefix:

  Prefix for the reverse relation name.

## Value

The extended `hetero_graph_data`.

## Examples

``` r
hg <- dataset_synthetic_hetero(seed = 1)
edge_types(add_reverse_relations(hg))
#>                               key     src           rel     dst num_edges
#> 1          drug__targets__protein    drug       targets protein       300
#> 2     protein__interacts__protein protein     interacts protein       234
#> 3      protein__rev_targets__drug protein   rev_targets    drug       300
#> 4 protein__rev_interacts__protein protein rev_interacts protein       234
#>                           reverse
#> 1      protein__rev_targets__drug
#> 2 protein__rev_interacts__protein
#> 3          drug__targets__protein
#> 4     protein__interacts__protein
```
