# Relation table of a heterogeneous graph

Relation table of a heterogeneous graph

## Usage

``` r
edge_types(hg)
```

## Arguments

- hg:

  A `hetero_graph_data` object.

## Value

A data frame with `key`, `src`, `rel`, `dst`, `num_edges` and `reverse`
(the paired reverse relation, if any).

## Examples

``` r
edge_types(dataset_synthetic_hetero(seed = 1))
#>                           key     src       rel     dst num_edges reverse
#> 1      drug__targets__protein    drug   targets protein       300    <NA>
#> 2 protein__interacts__protein protein interacts protein       234    <NA>
```
