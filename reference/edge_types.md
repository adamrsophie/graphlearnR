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
