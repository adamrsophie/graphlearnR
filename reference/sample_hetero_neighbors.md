# Typed neighbor sampling

Heterogeneous version of
[`sample_neighbors()`](https://adamrsophie.github.io/graphlearnR/reference/sample_neighbors.md).
Starting from seed nodes of one or more types, hop `k` samples up to
`fanouts[k]` incoming edges **per relation** for every frontier node of
that relation's target type. `fanouts` may also be a named list giving a
vector per relation key.

## Usage

``` r
sample_hetero_neighbors(hg, seeds, fanouts, seed = NULL)
```

## Arguments

- hg:

  A `hetero_graph_data` object.

- seeds:

  Named list of seed indices per node type.

- fanouts:

  Integer vector (all relations) or named list per relation.

- seed:

  Optional integer seed.

## Value

A `hetero_graph_data` of class `sampled_hetero_graph`.

## Details

The result is a `hetero_graph_data` with local indices; `n_id[[type]]`
holds the original indices (seeds first) and `num_seeds[[type]]` the
seed counts.
