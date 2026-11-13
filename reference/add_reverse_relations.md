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
