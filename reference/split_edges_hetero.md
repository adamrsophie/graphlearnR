# Split one relation of a heterogeneous graph for link prediction

Holds out edges of `relation` for validation and test. Held out edges
are removed from the training graph **and from the paired relation** (as
recorded by
[`add_reverse_relations()`](https://adamrsophie.github.io/graphlearnR/reference/add_reverse_relations.md),
whichever of the two keys is split), so a model cannot read a held out
link through its reverse edge. Parallel copies of a pair are split as
one unit. For a relation whose source and target type are the same,
`(u, v)` and `(v, u)` are one unit and negatives are unordered pairs
without self loops; otherwise negatives are typed pairs (source type,
target type). Negatives are drawn with
[`negative_sampling()`](https://adamrsophie.github.io/graphlearnR/reference/negative_sampling.md),
excluding every known positive of the relation.

## Usage

``` r
split_edges_hetero(
  hg,
  relation,
  val = 0.1,
  test = 0.1,
  neg_ratio = 1,
  seed = NULL
)
```

## Arguments

- hg:

  A `hetero_graph_data` object.

- relation:

  Relation key, `"src__rel__dst"`.

- val, test:

  Proportions (both below 1) or whole counts of pairs held out.

- neg_ratio:

  Negatives per positive for validation and test.

- seed:

  Optional integer seed.

## Value

An object of class `hetero_link_split`.

## Examples

``` r
hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 1))
sp <- split_edges_hetero(hg, "drug__targets__protein", seed = 1)
ncol(sp$test_pos)
#> [1] 30
```
