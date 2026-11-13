# Split one relation of a heterogeneous graph for link prediction

Holds out edges of `relation` for validation and test. Held out edges
are removed from the training graph **and from the paired reverse
relation** (as recorded by
[`add_reverse_relations()`](https://adamrsophie.github.io/graphlearnR/reference/add_reverse_relations.md)),
so a model cannot read a held out link through its reverse edge.
Negatives are typed pairs (source type, target type) drawn with
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

  Proportions (or counts) of pairs held out.

- neg_ratio:

  Negatives per positive for validation and test.

- seed:

  Optional integer seed.

## Value

An object of class `hetero_link_split`.
