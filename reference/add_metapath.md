# Add a metapath relation

Composes a chain of relations into one new relation, as PyG's
`AddMetaPaths`: an edge \\u \to w\\ is added when some walk follows the
relations of `path` in order from \\u\\ to \\w\\. Each pair is kept once
(walk counts are not recorded). Metapaths such as paper, author, paper
are the input of HAN style models (`hetero_gnn_model(conv = "han")`).
Build metapaths from training edges only when the composed relations
will be evaluated.

## Usage

``` r
add_metapath(hg, path, name = NULL)
```

## Arguments

- hg:

  A `hetero_graph_data` object.

- path:

  Relation keys (`"src__rel__dst"`), where the target type of each is
  the source type of the next.

- name:

  Relation name of the new key; defaults to the relation names joined by
  `"_"`.

## Value

`hg` with the new relation.

## Examples

``` r
hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 1))
names(hg$edges)
#> [1] "drug__targets__protein"          "protein__interacts__protein"    
#> [3] "protein__rev_targets__drug"      "protein__rev_interacts__protein"
hg2 <- add_metapath(hg, c("drug__targets__protein", "protein__rev_targets__drug"))
tail(names(hg2$edges), 1)
#> [1] "drug__targets_rev_targets__drug"
```
