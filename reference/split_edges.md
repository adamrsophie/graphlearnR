# Split edges for link prediction

Holds out edges for validation and test without leaking them into
training:

## Usage

``` r
split_edges(g, val = 0.1, test = 0.1, neg_ratio = 1, seed = NULL)
```

## Arguments

- g:

  A `graph_data` object.

- val, test:

  Proportions (or counts) of pairs held out.

- neg_ratio:

  Negatives per positive for validation and test.

- seed:

  Optional integer seed.

## Value

An object of class `link_split` with `train_graph`, `train_pos`,
`val_pos`, `val_neg`, `test_pos`, `test_neg` (2 by K integer matrices),
`all_pos` and settings.

## Details

1.  Undirected pairs are split as units, so `(u, v)` and `(v, u)` always
    land in the same set.

2.  `train_graph` keeps all nodes and features but only training edges;
    use it for message passing during training **and** evaluation.

3.  Validation and test negatives are drawn once with
    [`negative_sampling()`](https://adamrsophie.github.io/graphlearnR/reference/negative_sampling.md),
    excluding every positive edge of the full graph. Training negatives
    should be redrawn each epoch, excluding training positives only,
    which
    [`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
    does.

Positive pairs are returned in one direction (as given in the input).

## Examples

``` r
sp <- split_edges(dataset_karate(), val = 0.1, test = 0.1, seed = 1)
sp
#> <link_split> 34 nodes, undirected pairs: train 62, val 8 (+8 neg), test 8 (+8 neg)
```
