# Temporal train, validation and test split

Splits events by time: training events have `t < val_time`, validation
events `val_time <= t < test_time`, test events `t >= test_time`. Models
evaluated at time `t` may use any event strictly before `t`, including
earlier validation events, but never later ones. With proportions
instead of times, the cut points are event time quantiles.

## Usage

``` r
split_temporal(tg, val_time = NULL, test_time = NULL, val = 0.15, test = 0.15)
```

## Arguments

- tg:

  A `temporal_graph_data` object.

- val_time, test_time:

  Cut points.

- val, test:

  Proportions used when the cut points are `NULL`.

## Value

A list with `train`, `val`, `test` (each a `temporal_graph_data`) and
the cut points.

## Examples

``` r
tg <- dataset_synthetic_temporal(seed = 1)
sp <- split_temporal(tg)
lengths(sp[c("train", "val", "test")])
#> train   val  test 
#>    10    10    10 
```
