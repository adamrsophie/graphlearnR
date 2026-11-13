# Select events in a time window

Select events in a time window

## Usage

``` r
temporal_window(tg, start = NULL, end = NULL)
```

## Arguments

- tg:

  A `temporal_graph_data` object.

- start, end:

  Window `[start, end)`; `NULL` means unbounded.

## Value

A `temporal_graph_data` with the selected events.

## Examples

``` r
tg <- dataset_synthetic_temporal(seed = 1)
temporal_window(tg, start = 10, end = 20)
#> <temporal_graph_data> 40 nodes, 70 events, t in [10.18985, 19.73235]
```
