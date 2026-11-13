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
