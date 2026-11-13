# Snapshot sequences

`snapshot_sequence()` wraps an ordered list of graphs on the same nodes.
`temporal_snapshots()` builds one from events: snapshot `k` contains the
events with `breaks[k] <= t < breaks[k + 1]` (the last interval is
closed on the right), or with `cumulative = TRUE` all events with
`t < breaks[k + 1]`. Repeated interactions become edge weights (counts).

## Usage

``` r
snapshot_sequence(
  graphs,
  start = seq_along(graphs) - 1,
  end = seq_along(graphs)
)

temporal_snapshots(tg, breaks = 10L, cumulative = FALSE)
```

## Arguments

- graphs:

  List of `graph_data` objects with equal `num_nodes`.

- start, end:

  Interval bounds per snapshot.

- tg:

  A `temporal_graph_data` object.

- breaks:

  Number of equal width intervals, or a sorted vector of cut points.

- cumulative:

  Accumulate all earlier events into each snapshot.

## Value

An object of class `snapshot_sequence`.
