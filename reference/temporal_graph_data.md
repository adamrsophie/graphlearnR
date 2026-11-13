# Temporal graph representations

graphlearnR keeps three kinds of time varying graphs apart, because
their semantics differ:

## Usage

``` r
temporal_graph_data(
  src,
  dst,
  t,
  num_nodes = NULL,
  x = NULL,
  edge_attr = NULL,
  y = NULL,
  node_ids = NULL,
  directed = TRUE,
  metadata = list()
)

validate_temporal_graph(tg)
```

## Arguments

- src, dst:

  Integer vectors of one based node indices.

- t:

  Numeric (or `POSIXct`, `Date`) event times.

- num_nodes:

  Number of nodes; inferred when `NULL`.

- x:

  Optional static node features.

- edge_attr:

  Optional event features (one row per event).

- y:

  Optional event labels.

- node_ids:

  External node identifiers.

- directed:

  Whether interactions are directed.

- metadata:

  Named list.

- tg:

  A `temporal_graph_data` object.

## Value

An object of class `temporal_graph_data`.

## Details

1.  **Continuous time event graphs** (`temporal_graph_data()`): a stream
    of timestamped interactions `(src, dst, t)`, possibly with event
    features and labels. Events are stored sorted by time (stable, so
    ties keep input order); `event_id` gives each event's position in
    the input.

2.  **Snapshot sequences**
    ([`snapshot_sequence()`](https://adamrsophie.github.io/graphlearnR/reference/snapshot_sequence.md),
    [`temporal_snapshots()`](https://adamrsophie.github.io/graphlearnR/reference/snapshot_sequence.md)):
    an ordered list of static graphs over the same node set, each with
    an interval `[start, end)`.

3.  **Spatiotemporal signals**
    ([`spatiotemporal_signal()`](https://adamrsophie.github.io/graphlearnR/reference/spatiotemporal_signal.md)):
    a fixed graph with node features that change over time steps.

Conversions are explicit:
[`temporal_snapshots()`](https://adamrsophie.github.io/graphlearnR/reference/snapshot_sequence.md)
turns events into snapshots with a stated interval rule; nothing
converts between the kinds implicitly.

## Examples

``` r
tg <- temporal_graph_data(src = c(1, 2, 3, 1), dst = c(2, 3, 1, 3),
                          t = c(1, 2, 2, 5))
tg
#> <temporal_graph_data> 3 nodes, 4 events, t in [1, 5]
snaps <- temporal_snapshots(tg, breaks = c(0, 2, 4, 6))
length(snaps$graphs)
#> [1] 3
```
