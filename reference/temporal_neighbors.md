# Most recent temporal neighbors

For each query `(node, time)`, returns up to `k` most recent events that
involve `node` and happened **strictly before** `time` (or at or before
`time` with `strict = FALSE`). This is the neighborhood a continuous
time model such as TGN or TGAT may use at prediction time; no event at
or after the query time is ever returned with the default
`strict = TRUE`.

## Usage

``` r
temporal_neighbors(
  tg,
  nodes,
  times,
  k = 10L,
  strict = TRUE,
  direction = c("in", "both")
)
```

## Arguments

- tg:

  A `temporal_graph_data` object.

- nodes:

  Query nodes.

- times:

  Query times (recycled).

- k:

  Neighbors per query.

- strict:

  Exclude events at exactly the query time.

- direction:

  `"in"` or `"both"`.

## Value

A data frame with `query`, `node`, `neighbor`, `t` and `event` (position
in the sorted event list).

## Details

For directed graphs only events with `dst == node` (incoming messages)
are used unless `direction = "both"`.

## Examples

``` r
tg <- dataset_synthetic_temporal(seed = 1)
nb <- temporal_neighbors(tg, nodes = tg$dst[50], times = tg$t[50], k = 5)
nb
#>   query node neighbor         t event
#> 1     1   12       29 2.5367459    26
#> 2     1   12       29 1.8193165    20
#> 3     1   12       11 0.7279515     9
```
