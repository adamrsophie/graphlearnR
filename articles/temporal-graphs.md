# Temporal and dynamic graphs

``` r

library(graphlearnR)
```

Graphs change. Messages are sent, payments are made, sensors report
every few minutes. graphlearnR supports three views of time, and each
comes with a rule that keeps the future out of the past.

| view | object | leakage rule |
|----|----|----|
| events in continuous time | `temporal_graph_data` | a query at time t uses events with time \< t |
| discrete snapshots | `snapshot_sequence` | snapshot k covers \[b_k, b\_{k+1}); evaluation steps come after training steps |
| signals on a fixed graph | `spatiotemporal_signal` | train, validation, and test windows are consecutive in time |

## Event graphs

A temporal graph is a stream of timestamped edges, sorted by time on
creation. `event_id` remembers the input order.

``` r

tg <- temporal_graph_data(src = c(3, 1, 2, 1), dst = c(1, 2, 3, 3), t = c(5, 1, 3, 7),
                          edge_attr = matrix(c(0.5, 0.1, 0.3, 0.9), ncol = 1))
tg
#> <temporal_graph_data> 3 nodes, 4 events, t in [1, 7]
data.frame(src = tg$src, dst = tg$dst, t = tg$t, event_id = tg$event_id)
#>   src dst t event_id
#> 1   1   2 1        2
#> 2   2   3 3        3
#> 3   3   1 5        1
#> 4   1   3 7        4
```

### Windows and splits

[`temporal_window()`](https://adamrsophie.github.io/graphlearnR/reference/temporal_window.md)
keeps events with `start <= t < end`.
[`split_temporal()`](https://adamrsophie.github.io/graphlearnR/reference/split_temporal.md)
cuts the stream at two times, so every validation event is later than
every training event and every test event is later still.

``` r

events <- dataset_synthetic_temporal(num_nodes = 30, num_events = 600, seed = 1)
events
#> <temporal_graph_data> 30 nodes, 600 events, t in [0.06052661, 99.63633]
sp <- split_temporal(events, val_time = 60, test_time = 80)
c(train = length(sp$train$t), val = length(sp$val$t), test = length(sp$test$t))
#> train   val  test 
#>   374   109   117
c(max(sp$train$t), min(sp$val$t), max(sp$val$t), min(sp$test$t))
#> [1] 59.88687 60.06185 79.95084 80.04763
```

### Causal neighborhoods

[`temporal_neighbors()`](https://adamrsophie.github.io/graphlearnR/reference/temporal_neighbors.md)
returns the `k` most recent events of each query node strictly before
its query time. With `strict = FALSE` it also keeps events at exactly
the query time; use that only when an event at time t really is known at
time t.

``` r

nb <- temporal_neighbors(events, nodes = c(1, 2), times = c(50, 50), k = 3,
                         direction = "both")
nb
#>   query node neighbor        t event
#> 1     1    1       16 49.91498   314
#> 2     1    1       12 49.48825   308
#> 3     1    1       16 43.77153   278
#> 4     2    2       26 49.77389   312
#> 5     2    2       20 47.88741   301
#> 6     2    2       20 47.21785   296
all(nb$t < 50)
#> [1] TRUE
```

## Snapshots

[`temporal_snapshots()`](https://adamrsophie.github.io/graphlearnR/reference/snapshot_sequence.md)
bins events into graphs. Each snapshot uses the half open interval
\[b_k, b\_{k+1}); repeated events between the same pair become the edge
weight. With `cumulative = TRUE` snapshot k holds every event before
b\_{k+1}.

``` r

snaps <- temporal_snapshots(events, breaks = 8)
snaps
#> <snapshot_sequence> 8 snapshots of 30 nodes
vapply(snaps$graphs, num_edges, integer(1))
#> [1]  66  80  90 102  90  94 110 102
round(snaps$start, 1)
#> [1]  0.1 12.5 25.0 37.4 49.8 62.3 74.7 87.2
```

### Dynamic link prediction

[`train_dynamic_link_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_dynamic_link_model.md)
runs a graph GRU over the snapshots. At each training step it predicts
the edges of the *next* snapshot from everything up to the current one,
with fresh negatives. Evaluation steps are strictly later than all
training steps, and the result carries a baseline that scores a pair by
how often it appeared before.

``` r

library(torch)
torch_manual_seed(1)
fit <- train_dynamic_link_model(snaps, hidden_dim = 16, epochs = 30, seed = 1)
fit
#> <temporal_fit> dynamic link prediction, 30 epochs
#>   test:      roc_auc = 0.8668, average_precision = 0.8378 
#>   baseline:  roc_auc = 0.8823, average_precision = 0.8624
fit$train_steps
#> [1] 2 3 4 5
fit$eval_steps
#> [1] 6 7 8
fit$step_metrics
#>   step   roc_auc average_precision
#> 1    6 0.8358986         0.8012241
#> 2    7 0.8899174         0.8639584
#> 3    8 0.8746636         0.8482256
rbind(model = fit$test_metrics, history_baseline = fit$baseline_metrics)
#>                    roc_auc average_precision
#> model            0.8668265         0.8378027
#> history_baseline 0.8823424         0.8624293
```

Report the model next to the baseline. On many real dynamic graphs,
simply predicting that past edges repeat is hard to beat, and a model
that does not beat it has not learned anything new.

## Spatiotemporal signals

A spatiotemporal signal is a fixed graph (a road network, a sensor grid)
with a feature array of shape time by nodes by features.

``` r

sig <- dataset_synthetic_signal(num_nodes = 15, num_steps = 80, seed = 2)
sig
#> <spatiotemporal_signal> 80 steps, 15 nodes, 1 features
dim(sig$features)
#> [1] 80 15  1
```

[`train_spatiotemporal_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_spatiotemporal_model.md)
forecasts `horizon` steps ahead from `lags` past steps with a graph GRU.
Windows are assigned to training, validation, and test sets in time
order, and the naive forecast (the last observed value) is reported next
to the model.

``` r

torch_manual_seed(2)
fst <- train_spatiotemporal_model(sig, lags = 4, horizon = 1, hidden_dim = 16,
                                  epochs = 40, seed = 2)
fst
#> <temporal_fit> spatiotemporal forecasting, 40 epochs
#>   test:      mse = 0.009564, mae = 0.07996 
#>   baseline:  mse = 0.01375, mae = 0.09504
c(last_train = max(fst$windows$train), first_val = min(fst$windows$val),
  first_test = min(fst$windows$test))
#> last_train  first_val first_test 
#>         56         57         68
rbind(model = fst$test_metrics, naive = fst$baseline_metrics)
#>              mse        mae
#> model 0.00956420 0.07995598
#> naive 0.01375186 0.09503502
```

To bring your own data, build the graph with any of the constructors and
pass the array to
[`spatiotemporal_signal()`](https://adamrsophie.github.io/graphlearnR/reference/spatiotemporal_signal.md),
which checks that the node dimension matches.

## The recurrent cell

[`graph_gru_cell()`](https://adamrsophie.github.io/graphlearnR/reference/graph_gru_cell.md)
replaces the matrix products of a GRU with graph convolutions, as in
DCRNN and GConvGRU, so the hidden state of each node mixes with its
neighbors at every step.
[`temporal_gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/temporal_gnn_model.md)
wraps it with an output layer. Its output at step s depends only on
inputs up to step s; the test suite checks this by changing later inputs
and comparing.

``` r

g <- sig$graph
tt <- graph_tensors(g)
torch_manual_seed(3)
m <- temporal_gnn_model(input_dim = 1, hidden_dim = 8, output_dim = 1)
xs <- lapply(1:5, function(s) torch_tensor(matrix(sig$features[s, , 1], ncol = 1)))
out <- m(xs, tt$edge_index)
length(out$outputs)
#> [1] 5
dim(out$outputs[[5]])
#> [1] 15  1
```
