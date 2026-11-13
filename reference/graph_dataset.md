# Graph datasets

`graph_dataset()` wraps a list of graphs in memory. A `transform`
function is applied lazily each time a graph is accessed with `[[`, so
the stored graphs are not modified. `graph_dataset_disk()` reads graphs
on demand from a directory written by `write_graph_dataset()`, one
`.rds` file per graph, so large collections need not fit in memory.

## Usage

``` r
graph_dataset(graphs, transform = NULL, metadata = list())

graph_schema(g)

graph_dataset_disk(dir, transform = NULL)

write_graph_dataset(graphs, dir, metadata = list(), overwrite = FALSE)
```

## Arguments

- graphs:

  A list of `graph_data` objects.

- transform:

  Optional function applied to each graph on access.

- metadata:

  Named list (name, source, version, license, notes).

- g:

  A `graph_data` object.

- dir:

  Directory holding `graph_000001.rds`, ... and `metadata.rds`.

- overwrite:

  Replace the graph files of an existing dataset in `dir`; without it,
  writing into such a directory is an error.

## Value

An object of class `graph_dataset`.

## Details

Both support [`length()`](https://rdrr.io/r/base/length.html), `[[` and
`[` (subset, returning a dataset), and can be passed to
[`graph_loader()`](https://adamrsophie.github.io/graphlearnR/reference/node_loader.md),
[`train_graph_model()`](https://adamrsophie.github.io/graphlearnR/reference/train_graph_model.md)
and
[`split_graphs()`](https://adamrsophie.github.io/graphlearnR/reference/split_graphs.md).

**Feature schema.** All graphs of a dataset must share one schema: the
same number (and names, when set) of node feature and edge feature
columns, and the same kind of node, edge and graph labels (factor,
numeric vector or matrix with the same number of columns), each present
in all graphs or in none. `graph_dataset()` and `write_graph_dataset()`
check this; `graph_schema()` returns the schema of one graph.

## Examples

``` r
ds <- graph_dataset(dataset_synthetic_graphs(10, seed = 1),
                    metadata = list(name = "toy"))
length(ds)
#> [1] 10
ds[[1]]
#> <graph_data> undirected graph: 5 nodes, 10 edges (5 undirected pairs)
#>   node features [5 x 1]
#>   graph label: factor with 2 levels
```
