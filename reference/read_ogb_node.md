# Open Graph Benchmark datasets

Read node and graph property prediction datasets of the Open Graph
Benchmark (Hu et al. 2020) from the CSV files that OGB distributes, and
optionally download them first. Nothing is downloaded during package
checks and no data are redistributed; each dataset has its own license,
listed on the OGB website.

## Usage

``` r
read_ogb_node(dir, split = NULL, add_inverse_edge = FALSE)

read_ogb_graph(dir, split = NULL, add_inverse_edge = TRUE)

download_ogb(name, root = tempdir(), md5 = NULL, overwrite = FALSE)
```

## Arguments

- dir:

  Extracted dataset folder.

- split:

  Split folder name (for example `"time"`, `"sales_ranking"`,
  `"species"` or `"scaffold"`); default the only one present.

- add_inverse_edge:

  Store both directions of every edge.

- name:

  OGB dataset name, `"ogbn-arxiv"`, `"ogbn-products"`, `"ogbg-molhiv"`
  or `"ogbg-molpcba"`.

- root:

  Download directory.

- md5:

  Expected archive checksum; `NULL` uses the recorded value when there
  is one (currently only `ogbg-molhiv`; the others are not checked
  unless `md5` is given).

- overwrite:

  Download again.

## Value

`read_ogb_node()`: a `graph_data` object. `read_ogb_graph()`: a
`graph_dataset`. `download_ogb()`: either, depending on the dataset.

## Details

- `read_ogb_node()` reads an extracted `ogbn-*` folder (for example
  `arxiv/`) into one `graph_data` with node features, labels, and the
  official split as `train_mask`, `val_mask` and `test_mask`.

- `read_ogb_graph()` reads an extracted `ogbg-*` folder (for example
  `hiv/`) into a `graph_dataset` with the official split in
  `metadata$split` (1-based graph indices).

OGB files are 0-based; indices are converted to 1-based. When the
dataset stores each undirected edge once (`add_inverse_edge` in OGB's
`master.csv`), the graph is built as undirected so both directions are
stored. Molecule datasets carry categorical atom and bond codes in their
feature columns (OGB's `AtomEncoder` and `BondEncoder` inputs); they are
kept as numbers, so one hot encode or embed them before training.
Evaluate with the official split and metric (ROC AUC for `ogbg-molhiv`,
accuracy for `ogbn-arxiv`) before comparing with the OGB leaderboards.
Link property prediction datasets store their splits as PyTorch files
and are not supported.

## References

Hu, Fey, Zitnik, Dong, Ren, Liu, Catasta and Leskovec (2020) Open Graph
Benchmark: datasets for machine learning on graphs.

## Examples

``` r
g <- read_ogb_node(system.file("extdata", "ogbn-toy", package = "graphlearnR"))
g
#> <graph_data> directed graph: 6 nodes, 6 edges
#>   node features [6 x 2]
#>   node labels: factor with 3 levels
#>   masks: train=3, val=1, test=2
ds <- read_ogb_graph(system.file("extdata", "ogbg-toy", package = "graphlearnR"))
ds
#> <graph_dataset> 2 graphs
ds$metadata$split
#> $train
#> [1] 1
#> 
#> $val
#> [1] 2
#> 
#> $test
#> integer(0)
#> 
if (FALSE) { # \dontrun{
hiv <- download_ogb("ogbg-molhiv")
} # }
```
