# Download and read public benchmark datasets

These functions download data only when called, from the original
sources, into `root`, and check the archive's MD5 checksum when one is
known. Nothing is downloaded during package checks and no dataset is
redistributed with the package; check each source's terms of use before
redistributing the data.

## Usage

``` r
download_tu_dataset(name, root = tempdir(), md5 = NULL, overwrite = FALSE)

read_tu_dataset(dir, name = basename(dir))

download_planetoid_raw(
  name = c("cora", "citeseer"),
  root = tempdir(),
  md5 = NULL,
  overwrite = FALSE
)

download_pubmed_raw(root = tempdir(), md5 = NULL, overwrite = FALSE)

read_linqs_pubmed(dir)

read_linqs_citation(dir, name = basename(dir))
```

## Arguments

- name:

  Dataset name.

- root:

  Directory to store downloads.

- md5:

  Expected archive checksum; `NULL` uses the recorded value for known
  datasets and skips the check otherwise.

- overwrite:

  Download again even if the archive exists.

- dir:

  Directory with the extracted files.

## Value

`download_tu_dataset()` returns a `graph_dataset`;
`download_planetoid_raw()` returns a `graph_data`.

## Details

- `download_tu_dataset()`: TU Dortmund graph kernel benchmarks (Morris
  et al. 2020), for example `"MUTAG"` and `"PROTEINS"`, from
  <https://www.chrsmrrs.com/graphkerneldatasets/>.

- `download_planetoid_raw()`: the original LINQS releases of Cora and
  CiteSeer (Sen et al. 2008) from <https://linqs-data.soe.ucsc.edu/>;
  `download_pubmed_raw()` the Pubmed-Diabetes release (Namata et
  al. 2012) from the same site. These are the raw citation graphs. They
  are **not** the Planetoid splits used by Kipf and Welling (2017) and
  PyG's `Planetoid` class (which also drop and reorder nodes), so
  results are not comparable to published leaderboards. Open Graph
  Benchmark datasets are read by
  [`download_ogb()`](https://adamrsophie.github.io/graphlearnR/reference/read_ogb_node.md).

`read_tu_dataset()`, `read_linqs_citation()` and `read_linqs_pubmed()`
parse already downloaded files.

The Pubmed-Diabetes release stores sparse TF-IDF word weights as
`word=value` fields; absent words are zero. Its `.cites.tab` file lists
each link as `paper:A | paper:B` and does not document the direction, so
`read_linqs_pubmed()` keeps the file order (an edge from `A` to `B`);
use
[`as_undirected()`](https://adamrsophie.github.io/graphlearnR/reference/graph-transforms.md)
when direction should not matter. Links to unknown papers are dropped
and counted in `metadata$dropped_edges`.

In the LINQS `.cites` files each line is `cited citing`. The returned
graph is directed from the citing paper to the cited paper. Citations to
papers missing from `.content` are dropped and counted in
`metadata$dropped_edges`.

## Examples

``` r
if (FALSE) { # \dontrun{
mutag <- download_tu_dataset("MUTAG")
cora <- download_planetoid_raw("cora")
} # }
```
