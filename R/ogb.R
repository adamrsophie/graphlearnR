#' Open Graph Benchmark datasets
#'
#' Read node and graph property prediction datasets of the Open Graph
#' Benchmark (Hu et al. 2020) from the CSV files that OGB distributes, and
#' optionally download them first. Nothing is downloaded during package
#' checks and no data are redistributed; each dataset has its own license,
#' listed on the OGB website.
#'
#' * `read_ogb_node()` reads an extracted `ogbn-*` folder (for example
#'   `arxiv/`) into one `graph_data` with node features, labels, and the
#'   official split as `train_mask`, `val_mask` and `test_mask`.
#' * `read_ogb_graph()` reads an extracted `ogbg-*` folder (for example
#'   `hiv/`) into a `graph_dataset` with the official split in
#'   `metadata$split` (1-based graph indices).
#'
#' OGB files are 0-based; indices are converted to 1-based. When the dataset
#' stores each undirected edge once (`add_inverse_edge` in OGB's
#' `master.csv`), the graph is built as undirected so both directions are
#' stored. Molecule datasets carry categorical atom and bond codes in their
#' feature columns (OGB's `AtomEncoder` and `BondEncoder` inputs); they are
#' kept as numbers, so one hot encode or embed them before training.
#' Evaluate with the official split and metric (ROC AUC for `ogbg-molhiv`,
#' accuracy for `ogbn-arxiv`) before comparing with the OGB leaderboards.
#' Link property prediction datasets store their splits as PyTorch files
#' and are not supported.
#'
#' @param dir Extracted dataset folder.
#' @param split Split folder name (for example `"time"`, `"sales_ranking"`,
#'   `"species"` or `"scaffold"`); default the only one present.
#' @param add_inverse_edge Store both directions of every edge.
#' @param name OGB dataset name, `"ogbn-arxiv"`, `"ogbn-products"`,
#'   `"ogbg-molhiv"` or `"ogbg-molpcba"`.
#' @param root Download directory.
#' @param md5 Expected archive checksum; `NULL` uses the recorded value when
#'   there is one (currently only `ogbg-molhiv`; the others are not checked
#'   unless `md5` is given).
#' @param overwrite Download again.
#' @return `read_ogb_node()`: a `graph_data` object. `read_ogb_graph()`: a
#'   `graph_dataset`. `download_ogb()`: either, depending on the dataset.
#' @references Hu, Fey, Zitnik, Dong, Ren, Liu, Catasta and Leskovec (2020)
#'   Open Graph Benchmark: datasets for machine learning on graphs.
#' @examples
#' g <- read_ogb_node(system.file("extdata", "ogbn-toy", package = "graphlearnR"))
#' g
#' ds <- read_ogb_graph(system.file("extdata", "ogbg-toy", package = "graphlearnR"))
#' ds
#' ds$metadata$split
#' \dontrun{
#' hiv <- download_ogb("ogbg-molhiv")
#' }
#' @export
read_ogb_node <- function(dir, split = NULL, add_inverse_edge = FALSE) {
  raw <- function(f) ogb_csv(file.path(dir, "raw", f))
  ei <- raw("edge.csv.gz")
  if (is.null(ei)) stopf("`%s` does not look like an OGB node dataset (no raw/edge.csv.gz).", dir)
  n <- sum(raw("num-node-list.csv.gz") %||% (max(ei) + 1))
  x <- raw("node-feat.csv.gz")
  lab <- raw("node-label.csv.gz")
  y <- if (!is.null(lab)) {
    if (ncol(lab) == 1L) factor(lab[, 1L]) else lab
  }
  g <- graph_data(x = x, edge_index = t(ei) + 1L, y = y, edge_attr = raw("edge-feat.csv.gz"),
                  num_nodes = n, directed = !add_inverse_edge)
  sp <- ogb_split(dir, split)
  if (!is.null(sp)) {
    for (k in c("train", "val", "test")) {
      m <- logical(n)
      m[sp[[k]]] <- TRUE
      g[[paste0(k, "_mask")]] <- m
    }
  }
  g$metadata <- list(source = "Open Graph Benchmark", url = "https://ogb.stanford.edu/",
                     folder = basename(dir), split = attr(sp, "name"))
  g
}

#' @rdname read_ogb_node
#' @export
read_ogb_graph <- function(dir, split = NULL, add_inverse_edge = TRUE) {
  raw <- function(f) ogb_csv(file.path(dir, "raw", f))
  ei <- raw("edge.csv.gz")
  nn <- as.integer(raw("num-node-list.csv.gz"))
  ne <- as.integer(raw("num-edge-list.csv.gz"))
  if (is.null(ei) || !length(nn)) stopf("`%s` does not look like an OGB graph dataset.", dir)
  x <- raw("node-feat.csv.gz")
  ea <- raw("edge-feat.csv.gz")
  lab <- raw("graph-label.csv.gz")
  n0 <- c(0L, cumsum(nn))
  e0 <- c(0L, cumsum(ne))
  graphs <- lapply(seq_along(nn), function(k) {
    nodes <- seq_len(nn[k]) + n0[k]
    edges <- seq_len(ne[k]) + e0[k]
    g <- graph_data(x = if (!is.null(x)) x[nodes, , drop = FALSE] else matrix(1, nn[k], 1L),
                    edge_index = t(ei[edges, , drop = FALSE]) + 1L,
                    edge_attr = if (!is.null(ea)) ea[edges, , drop = FALSE],
                    num_nodes = nn[k], directed = !add_inverse_edge)
    if (!is.null(lab)) g$graph_y <- if (ncol(lab) == 1L) lab[k, 1L] else lab[k, ]
    g
  })
  if (!is.null(lab) && ncol(lab) == 1L && all(lab[, 1L] %in% c(0, 1), na.rm = TRUE)) {
    for (k in seq_along(graphs)) graphs[[k]]$graph_y <- factor(lab[k, 1L], levels = c(0, 1))
  }
  sp <- ogb_split(dir, split)
  graph_dataset(graphs, metadata = list(
    source = "Open Graph Benchmark", url = "https://ogb.stanford.edu/", folder = basename(dir),
    split = if (!is.null(sp)) unclass(sp)[c("train", "val", "test")], split_name = attr(sp, "name")))
}

ogb_csv <- function(path) {
  if (!file.exists(path)) return(NULL)
  m <- tryCatch(as.matrix(utils::read.csv(path, header = FALSE)),
                error = function(e) matrix(integer(0), 0L, 1L))
  dimnames(m) <- NULL
  if (all(m == round(m), na.rm = TRUE)) storage.mode(m) <- "integer"
  m
}

ogb_split <- function(dir, split) {
  sdir <- file.path(dir, "split")
  if (!dir.exists(sdir)) return(NULL)
  if (is.null(split)) {
    split <- list.dirs(sdir, full.names = FALSE, recursive = FALSE)
    if (length(split) == 0L) return(NULL)
    if (length(split) > 1L) stopf("Several splits in `%s`; choose one with `split`.", sdir)
  }
  rd <- function(f) as.integer(ogb_csv(file.path(sdir, split, f))) + 1L
  structure(list(train = rd("train.csv.gz"), val = rd("valid.csv.gz"), test = rd("test.csv.gz")),
            name = split)
}

ogb_registry <- list(
  "ogbn-arxiv" = list(url = "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
                      type = "node", inverse = FALSE, md5 = NULL),
  "ogbn-products" = list(url = "https://snap.stanford.edu/ogb/data/nodeproppred/products.zip",
                         type = "node", inverse = TRUE, md5 = NULL),
  "ogbg-molhiv" = list(url = "https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip",
                       type = "graph", inverse = TRUE, md5 = "21a81e983fc62dea72f9653683577eb6"),
  "ogbg-molpcba" = list(url = "https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/pcba.zip",
                        type = "graph", inverse = TRUE, md5 = NULL)
)

#' @rdname read_ogb_node
#' @export
download_ogb <- function(name, root = tempdir(), md5 = NULL, overwrite = FALSE) {
  info <- ogb_registry[[name]]
  if (is.null(info)) {
    stopf("Unknown OGB dataset `%s`; supported: %s.", name, paste(names(ogb_registry), collapse = ", "))
  }
  dest <- file.path(root, basename(info$url))
  fetch_archive(info$url, dest, md5 %||% info$md5, overwrite)
  files <- utils::unzip(dest, list = TRUE)$Name
  edge <- grep("(^|/)raw/edge\\.csv\\.gz$", files, value = TRUE)
  if (length(edge) != 1L) stopf("Unexpected archive layout in `%s`.", dest)
  utils::unzip(dest, exdir = root)
  dir <- file.path(root, dirname(dirname(edge)))
  if (info$type == "node") read_ogb_node(dir, add_inverse_edge = info$inverse)
  else read_ogb_graph(dir, add_inverse_edge = info$inverse)
}
