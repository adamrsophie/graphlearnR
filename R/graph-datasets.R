#' Graph datasets
#'
#' `graph_dataset()` wraps a list of graphs in memory. A `transform` function
#' is applied lazily each time a graph is accessed with `[[`, so the stored
#' graphs are not modified. `graph_dataset_disk()` reads graphs on demand from
#' a directory written by [write_graph_dataset()], one `.rds` file per graph,
#' so large collections need not fit in memory.
#'
#' Both support `length()`, `[[` and `[` (subset, returning a dataset), and
#' can be passed to [graph_loader()], [train_graph_model()] and
#' [split_graphs()].
#'
#' @param graphs A list of `graph_data` objects.
#' @param transform Optional function applied to each graph on access.
#' @param metadata Named list (name, source, version, license, notes).
#' @return An object of class `graph_dataset`.
#' @examples
#' ds <- graph_dataset(dataset_synthetic_graphs(10, seed = 1),
#'                     metadata = list(name = "toy"))
#' length(ds)
#' ds[[1]]
#' @export
graph_dataset <- function(graphs, transform = NULL, metadata = list()) {
  if (inherits(graphs, "graph_dataset")) graphs <- lapply(seq_len(length(graphs)), function(i) graphs[[i]])
  if (!is.list(graphs) || !all(vapply(graphs, inherits, logical(1), "graph_data"))) {
    stopf("`graphs` must be a list of graph_data objects.")
  }
  structure(list(graphs = graphs, transform = transform, metadata = metadata,
                 dir = NULL), class = "graph_dataset")
}

#' @rdname graph_dataset
#' @param dir Directory holding `graph_000001.rds`, ... and `metadata.rds`.
#' @export
graph_dataset_disk <- function(dir, transform = NULL) {
  files <- sort(list.files(dir, pattern = "^graph_[0-9]+\\.rds$", full.names = TRUE))
  if (!length(files)) stopf("No graph files found in `%s`.", dir)
  meta_file <- file.path(dir, "metadata.rds")
  structure(list(graphs = NULL, files = files, transform = transform,
                 metadata = if (file.exists(meta_file)) readRDS(meta_file) else list(),
                 dir = dir), class = "graph_dataset")
}

#' @rdname graph_dataset
#' @param overwrite Replace the graph files of an existing dataset in `dir`;
#'   without it, writing into such a directory is an error.
#' @export
write_graph_dataset <- function(graphs, dir, metadata = list(), overwrite = FALSE) {
  old <- list.files(dir, pattern = "^graph_[0-9]+\\.rds$|^metadata\\.rds$", full.names = TRUE)
  if (length(old)) {
    # Stale graph files from an earlier, larger dataset would be read back.
    if (!overwrite) stopf("`%s` already holds a graph dataset; use `overwrite = TRUE` to replace it.", dir)
    unlink(old)
  }
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  n <- length(graphs)
  for (i in seq_len(n)) {
    saveRDS(graphs[[i]], file.path(dir, sprintf("graph_%06d.rds", i)))
  }
  saveRDS(metadata, file.path(dir, "metadata.rds"))
  invisible(dir)
}

#' @export
length.graph_dataset <- function(x) {
  if (is.null(x$dir)) length(x$graphs) else length(x$files)
}

#' @export
`[[.graph_dataset` <- function(x, i, ...) {
  if (is.character(i)) return(.subset2(x, i))
  g <- if (is.null(.subset2(x, "dir"))) .subset2(x, "graphs")[[i]] else readRDS(.subset2(x, "files")[[i]])
  tf <- .subset2(x, "transform")
  if (is.null(tf)) g else tf(g)
}

#' @export
`[.graph_dataset` <- function(x, i, ...) {
  out <- unclass(x)
  if (is.null(out$dir)) out$graphs <- out$graphs[i] else out$files <- out$files[i]
  structure(out, class = "graph_dataset")
}

#' @export
print.graph_dataset <- function(x, ...) {
  cat(sprintf("<graph_dataset> %d graphs%s%s\n", length(x),
              if (!is.null(x$dir)) " (on disk)" else "",
              if (!is.null(x$metadata$name)) paste0(": ", x$metadata$name) else ""))
  invisible(x)
}

#' Zachary's karate club
#'
#' The social network of 34 members of a university karate club observed by
#' Zachary (1977), with the two factions that formed after the club split
#' (17 members followed the instructor, "Mr. Hi"; 17 the administrator,
#' "Officer"). Edges come from `igraph::make_graph("Zachary")`; faction
#' labels follow the `club` attribute distributed with NetworkX. Node
#' features are one-hot node indicators (34 columns), the featureless
#' setting of Kipf and Welling (2017).
#'
#' @param features `"identity"` for one-hot indicators, `"degree"` for a
#'   single degree column, or `"none"`.
#' @return An undirected `graph_data` with 34 nodes, 78 edges and factor
#'   labels `y`.
#' @references Zachary, W. W. (1977). An information flow model for conflict
#'   and fission in small groups. Journal of Anthropological Research, 33(4),
#'   452 to 473.
#' @examples
#' g <- dataset_karate()
#' table(g$y)
#' @export
dataset_karate <- function(features = c("identity", "degree", "none")) {
  features <- match.arg(features)
  ig <- igraph::make_graph("Zachary")
  el <- igraph::as_edgelist(ig, names = FALSE)
  hi <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18, 20, 22)
  y <- factor(ifelse(seq_len(34) %in% hi, "Mr. Hi", "Officer"), levels = c("Mr. Hi", "Officer"))
  g <- graph_data(edge_index = t(el), y = y, num_nodes = 34, directed = FALSE,
                  metadata = list(name = "karate", source = "Zachary (1977)",
                                  license = "factual data, widely redistributed"))
  g$x <- switch(features,
    identity = diag(34),
    degree = matrix(as.double(degree(g)), ncol = 1L),
    none = NULL
  )
  g
}

#' Stochastic block model node classification data
#'
#' Nodes are split into blocks (classes); edges inside a block appear with
#' probability `p_in` and between blocks with `p_out`. Node features are a
#' class specific mean plus Gaussian noise, so both structure and features
#' carry signal; set `feature_signal = 0` for a structure only task.
#'
#' @param block_sizes Nodes per block.
#' @param p_in,p_out Within and between block edge probabilities.
#' @param num_features Feature columns.
#' @param feature_signal Distance between class means relative to the noise.
#' @param seed Optional integer seed.
#' @return An undirected `graph_data` with factor labels.
#' @examples
#' g <- dataset_sbm(c(30, 30, 30), seed = 1)
#' g
#' @export
dataset_sbm <- function(block_sizes = c(50, 50, 50), p_in = 0.1, p_out = 0.01,
                        num_features = 8L, feature_signal = 1, seed = NULL) {
  k <- length(block_sizes)
  n <- sum(block_sizes)
  with_optional_seed(seed, {
    cls <- rep(seq_len(k), block_sizes)
    pairs <- which(upper.tri(matrix(0, n, n)), arr.ind = TRUE)
    p <- ifelse(cls[pairs[, 1L]] == cls[pairs[, 2L]], p_in, p_out)
    keep <- stats::runif(nrow(pairs)) < p
    means <- matrix(stats::rnorm(k * num_features), k, num_features) * feature_signal
    x <- means[cls, , drop = FALSE] + matrix(stats::rnorm(n * num_features), n, num_features)
  })
  graph_data(x = x, edge_index = t(pairs[keep, , drop = FALSE]),
             y = factor(paste0("block", cls), levels = paste0("block", seq_len(k))),
             num_nodes = n, directed = FALSE,
             metadata = list(name = "sbm", source = "synthetic", seed = seed,
                             p_in = p_in, p_out = p_out))
}

#' Synthetic graph classification and regression data
#'
#' Small graphs whose label depends only on structure, so a model must use
#' message passing to solve the task. Class `"cycle"` graphs are rings; class
#' `"wheel"` graphs are rings plus a hub joined to every ring node. Sizes vary
#' from `min_nodes` to `max_nodes`. Node features are a constant column, so
#' features alone carry no information. For `task = "regression"`, `graph_y`
#' is the average degree.
#'
#' @param n Number of graphs.
#' @param min_nodes,max_nodes Ring size range.
#' @param task `"classification"` or `"regression"`.
#' @param seed Optional integer seed.
#' @return A list of undirected `graph_data` objects with `graph_y`.
#' @examples
#' gs <- dataset_synthetic_graphs(6, seed = 1)
#' vapply(gs, function(g) as.character(g$graph_y), "")
#' @export
dataset_synthetic_graphs <- function(n = 100L, min_nodes = 5L, max_nodes = 12L,
                                     task = c("classification", "regression"),
                                     seed = NULL) {
  task <- match.arg(task)
  with_optional_seed(seed, {
    sizes <- sample(min_nodes:max_nodes, n, replace = TRUE)
    kind <- sample(c("cycle", "wheel"), n, replace = TRUE)
  })
  lapply(seq_len(n), function(i) {
    m <- sizes[i]
    ring <- rbind(seq_len(m), c(seq_len(m)[-1L], 1L))
    ei <- if (kind[i] == "wheel") cbind(ring, rbind(rep(m + 1L, m), seq_len(m))) else ring
    nn <- m + (kind[i] == "wheel")
    g <- graph_data(x = matrix(1, nn, 1L), edge_index = ei, num_nodes = nn, directed = FALSE)
    g$graph_y <- if (task == "classification") {
      factor(kind[i], levels = c("cycle", "wheel"))
    } else {
      2 * num_edges(g, TRUE) / nn
    }
    g
  })
}

#' Download and read public benchmark datasets
#'
#' These functions download data only when called, from the original
#' sources, into `root`, and check the archive's MD5 checksum when one is
#' known. Nothing is downloaded during package checks and no dataset is
#' redistributed with the package; check each source's terms of use before
#' redistributing the data.
#'
#' * `download_tu_dataset()`: TU Dortmund graph kernel benchmarks (Morris et
#'   al. 2020), for example `"MUTAG"` and `"PROTEINS"`, from
#'   <https://www.chrsmrrs.com/graphkerneldatasets/>.
#' * `download_planetoid_raw()`: the original LINQS releases of Cora and
#'   CiteSeer (Sen et al. 2008) from <https://linqs-data.soe.ucsc.edu/>.
#'   These are the raw citation graphs. They are **not** the Planetoid splits
#'   used by Kipf and Welling (2017) and PyG's `Planetoid` class (which also
#'   drop and reorder nodes), so results are not comparable to published
#'   leaderboards. Open Graph Benchmark datasets are read by
#'   [download_ogb()].
#'
#' `read_tu_dataset()` and `read_linqs_citation()` parse already downloaded
#' files.
#'
#' @param name Dataset name.
#' @param root Directory to store downloads.
#' @param md5 Expected archive checksum; `NULL` uses the recorded value
#'   for known datasets and skips the check otherwise.
#' @param overwrite Download again even if the archive exists.
#' @return `download_tu_dataset()` returns a `graph_dataset`;
#'   `download_planetoid_raw()` returns a `graph_data`.
#' @examples
#' \dontrun{
#' mutag <- download_tu_dataset("MUTAG")
#' cora <- download_planetoid_raw("cora")
#' }
#' @name download-datasets
NULL

known_md5 <- c(MUTAG = "dbe05fcee4e85280424d20db065b3e60",
               PROTEINS = "7afcf38db3ae46202b853f99e5230d6a",
               cora = "2fc040bee8ce3d920e4204effd1e9214",
               citeseer = "c8ded8ed395b31899576bfd1e91e4d6e")

fetch_archive <- function(url, dest, md5, overwrite) {
  if (overwrite || !file.exists(dest)) {
    dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
    utils::download.file(url, dest, mode = "wb", quiet = TRUE)
  }
  if (is.null(md5)) {
    warnf("No checksum is recorded for `%s`; the download is not verified. Pass `md5` to check it.",
          basename(dest))
  } else {
    got <- unname(tools::md5sum(dest))
    if (!identical(got, md5)) {
      stopf("Checksum mismatch for `%s`: expected %s, got %s.", dest, md5, got)
    }
  }
  dest
}

#' @rdname download-datasets
#' @export
download_tu_dataset <- function(name, root = tempdir(), md5 = NULL, overwrite = FALSE) {
  url <- sprintf("https://www.chrsmrrs.com/graphkerneldatasets/%s.zip", name)
  dest <- file.path(root, paste0(name, ".zip"))
  fetch_archive(url, dest, md5 %||% if (name %in% names(known_md5)) known_md5[[name]], overwrite)
  utils::unzip(dest, exdir = root)
  read_tu_dataset(file.path(root, name), name)
}

#' @rdname download-datasets
#' @param dir Directory with the extracted files.
#' @export
read_tu_dataset <- function(dir, name = basename(dir)) {
  f <- function(s) file.path(dir, paste0(name, "_", s, ".txt"))
  rd <- function(s) if (file.exists(f(s))) as.matrix(utils::read.csv(f(s), header = FALSE))
  A <- rd("A")
  gi <- as.integer(rd("graph_indicator"))
  gl <- as.vector(rd("graph_labels"))
  nl <- rd("node_labels")
  na <- rd("node_attributes")
  el <- rd("edge_labels")
  ea <- rd("edge_attributes")
  if (is.null(A) || is.null(gi)) stopf("`%s` does not look like a TU dataset.", dir)
  local <- stats::ave(seq_along(gi), gi, FUN = seq_along)
  node_feat <- cbind(if (!is.null(nl)) one_hot(nl[, 1L]), na)
  edge_feat <- cbind(if (!is.null(el)) one_hot(el[, 1L]), ea)
  eg <- gi[A[, 1L]]
  lab <- factor(gl)
  graphs <- lapply(sort(unique(gi)), function(k) {
    nodes <- which(gi == k)
    edges <- which(eg == k)
    ei <- rbind(local[A[edges, 1L]], local[A[edges, 2L]])
    g <- graph_data(x = if (!is.null(node_feat)) node_feat[nodes, , drop = FALSE] else matrix(1, length(nodes), 1L),
                    edge_index = ei,
                    edge_attr = if (!is.null(edge_feat)) edge_feat[edges, , drop = FALSE],
                    num_nodes = length(nodes), directed = TRUE)
    g$graph_y <- lab[k]
    g
  })
  graph_dataset(graphs, metadata = list(
    name = name, source = "TU Dortmund graph kernel datasets (Morris et al. 2020)",
    url = "https://www.chrsmrrs.com/graphkerneldatasets/",
    note = "Edges are kept as listed (TU files list both directions of undirected edges)."))
}

one_hot <- function(v) {
  f <- factor(v)
  m <- matrix(0, length(v), nlevels(f))
  m[cbind(seq_along(v), as.integer(f))] <- 1
  colnames(m) <- paste0("lab_", levels(f))
  m
}

#' @rdname download-datasets
#' @export
download_planetoid_raw <- function(name = c("cora", "citeseer"), root = tempdir(),
                                   md5 = NULL, overwrite = FALSE) {
  name <- match.arg(name)
  url <- sprintf("https://linqs-data.soe.ucsc.edu/public/lbc/%s.tgz", name)
  dest <- file.path(root, paste0(name, ".tgz"))
  fetch_archive(url, dest, md5 %||% known_md5[[name]], overwrite)
  utils::untar(dest, exdir = root)
  read_linqs_citation(file.path(root, name), name)
}

#' @rdname download-datasets
#' @details In the LINQS `.cites` files each line is `cited citing`. The
#'   returned graph is directed from the citing paper to the cited paper.
#'   Citations to papers missing from `.content` are dropped and counted in
#'   `metadata$dropped_edges`.
#' @export
read_linqs_citation <- function(dir, name = basename(dir)) {
  content <- utils::read.delim(file.path(dir, paste0(name, ".content")), header = FALSE,
                               colClasses = "character")
  cites <- utils::read.delim(file.path(dir, paste0(name, ".cites")), header = FALSE,
                             colClasses = "character")
  ids <- content[[1L]]
  x <- as.matrix(content[, 2:(ncol(content) - 1L)])
  storage.mode(x) <- "double"
  y <- factor(content[[ncol(content)]])
  cited <- match(cites[[1L]], ids)
  citing <- match(cites[[2L]], ids)
  ok <- !is.na(cited) & !is.na(citing)
  graph_data(x = x, edge_index = rbind(citing[ok], cited[ok]), y = y,
             num_nodes = length(ids), directed = TRUE, node_ids = ids,
             metadata = list(name = name, source = "LINQS (Sen et al. 2008)",
                             url = "https://linqs-data.soe.ucsc.edu/public/lbc/",
                             dropped_edges = sum(!ok),
                             note = "Raw release; not the Planetoid split."))
}
