#' Cluster and GraphSAINT subgraph loaders
#'
#' Loaders that train node models on whole subgraphs instead of sampled
#' neighborhoods. Pass one to [train_graph_model()] as `loader`; every batch
#' is an induced subgraph, the loss uses the training nodes inside it, and
#' validation uses the full graph.
#'
#' * `cluster_loader()` follows Cluster-GCN (Chiang et al. 2019): the graph is
#'   partitioned once into `num_parts` parts and each batch is the subgraph
#'   induced by `clusters_per_batch` randomly chosen parts, including the
#'   edges between them. With `method = "louvain"`, nodes are ordered by
#'   Louvain community (igraph's `cluster_louvain()` on the undirected view)
#'   and then by breadth first search order, and the ordering is cut into
#'   `num_parts` parts of equal size, so parts follow communities where
#'   sizes allow; `method = "random"` assigns nodes at random. METIS is not
#'   used.
#' * `saint_loader()` follows the GraphSAINT random walk sampler (Zeng et al.
#'   2020): each batch is the subgraph induced by the nodes visited by
#'   `batch_size` random walks of length `walk_length` from uniformly drawn
#'   roots. Normalization coefficients are estimated from `sample_coverage`
#'   pre-sampled subgraphs, as in PyTorch Geometric's `GraphSAINTSampler`:
#'   each batch carries `node_norm` (loss weight \eqn{\propto 1 / p_v}) and
#'   edge weights multiplied by \eqn{p_v / p_{uv}} (aggregator
#'   normalization), where \eqn{p} are empirical inclusion frequencies.
#'   The trainer sums the per node loss of the training nodes in a batch,
#'   weighted by `node_norm`, for the built in losses and for loss functions
#'   with a `reduction` argument; in expectation this is the mean loss over
#'   all nodes of the graph.
#'
#' Batches are drawn in R on the CPU.
#'
#' @param g A `graph_data` object.
#' @param num_parts Number of parts.
#' @param clusters_per_batch Parts combined into one batch.
#' @param method `"louvain"` or `"random"`.
#' @param shuffle Shuffle parts every epoch.
#' @param seed Optional seed; the caller's random stream is left untouched.
#' @return A loader of class `subgraph_loader`; [loader_batches()] returns a
#'   list of `graph_data` batches whose `node_ids` hold original node indices.
#' @references Chiang et al. (2019) Cluster-GCN. Zeng et al. (2020)
#'   GraphSAINT: graph sampling based inductive learning method.
#' @examples
#' g <- split_nodes(dataset_sbm(c(40, 40, 40), seed = 1), seed = 1)
#' cl <- cluster_loader(g, num_parts = 6, clusters_per_batch = 2, seed = 1)
#' lengths(cl$parts)
#' vapply(loader_batches(cl), num_nodes, 1L)
#' sl <- saint_loader(g, batch_size = 10, walk_length = 3, seed = 1)
#' summary(loader_batches(sl)[[1]]$node_norm)
#' @export
cluster_loader <- function(g, num_parts = 10L, clusters_per_batch = 1L,
                           method = c("louvain", "random"), shuffle = TRUE, seed = NULL) {
  method <- match.arg(method)
  num_parts <- check_count(num_parts, "num_parts", 1L)
  clusters_per_batch <- check_count(clusters_per_batch, "clusters_per_batch", 1L)
  n <- g$num_nodes
  if (num_parts > n) stopf("`num_parts` (%d) exceeds the number of nodes (%d).", num_parts, n)
  part <- with_optional_seed(seed, {
    if (method == "random") {
      sample(rep_len(seq_len(num_parts), n))
    } else {
      ig <- igraph::graph_from_edgelist(t(g$edge_index), directed = FALSE)
      ig <- igraph::add_vertices(ig, n - igraph::vcount(ig))
      ig <- igraph::simplify(ig)
      comm <- as.integer(igraph::membership(igraph::cluster_louvain(ig)))
      bfs <- as.integer(igraph::bfs(ig, root = 1, unreachable = TRUE)$order)
      rank <- integer(n)
      rank[bfs] <- seq_along(bfs)
      # Nodes sorted by community, then breadth first order, cut into equal chunks.
      ord <- order(comm, rank)
      part <- integer(n)
      part[ord] <- ceiling(seq_len(n) * num_parts / n)
      part
    }
  })
  structure(list(graph = g, parts = unname(split(seq_len(n), factor(part, seq_len(num_parts)))),
                 clusters_per_batch = clusters_per_batch, shuffle = shuffle, seed = seed,
                 method = method),
            class = c("cluster_loader", "subgraph_loader", "graph_loader_base"))
}

#' @export
loader_batches.cluster_loader <- function(loader, epoch = 1L) {
  s <- if (is.null(loader$seed)) NULL else loader$seed + epoch
  groups <- batch_split(length(loader$parts), loader$clusters_per_batch, loader$shuffle, s)
  unname(lapply(groups, function(idx) {
    nodes <- sort(unlist(loader$parts[idx], use.names = FALSE))
    subgraph_batch(loader$graph, nodes)
  }))
}

subgraph_batch <- function(g, nodes) {
  sub <- induced_subgraph(g, nodes)
  sub$node_ids <- nodes
  sub$edge_original <- NULL
  sub
}

#' @rdname cluster_loader
#' @param batch_size Number of random walk roots per batch.
#' @param walk_length Steps per walk.
#' @param num_steps Batches per epoch.
#' @param sample_coverage Number of subgraphs pre-sampled per node (on
#'   average) to estimate normalization; `0` disables normalization.
#' @export
saint_loader <- function(g, batch_size = 100L, walk_length = 2L, num_steps = 10L,
                         sample_coverage = 20L, seed = NULL) {
  batch_size <- check_count(batch_size, "batch_size", 1L)
  walk_length <- check_count(walk_length, "walk_length", 0L)
  num_steps <- check_count(num_steps, "num_steps", 1L)
  sample_coverage <- check_count(sample_coverage, "sample_coverage", 0L)
  ei <- g$edge_index
  ord <- order(ei[1L, ])
  csr <- list(ptr = c(0L, cumsum(tabulate(ei[1L, ], g$num_nodes))), dst = ei[2L, ord],
              eid = ord)
  loader <- structure(list(graph = g, csr = csr, batch_size = batch_size,
                           walk_length = walk_length, num_steps = num_steps, seed = seed,
                           node_norm = NULL, edge_norm = NULL),
                      class = c("saint_loader", "subgraph_loader", "graph_loader_base"))
  if (sample_coverage > 0L) {
    n <- g$num_nodes
    node_count <- numeric(n)
    edge_count <- numeric(ncol(ei))
    num_samples <- 0L
    s <- if (is.null(seed)) NULL else seed - 1L
    with_optional_seed(s, {
      while (num_samples < 1L || sum(node_count) < sample_coverage * n) {
        nodes <- saint_walk_nodes(loader)
        node_count[nodes] <- node_count[nodes] + 1
        inside <- logical(n)
        inside[nodes] <- TRUE
        e <- which(inside[ei[1L, ]] & inside[ei[2L, ]])
        edge_count[e] <- edge_count[e] + 1
        num_samples <- num_samples + 1L
      }
    })
    loader$node_norm <- ifelse(node_count > 0, num_samples / pmax(node_count, 1) / n, 0)
    loader$edge_norm <- pmin(ifelse(edge_count > 0, node_count[ei[2L, ]] / pmax(edge_count, 1), 1), 1e4)
    loader$num_samples <- num_samples
  }
  loader
}

saint_walk_nodes <- function(loader) {
  csr <- loader$csr
  n <- loader$graph$num_nodes
  cur <- sample.int(n, loader$batch_size, replace = TRUE)
  visited <- cur
  for (step in seq_len(loader$walk_length)) {
    deg <- csr$ptr[cur + 1L] - csr$ptr[cur]
    pick <- floor(stats::runif(length(cur)) * deg)
    has <- deg > 0L
    cur[has] <- csr$dst[csr$ptr[cur[has]] + pick[has] + 1L]
    visited <- c(visited, cur)
  }
  sort(unique(visited))
}

#' @export
loader_batches.saint_loader <- function(loader, epoch = 1L) {
  s <- if (is.null(loader$seed)) NULL else loader$seed + epoch
  g <- loader$graph
  with_optional_seed(s, lapply(seq_len(loader$num_steps), function(i) {
    nodes <- saint_walk_nodes(loader)
    sub <- subgraph_batch(g, nodes)
    if (!is.null(loader$node_norm)) {
      sub$node_norm <- loader$node_norm[nodes]
      local <- integer(g$num_nodes)
      local[nodes] <- 1L
      keep <- which(local[g$edge_index[1L, ]] > 0L & local[g$edge_index[2L, ]] > 0L)
      w <- g$edge_weight[keep] %||% rep(1, length(keep))
      sub$edge_weight <- w * loader$edge_norm[keep]
    }
    sub
  }))
}

#' @export
print.subgraph_loader <- function(x, ...) {
  if (inherits(x, "cluster_loader")) {
    cat(sprintf("<cluster_loader> %d parts (%s), %d per batch\n", length(x$parts), x$method,
                x$clusters_per_batch))
  } else {
    cat(sprintf("<saint_loader> %d roots, walk length %d, %d batches per epoch%s\n",
                x$batch_size, x$walk_length, x$num_steps,
                if (is.null(x$node_norm)) "" else sprintf(", normalized from %d samples", x$num_samples)))
  }
  invisible(x)
}
