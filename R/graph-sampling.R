#' Multi-hop neighbor sampling
#'
#' Samples the computation graph of a set of seed nodes, as in GraphSAGE
#' (Hamilton et al. 2017) and PyG's `NeighborLoader`. Hop `k` samples up to
#' `fanouts[k]` **incoming** edges of every node reached in hop `k - 1`
#' (hop 0 is the seed set); sources of the sampled edges join the subgraph.
#' Messages in a GNN flow along these edges towards the seeds, so an
#' `L` layer model with `L` fanouts sees exactly the information it would see
#' on the full graph when every fanout is `-1` (all neighbors).
#'
#' The result is a `graph_data` object with local indices and these extra
#' fields:
#'
#' * `n_id`: original index of every local node; the seeds come first, in
#'   the order given, so `seed_index = seq_along(seeds)`;
#' * `e_id`: original stored edge index of every sampled edge;
#' * `hop`: hop at which each node was first reached (0 for seeds);
#' * `num_seeds`.
#'
#' Losses and metrics must use only the seed rows (`seed_index`); the other
#' nodes are context. [train_graph_model()] does this automatically.
#' The sampled graph is marked directed because sampling may keep only one
#' direction of an undirected pair.
#'
#' Sampling uses R's random number generator. With `seed`, it runs inside
#' [withr::with_seed()], so the result is reproducible and the caller's
#' random state is left untouched.
#'
#' @param g A `graph_data` object.
#' @param seeds Integer vector of seed node indices (one based).
#' @param fanouts Integer vector, one entry per hop; `-1` keeps all incoming
#'   edges.
#' @param replace Sample edges with replacement.
#' @param seed Optional integer seed.
#' @return A `graph_data` object of class `sampled_graph`.
#' @examples
#' g <- dataset_karate()
#' sg <- sample_neighbors(g, seeds = c(1, 34), fanouts = c(3, 2), seed = 1)
#' sg$num_seeds
#' sg$n_id
#' @export
sample_neighbors <- function(g, seeds, fanouts, replace = FALSE, seed = NULL) {
  seeds <- as.integer(seeds)
  n <- g$num_nodes
  if (!length(seeds) || anyNA(seeds) || any(seeds < 1L | seeds > n)) {
    stopf("`seeds` must be node indices in 1:%d.", n)
  }
  if (anyDuplicated(seeds)) stopf("`seeds` must not contain duplicates.")
  fanouts <- as.integer(fanouts)
  csc <- incoming_index(g)
  with_optional_seed(seed, {
    local <- integer(n)
    local[seeds] <- seq_along(seeds)
    n_id <- seeds
    hop <- integer(length(seeds))
    frontier <- seeds
    e_keep <- integer()
    for (k in seq_along(fanouts)) {
      if (!length(frontier)) break
      eids <- sample_incoming(csc, frontier, fanouts[k], replace)
      e_keep <- c(e_keep, eids)
      srcs <- g$edge_index[1L, eids]
      new <- unique(srcs[local[srcs] == 0L])
      if (length(new)) {
        local[new] <- length(n_id) + seq_along(new)
        n_id <- c(n_id, new)
        hop <- c(hop, rep.int(k, length(new)))
      }
      frontier <- new
    }
    if (!replace) e_keep <- unique(e_keep)
    induced_from_edges(g, n_id, local, e_keep, hop, length(seeds))
  })
}

# Edges grouped by target: ord lists edge ids sorted by target, ptr gives the
# start offset (zero based) of each target's block.
incoming_index <- function(g) {
  dst <- g$edge_index[2L, ]
  ord <- order(dst, method = "radix")
  cnt <- tabulate(dst, nbins = g$num_nodes)
  list(ord = ord, ptr = c(0L, cumsum(cnt)), cnt = cnt)
}

sample_incoming <- function(csc, nodes, fanout, replace) {
  cnt <- csc$cnt[nodes]
  has <- cnt > 0L
  nodes <- nodes[has]
  cnt <- cnt[has]
  if (!length(nodes)) {
    return(integer())
  }
  start <- csc$ptr[nodes]
  if (fanout < 0L || (!replace && all(cnt <= fanout))) {
    return(csc$ord[sequence(cnt, from = start + 1L)])
  }
  if (replace) {
    grp <- rep.int(seq_along(nodes), fanout)
    off <- floor(stats::runif(length(grp)) * cnt[grp])
    return(csc$ord[start[grp] + off + 1L])
  }
  pos <- sequence(cnt, from = start + 1L)
  grp <- rep.int(seq_along(nodes), cnt)
  key <- stats::runif(length(pos))
  o <- order(grp, key, method = "radix")
  rank <- sequence(cnt)
  keep <- o[rank <= fanout]
  csc$ord[pos[sort(keep)]]
}

induced_from_edges <- function(g, n_id, local, e_id, hop, num_seeds) {
  ei <- if (length(e_id)) {
    rbind(local[g$edge_index[1L, e_id]], local[g$edge_index[2L, e_id]])
  } else {
    matrix(integer(), 2L, 0L)
  }
  sg <- new_graph_data(
    x = take_rows(g$x, n_id), edge_index = ei, y = take_rows(g$y, n_id),
    edge_attr = take_rows(g$edge_attr, e_id),
    edge_weight = take_rows(g$edge_weight, e_id),
    edge_y = take_rows(g$edge_y, e_id),
    graph_y = g$graph_y, graph_attr = g$graph_attr,
    num_nodes = length(n_id), directed = TRUE,
    node_ids = g$node_ids[n_id], node_type = take_rows(g$node_type, n_id),
    edge_type = take_rows(g$edge_type, e_id), edge_pair = seq_along(e_id),
    train_mask = take_rows(g$train_mask, n_id),
    val_mask = take_rows(g$val_mask, n_id),
    test_mask = take_rows(g$test_mask, n_id),
    metadata = c(g$metadata, list(sampled_from_directed = g$directed)),
    class = "sampled_graph"
  )
  sg$n_id <- n_id
  sg$e_id <- e_id
  sg$hop <- hop
  sg$num_seeds <- num_seeds
  sg$seed_index <- seq_len(num_seeds)
  sg
}

#' K-hop subgraph
#'
#' Exact `k` hop computation graph of `nodes`: all incoming edges up to `k`
#' hops, equivalent to [sample_neighbors()] with every fanout `-1`.
#'
#' @inheritParams sample_neighbors
#' @param nodes Seed nodes.
#' @param k Number of hops.
#' @return A `sampled_graph`.
#' @export
k_hop_subgraph <- function(g, nodes, k) {
  sample_neighbors(g, nodes, rep(-1L, check_count(k, "k")))
}

#' Mini-batch loaders
#'
#' `node_loader()` splits seed nodes into batches and samples each batch's
#' neighborhood with [sample_neighbors()]. `graph_loader()` splits a list of
#' graphs (or a [graph_dataset()]) into [graph_batch()] objects. Call
#' [loader_batches()] to draw the batches of one epoch; with a `seed`, epoch
#' `e` uses seed `seed + e`, so every epoch differs but runs are
#' reproducible.
#'
#' @param g A `graph_data` object.
#' @param nodes Seed nodes; default the training mask (or all nodes).
#' @param batch_size Seeds or graphs per batch.
#' @param fanouts Fanout per hop, see [sample_neighbors()].
#' @param shuffle Shuffle before splitting into batches.
#' @param seed Optional integer seed.
#' @return A loader object; use [loader_batches()] to get batches.
#' @examples
#' g <- dataset_karate()
#' ld <- node_loader(g, nodes = 1:20, batch_size = 8, fanouts = c(5, 5), seed = 1)
#' length(loader_batches(ld, epoch = 1))
#' @export
node_loader <- function(g, nodes = NULL, batch_size = 64L, fanouts = c(10L, 10L),
                        shuffle = TRUE, seed = NULL) {
  nodes <- nodes %||% if (!is.null(g$train_mask)) which(g$train_mask) else seq_len(g$num_nodes)
  structure(list(graph = g, nodes = as.integer(nodes),
                 batch_size = check_count(batch_size, "batch_size", 1L),
                 fanouts = as.integer(fanouts), shuffle = shuffle, seed = seed),
            class = c("node_loader", "graph_loader_base"))
}

#' @rdname node_loader
#' @param dataset A list of `graph_data` objects or a `graph_dataset`.
#' @export
graph_loader <- function(dataset, batch_size = 32L, shuffle = TRUE, seed = NULL) {
  structure(list(dataset = dataset,
                 batch_size = check_count(batch_size, "batch_size", 1L),
                 shuffle = shuffle, seed = seed),
            class = c("graph_loader", "graph_loader_base"))
}

#' @rdname node_loader
#' @param loader A loader.
#' @param epoch Epoch number, used to derive the epoch seed.
#' @export
loader_batches <- function(loader, epoch = 1L) UseMethod("loader_batches")

batch_split <- function(n, size, shuffle, seed) {
  idx <- with_optional_seed(seed, if (shuffle) sample.int(n) else seq_len(n))
  split(idx, ceiling(seq_along(idx) / size))
}

#' @export
loader_batches.node_loader <- function(loader, epoch = 1L) {
  s <- if (is.null(loader$seed)) NULL else loader$seed + epoch
  parts <- batch_split(length(loader$nodes), loader$batch_size, loader$shuffle, s)
  lapply(seq_along(parts), function(i) {
    sample_neighbors(loader$graph, loader$nodes[parts[[i]]], loader$fanouts,
                     seed = if (is.null(s)) NULL else s * 1000L + i)
  })
}

#' @export
loader_batches.graph_loader <- function(loader, epoch = 1L) {
  s <- if (is.null(loader$seed)) NULL else loader$seed + epoch
  n <- length(loader$dataset)
  parts <- batch_split(n, loader$batch_size, loader$shuffle, s)
  lapply(parts, function(ix) graph_batch(lapply(ix, function(i) loader$dataset[[i]])))
}

#' @export
length.graph_loader_base <- function(x) {
  n <- if (inherits(x, "node_loader")) length(x$nodes) else length(x$dataset)
  as.integer(ceiling(n / x$batch_size))
}

#' @export
print.graph_loader_base <- function(x, ...) {
  if (inherits(x, "node_loader")) {
    cat(sprintf("<node_loader> %d seed nodes in %d batches of up to %d, fanouts %s\n",
                length(x$nodes), length(x), x$batch_size, paste(x$fanouts, collapse = ", ")))
  } else {
    cat(sprintf("<graph_loader> %d graphs in %d batches of up to %d\n",
                length(x$dataset), length(x), x$batch_size))
  }
  invisible(x)
}

#' Negative edge sampling
#'
#' Draws node pairs that are not edges. Pairs are drawn uniformly from all
#' allowed pairs by rejection: candidate pairs are sampled uniformly, and
#' pairs that appear in `exclude`, self loops (unless allowed) and repeats are
#' discarded. For undirected graphs `(u, v)` and `(v, u)` are the same pair,
#' each returned pair has `u < v`, and a pair is excluded if either
#' orientation is in `exclude`.
#'
#' For bipartite or typed relations give `num_src` and `num_dst`; pairs are
#' then (source of type A, target of type B) and self loops are not special.
#'
#' @param exclude 2 by E matrix of known positive pairs to avoid (all known
#'   positives, including validation and test edges, for the usual filtered
#'   protocol).
#' @param num_nodes Number of nodes (homogeneous graphs).
#' @param num_samples Number of negative pairs.
#' @param directed Treat pairs as ordered.
#' @param allow_self_loops Allow `(u, u)` pairs.
#' @param num_src,num_dst Node counts for bipartite sampling.
#' @param seed Optional integer seed (see [sample_neighbors()]).
#' @param max_tries Rejection rounds before giving up.
#' @return A 2 by `num_samples` integer matrix.
#' @examples
#' negative_sampling(rbind(c(1, 2), c(2, 3)), num_nodes = 4, num_samples = 3,
#'                   seed = 1)
#' @export
negative_sampling <- function(exclude, num_nodes = NULL, num_samples,
                              directed = TRUE, allow_self_loops = FALSE,
                              num_src = NULL, num_dst = NULL, seed = NULL,
                              max_tries = 100L) {
  exclude <- normalize_edge_index(exclude)
  bip <- !is.null(num_src) || !is.null(num_dst)
  ns <- num_src %||% num_nodes
  nd <- num_dst %||% num_nodes
  if (is.null(ns) || is.null(nd)) stopf("Give `num_nodes`, or `num_src` and `num_dst`.")
  if (bip) {
    directed <- TRUE
    allow_self_loops <- TRUE
  }
  key <- function(u, v) (as.double(u) - 1) * nd + v
  known <- key(exclude[1L, ], exclude[2L, ])
  if (!directed) known <- c(known, key(exclude[2L, ], exclude[1L, ]))
  total <- as.double(ns) * nd
  if (!allow_self_loops) total <- total - min(ns, nd)
  if (!directed) total <- (total - if (allow_self_loops) min(ns, nd) else 0) / 2 +
    if (allow_self_loops) min(ns, nd) else 0
  ex <- exclude
  if (!allow_self_loops) ex <- ex[, ex[1L, ] != ex[2L, ], drop = FALSE]
  ck <- if (directed) key(ex[1L, ], ex[2L, ]) else key(pmin(ex[1L, ], ex[2L, ]), pmax(ex[1L, ], ex[2L, ]))
  avail <- total - length(unique(ck))
  if (num_samples > max(avail, 0)) {
    stopf("Only about %.0f candidate negative pairs exist; asked for %d.", avail, num_samples)
  }
  with_optional_seed(seed, {
    got_u <- integer()
    got_v <- integer()
    seen <- known
    for (t in seq_len(max_tries)) {
      need <- num_samples - length(got_u)
      if (need <= 0L) break
      m <- ceiling(need * 1.5) + 10L
      u <- sample.int(ns, m, replace = TRUE)
      v <- sample.int(nd, m, replace = TRUE)
      if (!directed) {
        lo <- pmin(u, v)
        v <- pmax(u, v)
        u <- lo
      }
      ok <- if (allow_self_loops) rep(TRUE, m) else u != v
      k <- key(u, v)
      ok <- ok & !(k %in% seen) & !duplicated(k)
      u <- u[ok][seq_len(min(sum(ok), need))]
      v <- v[ok][seq_len(min(sum(ok), need))]
      seen <- c(seen, key(u, v))
      got_u <- c(got_u, u)
      got_v <- c(got_v, v)
    }
    if (length(got_u) < num_samples) {
      stopf("Negative sampling found only %d of %d pairs; the graph may be too dense.",
            length(got_u), num_samples)
    }
    rbind(got_u, got_v, deparse.level = 0L)
  })
}

#' Corrupted tails for ranking evaluation
#'
#' For each positive pair `(u, v)`, draws `k` candidate targets `v'` such
#' that `(u, v')` is not a known positive (filtered setting, Bordes et al.
#' 2013) unless `filtered = FALSE`. Used for mean reciprocal rank and
#' per-query Hits@K.
#'
#' @param pos 2 by K matrix of positive pairs.
#' @param num_nodes Number of candidate targets.
#' @param k Negatives per positive.
#' @param known 2 by E matrix of all known positives (filtered setting).
#' @param filtered Exclude known positives.
#' @param seed Optional integer seed.
#' @return A K by `k` integer matrix of negative targets.
#' @export
corrupt_tails <- function(pos, num_nodes, k, known = pos, filtered = TRUE, seed = NULL) {
  pos <- normalize_edge_index(pos)
  known <- normalize_edge_index(known)
  kk <- (as.double(known[1L, ]) - 1) * num_nodes + known[2L, ]
  with_optional_seed(seed, {
    out <- matrix(0L, ncol(pos), k)
    for (i in seq_len(ncol(pos))) {
      u <- pos[1L, i]
      bad <- if (filtered) kk[kk > (u - 1) * num_nodes & kk <= u * num_nodes] - (u - 1) * num_nodes else pos[2L, i]
      cand <- setdiff(seq_len(num_nodes), c(bad, pos[2L, i]))
      if (length(cand) < k) stopf("Too few candidate tails for positive %d.", i)
      out[i, ] <- cand[sample.int(length(cand), k)]
    }
    out
  })
}
