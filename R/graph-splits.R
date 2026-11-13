#' Split nodes into training, validation and test sets
#'
#' Sets `train_mask`, `val_mask` and `test_mask` on a graph for transductive
#' node prediction. Proportions or counts are accepted. With `stratify`, each
#' class is split separately so class proportions are kept. Nodes with a
#' missing label are never assigned.
#'
#' Transductive splits share one graph: the features and edges of test nodes
#' are visible during training, only their labels are held out. This is the
#' standard protocol for citation benchmarks (Kipf and Welling 2017); for
#' inductive evaluation split graphs with [split_graphs()] instead.
#'
#' @param g A `graph_data` object with node labels `y`.
#' @param train,val,test Proportions (summing to at most 1) or counts.
#' @param stratify Split each class separately (factor labels only).
#' @param seed Optional integer seed; the caller's random state is not
#'   changed.
#' @return `g` with the three masks set.
#' @examples
#' g <- split_nodes(dataset_karate(), train = 0.5, val = 0.25, test = 0.25,
#'                  seed = 1)
#' sum(g$train_mask)
#' @export
split_nodes <- function(g, train = 0.6, val = 0.2, test = 0.2,
                        stratify = is.factor(g$y), seed = NULL) {
  n <- g$num_nodes
  lab <- if (is.null(g$y)) rep(TRUE, n) else
    if (is.matrix(g$y)) stats::complete.cases(g$y) else !is.na(g$y)
  pool <- which(lab)
  groups <- if (stratify && is.factor(g$y)) split(pool, droplevels(g$y[pool])) else list(pool)
  with_optional_seed(seed, {
    assign <- integer(n)
    for (grp in groups) {
      m <- length(grp)
      sizes <- split_sizes(m, train, val, test)
      perm <- grp[sample.int(m)]
      code <- rep(c(1L, 2L, 3L, 0L), c(sizes, m - sum(sizes)))
      assign[perm] <- code
    }
  })
  g$train_mask <- assign == 1L
  g$val_mask <- assign == 2L
  g$test_mask <- assign == 3L
  g
}

split_sizes <- function(m, train, val, test) {
  p <- c(train, val, test)
  if (any(p < 0)) stopf("Split sizes must be non-negative.")
  if (all(p <= 1) && sum(p) <= 1 + 1e-8 && any(p < 1 & p > 0)) {
    s <- floor(p * m)
    rest <- min(m, round(sum(p) * m)) - sum(s)
    if (rest > 0) {
      frac <- p * m - s
      add <- order(-frac)[seq_len(rest)]
      s[add] <- s[add] + 1L
    }
    return(as.integer(s))
  }
  if (sum(p) > m) stopf("Requested %d items but only %d are available.", sum(p), m)
  as.integer(p)
}

#' Split edges for link prediction
#'
#' Holds out edges for validation and test without leaking them into
#' training:
#'
#' 1. Undirected pairs are split as units, so `(u, v)` and `(v, u)` always
#'    land in the same set.
#' 2. `train_graph` keeps all nodes and features but only training edges;
#'    use it for message passing during training **and** evaluation.
#' 3. Validation and test negatives are drawn once with
#'    [negative_sampling()], excluding every positive edge of the full graph.
#'    Training negatives should be redrawn each epoch, excluding training
#'    positives only, which [train_graph_model()] does.
#'
#' Positive pairs are returned in one direction (as given in the input).
#'
#' @param g A `graph_data` object.
#' @param val,test Proportions (or counts) of pairs held out.
#' @param neg_ratio Negatives per positive for validation and test.
#' @param seed Optional integer seed.
#' @return An object of class `link_split` with `train_graph`, `train_pos`,
#'   `val_pos`, `val_neg`, `test_pos`, `test_neg` (2 by K integer matrices),
#'   `all_pos` and settings.
#' @examples
#' sp <- split_edges(dataset_karate(), val = 0.1, test = 0.1, seed = 1)
#' sp
#' @export
split_edges <- function(g, val = 0.1, test = 0.1, neg_ratio = 1, seed = NULL) {
  cols <- input_edge_columns(g)
  pairs <- g$edge_index[, cols, drop = FALSE]
  m <- ncol(pairs)
  sizes <- split_sizes(m, 1 - val - test, val, test)
  if (val < 1 && test < 1) sizes[1L] <- m - sizes[2L] - sizes[3L]
  with_optional_seed(seed, {
    perm <- sample.int(m)
    code <- rep(1:3, sizes)
    which_set <- integer(m)
    which_set[perm] <- code
    pick <- function(k) pairs[, which_set == k, drop = FALSE]
    val_pos <- pick(2L)
    test_pos <- pick(3L)
    neg_seed <- if (is.null(seed)) NULL else seed + 1L
    neg <- negative_sampling(pairs, num_nodes = g$num_nodes,
                             num_samples = ceiling(neg_ratio * (ncol(val_pos) + ncol(test_pos))),
                             directed = g$directed, seed = neg_seed)
    nv <- ceiling(neg_ratio * ncol(val_pos))
  })
  keep_pairs <- which(which_set == 1L)
  stored <- which(g$edge_pair %in% g$edge_pair[cols[keep_pairs]])
  train_graph <- edge_subset(g, stored)
  structure(list(
    train_graph = train_graph,
    train_pos = pairs[, keep_pairs, drop = FALSE],
    val_pos = val_pos, val_neg = neg[, seq_len(nv), drop = FALSE],
    test_pos = test_pos, test_neg = neg[, nv + seq_len(ncol(neg) - nv), drop = FALSE],
    all_pos = pairs, directed = g$directed, num_nodes = g$num_nodes,
    seed = seed
  ), class = "link_split")
}

#' @export
print.link_split <- function(x, ...) {
  cat(sprintf("<link_split> %d nodes, %s pairs: train %d, val %d (+%d neg), test %d (+%d neg)\n",
              x$num_nodes, if (x$directed) "directed" else "undirected",
              ncol(x$train_pos), ncol(x$val_pos), ncol(x$val_neg),
              ncol(x$test_pos), ncol(x$test_neg)))
  invisible(x)
}

# Keep only the given stored edges (by column), all nodes.
edge_subset <- function(g, cols) {
  out <- g
  out$edge_index <- g$edge_index[, cols, drop = FALSE]
  for (f in c("edge_attr", "edge_weight", "edge_y", "edge_type")) {
    out[[f]] <- take_rows(g[[f]], cols)
  }
  for (f in c("edge_train_mask", "edge_val_mask", "edge_test_mask")) {
    out[[f]] <- take_rows(g[[f]], cols)
  }
  out$edge_pair <- match(g$edge_pair[cols], unique(g$edge_pair[cols]))
  out
}

#' Split labeled edges for edge classification or regression
#'
#' Sets logical masks `edge_train_mask`, `edge_val_mask` and
#' `edge_test_mask` over stored edges. Both directions of an undirected pair
#' get the same assignment. Only edges with a non missing `edge_y` are
#' assigned. The graph structure stays intact: edge labels, not edges, are
#' held out.
#'
#' @inheritParams split_nodes
#' @return `g` with the edge masks set.
#' @export
split_edge_labels <- function(g, train = 0.6, val = 0.2, test = 0.2, seed = NULL) {
  if (is.null(g$edge_y)) stopf("The graph has no `edge_y` labels.")
  cols <- input_edge_columns(g)
  ok <- !is.na(take_rows(g$edge_y, cols))
  if (is.matrix(ok)) ok <- rowSums(!ok) == 0L
  pool <- cols[ok]
  sizes <- split_sizes(length(pool), train, val, test)
  with_optional_seed(seed, {
    perm <- pool[sample.int(length(pool))]
  })
  code <- integer(ncol(g$edge_index))
  pair_code <- integer(max(g$edge_pair, 0L))
  pair_code[g$edge_pair[perm]] <- rep(c(1L, 2L, 3L, 0L), c(sizes, length(pool) - sum(sizes)))
  code <- pair_code[g$edge_pair]
  g$edge_train_mask <- code == 1L
  g$edge_val_mask <- code == 2L
  g$edge_test_mask <- code == 3L
  g
}

#' Split graphs for graph level prediction
#'
#' Returns indices of training, validation and test graphs.
#'
#' * `"random"`: random split, stratified by `labels` when given.
#' * `"group"`: group disjoint split; all graphs of a group go to the same
#'   set (for example several conformers of one molecule, or graphs from the
#'   same patient). Groups are shuffled, then assigned in order until each
#'   set reaches its target size.
#' * `"scaffold"`: deterministic scaffold split as in MoleculeNet (Wu et al.
#'   2018): `groups` holds a scaffold identifier per molecule (for example a
#'   Bemis-Murcko scaffold computed with RDKit); scaffold sets are sorted
#'   from largest to smallest and assigned to training first, then
#'   validation, then test, so test molecules have scaffolds not seen in
#'   training. graphlearnR does not compute scaffolds itself.
#'
#' @param n Number of graphs, or the dataset (its length is used).
#' @param train,val,test Proportions.
#' @param method Split method.
#' @param labels Optional labels for stratified random splits.
#' @param groups Group or scaffold identifiers, one per graph.
#' @param seed Optional integer seed.
#' @return A list with integer vectors `train`, `val`, `test`.
#' @examples
#' split_graphs(10, seed = 1)
#' split_graphs(6, method = "scaffold", groups = c("a", "a", "a", "b", "b", "c"))
#' @export
split_graphs <- function(n, train = 0.8, val = 0.1, test = 0.1,
                         method = c("random", "group", "scaffold"),
                         labels = NULL, groups = NULL, seed = NULL) {
  method <- match.arg(method)
  if (!is.numeric(n) || length(n) != 1L) n <- length(n)
  if (method != "random") {
    if (is.null(groups) || length(groups) != n) {
      stopf("`groups` must give one group per graph for method \"%s\".", method)
    }
    grp <- split(seq_len(n), groups)
    if (method == "scaffold") {
      sz <- vapply(grp, length, integer(1))
      grp <- grp[order(-sz, names(grp))]
    } else {
      grp <- with_optional_seed(seed, grp[sample.int(length(grp))])
    }
    target <- cumsum(c(train, val)) * n
    out <- list(train = integer(), val = integer(), test = integer())
    for (gi in grp) {
      k <- if (length(out$train) + length(gi) <= target[1L]) "train"
           else if (length(out$train) + length(out$val) + length(gi) <= target[2L]) "val"
           else "test"
      out[[k]] <- c(out[[k]], gi)
    }
    return(out)
  }
  pools <- if (is.null(labels)) list(seq_len(n)) else split(seq_len(n), labels)
  with_optional_seed(seed, {
    out <- list(train = integer(), val = integer(), test = integer())
    for (p in pools) {
      s <- split_sizes(length(p), train, val, test)
      perm <- p[sample.int(length(p))]
      out$train <- c(out$train, perm[seq_len(s[1L])])
      out$val <- c(out$val, perm[s[1L] + seq_len(s[2L])])
      out$test <- c(out$test, perm[s[1L] + s[2L] + seq_len(s[3L])])
    }
    out
  })
}
