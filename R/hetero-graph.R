#' Heterogeneous graph data
#'
#' Stores a graph with several node types and several relation types. Each
#' node type has its own feature matrix (feature sizes may differ), labels
#' and masks; each relation is identified by an explicit triple
#' `(source type, relation, target type)` and stores its edges as a 2 by E
#' matrix whose first row indexes nodes **of the source type** and whose
#' second row indexes nodes **of the target type**, both one based.
#'
#' Relations are directed. To let information flow both ways, add reverse
#' relations with [add_reverse_relations()]; the pairing between a relation
#' and its reverse is recorded so that link prediction splits remove held out
#' edges from both ([split_edges_hetero()]).
#'
#' @param nodes Named list, one entry per node type: a feature matrix, a
#'   node count, or a list with any of `x`, `y`, `num_nodes`, `node_ids`,
#'   `train_mask`, `val_mask`, `test_mask`.
#' @param edges Named list, one entry per relation. Name an entry
#'   `"src__relation__dst"` and give a 2 by E (or E by 2) matrix, or give a
#'   list with `edge_index` and optional `src`, `rel`, `dst`, `edge_attr`,
#'   `edge_weight`, `edge_y`, `edge_time`. `edge_time` is a relative time
#'   per edge (for example the target's year minus the source's), used by
#'   the relative temporal encoding of [hgt_conv()]; reverse relations get
#'   its negative.
#' @param metadata Named list of free metadata.
#' @param validate Run [validate_hetero_graph()].
#' @return An object of class `hetero_graph_data`.
#' @examples
#' hg <- hetero_graph_data(
#'   nodes = list(drug = matrix(rnorm(6), 3, 2), protein = matrix(rnorm(12), 4, 3)),
#'   edges = list(drug__targets__protein = rbind(c(1, 2, 3), c(1, 1, 4)))
#' )
#' hg
#' hg <- add_reverse_relations(hg)
#' names(hg$edges)
#' @export
hetero_graph_data <- function(nodes, edges, metadata = list(), validate = TRUE) {
  if (!is.list(nodes) || is.null(names(nodes)) || any(names(nodes) == "")) {
    stopf("`nodes` must be a named list, one entry per node type.")
  }
  if (any(grepl("__", names(nodes), fixed = TRUE))) {
    stopf("Node type names must not contain \"__\".")
  }
  nl <- lapply(nodes, function(v) {
    if (is.list(v) && !is.data.frame(v)) {
      v$x <- as_feature_matrix(v$x, "x")
      v
    } else if (is.numeric(v) && length(v) == 1L) {
      list(num_nodes = as.integer(v))
    } else {
      list(x = as_feature_matrix(v, "x"))
    }
  })
  el <- list()
  for (i in seq_along(edges)) {
    nm <- names(edges)[i] %||% ""
    v <- edges[[i]]
    if (!(is.list(v) && !is.data.frame(v))) v <- list(edge_index = v)
    parts <- strsplit(nm, "__", fixed = TRUE)[[1L]]
    src <- v$src %||% if (length(parts) == 3L) parts[1L]
    rel <- v$rel %||% if (length(parts) == 3L) parts[2L]
    dst <- v$dst %||% if (length(parts) == 3L) parts[3L]
    if (is.null(src) || is.null(rel) || is.null(dst)) {
      stopf("Name relation %d \"src__relation__dst\" or give `src`, `rel` and `dst`.", i)
    }
    if (grepl("__", rel, fixed = TRUE)) {
      stopf("Relation names must not contain \"__\" (relation %d: `%s`).", i, rel)
    }
    key <- paste(src, rel, dst, sep = "__")
    el[[key]] <- list(src = src, rel = rel, dst = dst,
                      edge_index = normalize_edge_index(v$edge_index),
                      edge_attr = as_feature_matrix(v$edge_attr, "edge_attr"),
                      edge_weight = if (!is.null(v$edge_weight)) as.double(v$edge_weight),
                      edge_y = v$edge_y,
                      edge_time = if (!is.null(v$edge_time)) as.double(v$edge_time),
                      reverse = v$reverse, reverse_of = v$reverse_of)
  }
  for (t in names(nl)) {
    nd <- nl[[t]]
    n <- nd$num_nodes %||% n_rows(nd$x)
    if (is.na(n)) n <- n_rows(nd$y)
    if (is.na(n)) {
      idx <- unlist(lapply(el, function(e) c(if (e$src == t) e$edge_index[1L, ],
                                             if (e$dst == t) e$edge_index[2L, ])))
      n <- if (length(idx)) max(idx) else 0L
    }
    nd$num_nodes <- as.integer(n)
    nd$node_ids <- nd$node_ids %||% seq_len(n)
    nl[[t]] <- nd
  }
  hg <- structure(list(nodes = nl, edges = el, node_types = names(nl),
                       metadata = metadata), class = "hetero_graph_data")
  if (validate) validate_hetero_graph(hg)
  hg
}

#' @rdname hetero_graph_data
#' @param hg A `hetero_graph_data` object.
#' @export
validate_hetero_graph <- function(hg) {
  problems <- character()
  add <- function(...) problems <<- c(problems, sprintf(...))
  for (t in hg$node_types) {
    nd <- hg$nodes[[t]]
    n <- nd$num_nodes
    for (f in c("x", "y", "node_ids")) {
      if (!is.null(nd[[f]]) && n_rows(nd[[f]]) != n) {
        add("Node type `%s`: `%s` has %d entries for %d nodes.", t, f, n_rows(nd[[f]]), n)
      }
    }
    for (m in c("train_mask", "val_mask", "test_mask")) {
      if (!is.null(nd[[m]]) && (!is.logical(nd[[m]]) || length(nd[[m]]) != n ||
                                anyNA(nd[[m]]))) {
        add("Node type `%s`: `%s` must be logical of length %d without NA.", t, m, n)
      }
    }
    for (msg in mask_overlaps(nd, c("train_mask", "val_mask", "test_mask"),
                              sprintf("Node type `%s`: ", t))) add("%s", msg)
  }
  for (key in names(hg$edges)) {
    e <- hg$edges[[key]]
    if (!e$src %in% hg$node_types) add("Relation `%s`: unknown source type `%s`.", key, e$src)
    if (!e$dst %in% hg$node_types) add("Relation `%s`: unknown target type `%s`.", key, e$dst)
    if (!e$src %in% hg$node_types || !e$dst %in% hg$node_types) next
    ns <- hg$nodes[[e$src]]$num_nodes
    nd <- hg$nodes[[e$dst]]$num_nodes
    ei <- e$edge_index
    if (ncol(ei)) {
      if (any(ei[1L, ] < 1L | ei[1L, ] > ns)) {
        add("Relation `%s`: source indices must be in 1:%d (type `%s`).", key, ns, e$src)
      }
      if (any(ei[2L, ] < 1L | ei[2L, ] > nd)) {
        add("Relation `%s`: target indices must be in 1:%d (type `%s`).", key, nd, e$dst)
      }
    }
    for (f in c("edge_attr", "edge_weight", "edge_y", "edge_time")) {
      if (!is.null(e[[f]]) && n_rows(e[[f]]) != ncol(ei)) {
        add("Relation `%s`: `%s` has %d entries for %d edges.", key, f, n_rows(e[[f]]), ncol(ei))
      }
    }
  }
  if (length(problems)) {
    stop(paste(c("Invalid hetero_graph_data:", paste0("* ", problems)), collapse = "\n"),
         call. = FALSE)
  }
  invisible(hg)
}

#' @export
num_nodes.hetero_graph_data <- function(g) {
  vapply(g$nodes, function(n) n$num_nodes, integer(1))
}

#' @export
num_edges.hetero_graph_data <- function(g, unique_pairs = FALSE) {
  vapply(g$edges, function(e) ncol(e$edge_index), integer(1))
}

#' Relation table of a heterogeneous graph
#'
#' @param hg A `hetero_graph_data` object.
#' @return A data frame with `key`, `src`, `rel`, `dst`, `num_edges` and
#'   `reverse` (the paired reverse relation, if any).
#' @export
edge_types <- function(hg) {
  data.frame(
    key = names(hg$edges),
    src = vapply(hg$edges, `[[`, "", "src"),
    rel = vapply(hg$edges, `[[`, "", "rel"),
    dst = vapply(hg$edges, `[[`, "", "dst"),
    num_edges = vapply(hg$edges, function(e) ncol(e$edge_index), integer(1)),
    reverse = vapply(hg$edges, function(e) e$reverse %||% e$reverse_of %||% NA_character_, ""),
    row.names = NULL
  )
}

#' @export
print.hetero_graph_data <- function(x, ...) {
  cat(sprintf("<hetero_graph_data> %d node types, %d relations\n",
              length(x$node_types), length(x$edges)))
  for (t in x$node_types) {
    nd <- x$nodes[[t]]
    cat(sprintf("  node %-14s %6d nodes%s%s\n", t, nd$num_nodes,
                if (!is.null(nd$x)) sprintf(", %d features", ncol(nd$x)) else "",
                if (!is.null(nd$y)) ", labeled" else ""))
  }
  for (k in names(x$edges)) {
    cat(sprintf("  edge %s: %d\n", k, ncol(x$edges[[k]]$edge_index)))
  }
  invisible(x)
}

#' Add reverse relations
#'
#' For every relation `src__rel__dst` without a reverse, adds
#' `dst__<prefix>rel__src` with flipped edges and the same edge attributes,
#' and records the pairing in `reverse` and `reverse_of`.
#'
#' @param hg A `hetero_graph_data` object.
#' @param prefix Prefix for the reverse relation name.
#' @return The extended `hetero_graph_data`.
#' @export
add_reverse_relations <- function(hg, prefix = "rev_") {
  for (key in names(hg$edges)) {
    e <- hg$edges[[key]]
    if (!is.null(e$reverse) || !is.null(e$reverse_of)) next
    rkey <- paste(e$dst, paste0(prefix, e$rel), e$src, sep = "__")
    if (rkey %in% names(hg$edges)) next
    hg$edges[[rkey]] <- list(src = e$dst, rel = paste0(prefix, e$rel), dst = e$src,
                             edge_index = e$edge_index[2:1, , drop = FALSE],
                             edge_attr = e$edge_attr, edge_weight = e$edge_weight,
                             edge_y = NULL, edge_time = if (!is.null(e$edge_time)) -e$edge_time,
                             reverse = NULL, reverse_of = key)
    hg$edges[[key]]$reverse <- rkey
  }
  hg
}

#' Convert a heterogeneous graph to a homogeneous one
#'
#' Concatenates node types in the order of `hg$node_types` and relations in
#' the order of `names(hg$edges)`. `node_type` and `edge_type` become factors
#' with these levels; `metadata$type_ptr` gives node offsets, so node `i` of
#' type `t` becomes global node `type_ptr[t] + i`. Feature matrices of
#' different sizes are padded with zeros to the largest size, and
#' `metadata$feature_dims` records the original sizes. Labels and masks come
#' from `target_type` only (other nodes get `NA` labels and `FALSE` masks).
#'
#' @param hg A `hetero_graph_data` object.
#' @param target_type Node type whose labels and masks are kept.
#' @return A directed `graph_data` object.
#' @export
to_homogeneous <- function(hg, target_type = NULL) {
  types <- hg$node_types
  n <- num_nodes(hg)
  ptr <- stats::setNames(c(0L, cumsum(n))[seq_along(types)], types)
  dims <- vapply(hg$nodes, function(nd) if (is.null(nd$x)) 0L else ncol(nd$x), integer(1))
  x <- if (any(dims > 0L)) {
    do.call(rbind, lapply(types, function(t) {
      m <- matrix(0, n[[t]], max(dims))
      if (dims[[t]] > 0L) m[, seq_len(dims[[t]])] <- hg$nodes[[t]]$x
      m
    }))
  }
  keys <- names(hg$edges)
  ei <- do.call(cbind, c(list(matrix(integer(), 2L, 0L)), lapply(keys, function(k) {
    e <- hg$edges[[k]]
    rbind(e$edge_index[1L, ] + ptr[[e$src]], e$edge_index[2L, ] + ptr[[e$dst]])
  })))
  etype <- factor(rep(keys, vapply(hg$edges, function(e) ncol(e$edge_index), integer(1))),
                  levels = keys)
  pick_target <- function(f, fill) {
    if (is.null(target_type)) return(NULL)
    v <- hg$nodes[[target_type]][[f]]
    if (is.null(v)) return(NULL)
    out <- rep(fill, sum(n))
    if (is.factor(v)) out <- factor(out, levels = levels(v))
    out[ptr[[target_type]] + seq_len(n[[target_type]])] <- v
    out
  }
  has_w <- all(vapply(hg$edges, function(e) !is.null(e$edge_weight), logical(1)))
  new_graph_data(
    x = x, edge_index = ei, y = pick_target("y", NA),
    edge_weight = if (has_w && length(keys)) unlist(lapply(hg$edges, `[[`, "edge_weight"), use.names = FALSE),
    num_nodes = sum(n), directed = TRUE,
    node_ids = seq_len(sum(n)),
    node_type = factor(rep(types, n), levels = types), edge_type = etype,
    train_mask = pick_target("train_mask", FALSE),
    val_mask = pick_target("val_mask", FALSE),
    test_mask = pick_target("test_mask", FALSE),
    metadata = c(hg$metadata, list(type_ptr = ptr, feature_dims = dims,
                                   node_type_levels = types, edge_type_levels = keys))
  )
}

#' Heterogeneous graph tensors
#'
#' Converts every node type and relation of a heterogeneous graph to `torch`
#' tensors, once, on one device.
#'
#' @param hg A `hetero_graph_data` object.
#' @param device Device, see [graph_tensors()].
#' @return A list with named lists `x`, `y`, `edge_index`, `edge_attr`,
#'   `edge_weight` and the masks, plus `num_nodes`, `node_types` and
#'   `relations` (the result of [edge_types()]).
#' @export
hetero_tensors <- function(hg, device = NULL) {
  dev <- resolve_device(device)
  ft <- function(v) if (!is.null(v)) torch_tensor(v, dtype = torch_float(), device = dev)
  lab <- function(v) {
    if (is.null(v)) return(NULL)
    if (is.factor(v) || (is.integer(v) && !is.matrix(v))) {
      torch_tensor(as.integer(v), dtype = torch_long(), device = dev)
    } else {
      if (!is.double(v)) storage.mode(v) <- "double"
      ft(v)
    }
  }
  mk <- function(v) if (!is.null(v)) torch_tensor(v, dtype = torch_bool(), device = dev)
  ei <- function(m) if (ncol(m)) torch_tensor(m, dtype = torch_long(), device = dev)
                    else torch_empty(2L, 0L, dtype = torch_long(), device = dev)
  out <- list(
    x = lapply(hg$nodes, function(n) ft(n$x)),
    y = lapply(hg$nodes, function(n) lab(n$y)),
    train_mask = lapply(hg$nodes, function(n) mk(n$train_mask)),
    val_mask = lapply(hg$nodes, function(n) mk(n$val_mask)),
    test_mask = lapply(hg$nodes, function(n) mk(n$test_mask)),
    edge_index = lapply(hg$edges, function(e) ei(e$edge_index)),
    edge_attr = lapply(hg$edges, function(e) ft(e$edge_attr)),
    edge_weight = lapply(hg$edges, function(e) ft(e$edge_weight)),
    edge_time = lapply(hg$edges, function(e) ft(e$edge_time)),
    node_index = lapply(stats::setNames(hg$node_types, hg$node_types), function(t) {
      if (!is.null(hg$n_id[[t]])) torch_tensor(as.integer(hg$n_id[[t]]), dtype = torch_long(), device = dev)
    }),
    num_nodes = num_nodes(hg), node_types = hg$node_types,
    relations = edge_types(hg), device = dev
  )
  structure(out, class = "hetero_tensors")
}

#' Batch heterogeneous graphs
#'
#' Concatenates graphs with the same node and relation types. Within each
#' node type, indices of graph `k` are shifted by the node counts of that
#' type in graphs `1, ..., k - 1`; `batch[[type]]` records graph membership.
#'
#' @param graphs A list of `hetero_graph_data` objects.
#' @return A `hetero_graph_data` of class `hetero_graph_batch`.
#' @export
hetero_batch <- function(graphs) {
  types <- graphs[[1L]]$node_types
  keys <- names(graphs[[1L]]$edges)
  for (g in graphs) {
    if (!identical(g$node_types, types) || !identical(names(g$edges), keys)) {
      stopf("All graphs must have the same node types and relations, in the same order.")
    }
  }
  counts <- sapply(graphs, num_nodes)
  counts <- matrix(counts, nrow = length(types), dimnames = list(types, NULL))
  ptr <- t(apply(counts, 1L, function(v) c(0L, cumsum(v))[seq_along(v)]))
  if (length(graphs) == 1L) ptr <- matrix(0L, length(types), 1L, dimnames = list(types, NULL))
  nodes <- lapply(stats::setNames(types, types), function(t) {
    pk <- function(f) bind_rows_values(lapply(graphs, function(g) g$nodes[[t]][[f]]))
    list(x = pk("x"), y = pk("y"), num_nodes = sum(counts[t, ]),
         train_mask = pk("train_mask"), val_mask = pk("val_mask"), test_mask = pk("test_mask"))
  })
  # Original node indices per type (for featureless types, these select each
  # node's embedding row), and the external identifiers of every graph.
  n_id <- lapply(stats::setNames(types, types), function(t) {
    unlist(lapply(graphs, function(g) g$n_id[[t]] %||% seq_len(g$nodes[[t]]$num_nodes)),
           use.names = FALSE)
  })
  original_node_ids <- lapply(stats::setNames(types, types), function(t) {
    lapply(graphs, function(g) g$nodes[[t]]$node_ids)
  })
  edges <- lapply(stats::setNames(keys, keys), function(k) {
    e1 <- graphs[[1L]]$edges[[k]]
    ei <- do.call(cbind, lapply(seq_along(graphs), function(j) {
      m <- graphs[[j]]$edges[[k]]$edge_index
      rbind(m[1L, ] + ptr[e1$src, j], m[2L, ] + ptr[e1$dst, j])
    }))
    list(src = e1$src, rel = e1$rel, dst = e1$dst, edge_index = ei,
         edge_attr = bind_rows_values(lapply(graphs, function(g) g$edges[[k]]$edge_attr)),
         edge_weight = bind_rows_values(lapply(graphs, function(g) g$edges[[k]]$edge_weight)),
         edge_time = bind_rows_values(lapply(graphs, function(g) g$edges[[k]]$edge_time)),
         edge_y = bind_rows_values(lapply(graphs, function(g) g$edges[[k]]$edge_y)),
         reverse = e1$reverse, reverse_of = e1$reverse_of)
  })
  hb <- hetero_graph_data(nodes, edges, metadata = list(num_graphs = length(graphs)))
  hb$n_id <- n_id
  hb$original_node_ids <- original_node_ids
  hb$batch <- lapply(stats::setNames(types, types), function(t) rep.int(seq_along(graphs), counts[t, ]))
  hb$num_graphs <- length(graphs)
  class(hb) <- c("hetero_graph_batch", class(hb))
  hb
}

#' Typed neighbor sampling
#'
#' Heterogeneous version of [sample_neighbors()]. Starting from seed nodes of
#' one or more types, hop `k` samples up to `fanouts[k]` incoming edges **per
#' relation** for every frontier node of that relation's target type.
#' `fanouts` may also be a named list giving a vector per relation key.
#'
#' The result is a `hetero_graph_data` with local indices; `n_id[[type]]`
#' holds the original indices (seeds first) and `num_seeds[[type]]` the seed
#' counts.
#'
#' @param hg A `hetero_graph_data` object.
#' @param seeds Named list of seed indices per node type.
#' @param fanouts Integer vector (all relations) or named list per relation.
#' @param seed Optional integer seed.
#' @return A `hetero_graph_data` of class `sampled_hetero_graph`.
#' @export
sample_hetero_neighbors <- function(hg, seeds, fanouts, seed = NULL) {
  types <- hg$node_types
  keys <- names(hg$edges)
  nhops <- if (is.list(fanouts)) max(lengths(fanouts)) else length(fanouts)
  fan_for <- function(k, hop) {
    f <- if (is.list(fanouts)) fanouts[[k]] else fanouts
    if (is.null(f) || hop > length(f)) 0L else as.integer(f[hop])
  }
  csc <- lapply(hg$edges, function(e) {
    dst <- e$edge_index[2L, ]
    cnt <- tabulate(dst, nbins = hg$nodes[[e$dst]]$num_nodes)
    list(ord = order(dst, method = "radix"), ptr = c(0L, cumsum(cnt)), cnt = cnt)
  })
  with_optional_seed(seed, {
    local <- lapply(hg$nodes, function(nd) integer(nd$num_nodes))
    n_id <- lapply(stats::setNames(types, types), function(t) integer())
    frontier <- n_id
    for (t in names(seeds)) {
      s <- as.integer(seeds[[t]])
      local[[t]][s] <- seq_along(s)
      n_id[[t]] <- s
      frontier[[t]] <- s
    }
    e_keep <- lapply(hg$edges, function(e) integer())
    for (hop in seq_len(nhops)) {
      new_front <- lapply(stats::setNames(types, types), function(t) integer())
      for (k in keys) {
        e <- hg$edges[[k]]
        f <- fan_for(k, hop)
        if (f == 0L || !length(frontier[[e$dst]])) next
        eids <- sample_incoming(csc[[k]], frontier[[e$dst]], f, FALSE)
        e_keep[[k]] <- c(e_keep[[k]], eids)
        srcs <- e$edge_index[1L, eids]
        new <- unique(srcs[local[[e$src]][srcs] == 0L])
        if (length(new)) {
          local[[e$src]][new] <- length(n_id[[e$src]]) + seq_along(new)
          n_id[[e$src]] <- c(n_id[[e$src]], new)
          new_front[[e$src]] <- c(new_front[[e$src]], new)
        }
      }
      frontier <- new_front
    }
  })
  nodes <- lapply(stats::setNames(types, types), function(t) {
    nd <- hg$nodes[[t]]
    ids <- n_id[[t]]
    list(x = take_rows(nd$x, ids), y = take_rows(nd$y, ids), num_nodes = length(ids),
         node_ids = nd$node_ids[ids], train_mask = take_rows(nd$train_mask, ids),
         val_mask = take_rows(nd$val_mask, ids), test_mask = take_rows(nd$test_mask, ids))
  })
  edges <- lapply(stats::setNames(keys, keys), function(k) {
    e <- hg$edges[[k]]
    ids <- unique(e_keep[[k]])
    list(src = e$src, rel = e$rel, dst = e$dst,
         edge_index = if (length(ids)) rbind(local[[e$src]][e$edge_index[1L, ids]],
                                             local[[e$dst]][e$edge_index[2L, ids]])
                      else matrix(integer(), 2L, 0L),
         edge_attr = take_rows(e$edge_attr, ids), edge_weight = take_rows(e$edge_weight, ids),
         edge_time = take_rows(e$edge_time, ids), reverse = e$reverse, reverse_of = e$reverse_of)
  })
  out <- hetero_graph_data(nodes, edges, metadata = hg$metadata)
  out$n_id <- n_id
  out$num_seeds <- vapply(stats::setNames(types, types),
                          function(t) length(seeds[[t]] %||% integer()), integer(1))
  class(out) <- c("sampled_hetero_graph", class(out))
  out
}

#' Split one relation of a heterogeneous graph for link prediction
#'
#' Holds out edges of `relation` for validation and test. Held out edges are
#' removed from the training graph **and from the paired relation** (as
#' recorded by [add_reverse_relations()], whichever of the two keys is
#' split), so a model cannot read a held out link through its reverse edge.
#' Parallel copies of a pair are split as one unit. For a relation whose
#' source and target type are the same, `(u, v)` and `(v, u)` are one unit
#' and negatives are unordered pairs without self loops; otherwise negatives
#' are typed pairs (source type, target type). Negatives are drawn with
#' [negative_sampling()], excluding every known positive of the relation.
#'
#' @param hg A `hetero_graph_data` object.
#' @param relation Relation key, `"src__rel__dst"`.
#' @inheritParams split_edges
#' @return An object of class `hetero_link_split`.
#' @export
split_edges_hetero <- function(hg, relation, val = 0.1, test = 0.1, neg_ratio = 1,
                               seed = NULL) {
  if (!relation %in% names(hg$edges)) stopf("Unknown relation `%s`.", relation)
  e <- hg$edges[[relation]]
  same <- identical(e$src, e$dst)
  un <- pair_units(e$edge_index, symmetric = same)
  pairs <- e$edge_index[, un$first, drop = FALSE]
  m <- ncol(pairs)
  sizes <- edge_split_sizes(m, val, test)
  ns <- hg$nodes[[e$src]]$num_nodes
  nd <- hg$nodes[[e$dst]]$num_nodes
  with_optional_seed(seed, {
    code <- integer(m)
    code[sample.int(m)] <- rep(1:3, sizes)
    nneg <- ceiling(neg_ratio * (sizes[2L] + sizes[3L]))
    nseed <- if (is.null(seed)) NULL else seed + 1L
    neg <- if (same) {
      negative_sampling(pairs, num_nodes = ns, num_samples = nneg, directed = FALSE, seed = nseed)
    } else {
      negative_sampling(pairs, num_src = ns, num_dst = nd, num_samples = nneg, seed = nseed)
    }
  })
  nv <- ceiling(neg_ratio * sizes[2L])
  held_keys <- un$key[code > 1L]
  train_graph <- hg
  train_graph$edges[[relation]] <- subset_relation(
    e, which(!pair_keys(e$edge_index, same) %in% held_keys))
  # Remove held out links from every relation paired with this one, in either
  # direction of the pairing (`reverse` on the forward relation, `reverse_of`
  # on a generated reverse relation).
  partners <- unique(c(e$reverse, e$reverse_of, names(Filter(function(r) {
    identical(r$reverse, relation) || identical(r$reverse_of, relation)
  }, hg$edges))))
  for (rk in setdiff(intersect(partners, names(hg$edges)), relation)) {
    r <- hg$edges[[rk]]
    flipped <- r$edge_index[2:1, , drop = FALSE]
    keep <- which(!pair_keys(flipped, same) %in% held_keys)
    train_graph$edges[[rk]] <- subset_relation(r, keep)
  }
  structure(list(
    train_graph = train_graph, relation = relation, src_type = e$src, dst_type = e$dst,
    train_pos = pairs[, code == 1L, drop = FALSE],
    val_pos = pairs[, code == 2L, drop = FALSE], val_neg = neg[, seq_len(nv), drop = FALSE],
    test_pos = pairs[, code == 3L, drop = FALSE],
    test_neg = neg[, nv + seq_len(ncol(neg) - nv), drop = FALSE],
    all_pos = pairs, num_src = ns, num_dst = nd, same_type = same, seed = seed
  ), class = "hetero_link_split")
}

subset_relation <- function(e, cols) {
  e$edge_index <- e$edge_index[, cols, drop = FALSE]
  e$edge_attr <- take_rows(e$edge_attr, cols)
  e$edge_weight <- take_rows(e$edge_weight, cols)
  e$edge_y <- take_rows(e$edge_y, cols)
  e$edge_time <- take_rows(e$edge_time, cols)
  e
}

#' @export
print.hetero_link_split <- function(x, ...) {
  cat(sprintf("<hetero_link_split> %s: train %d, val %d (+%d neg), test %d (+%d neg)\n",
              x$relation, ncol(x$train_pos), ncol(x$val_pos), ncol(x$val_neg),
              ncol(x$test_pos), ncol(x$test_neg)))
  invisible(x)
}

#' Heterogeneous convolution wrapper
#'
#' Applies one bipartite layer per relation and sums (or averages) the
#' results arriving at each target node type, the scheme of PyG's
#' `HeteroConv`. Node types that receive no relation keep their input
#' features.
#'
#' @param convs Named list of layers keyed by relation key; each is called as
#'   `layer(list(x_src, x_dst), edge_index)`.
#' @param aggr `"sum"` or `"mean"` over relations.
#' @return An `nn_module` called as `module(x_dict, edge_index_dict)`.
#' @export
hetero_conv <- nn_module(
  "hetero_conv",
  initialize = function(convs, aggr = c("sum", "mean")) {
    self$aggr <- match.arg(aggr)
    self$keys <- names(convs)
    self$convs <- nn_module_dict(convs)
  },
  forward = function(x_dict, edge_index_dict) {
    out <- list()
    cnt <- list()
    for (k in self$keys) {
      parts <- strsplit(k, "__", fixed = TRUE)[[1L]]
      s <- parts[1L]
      d <- parts[3L]
      ei <- edge_index_dict[[k]]
      if (is.null(ei) || is.null(x_dict[[s]]) || is.null(x_dict[[d]])) next
      h <- self$convs[[k]](list(x_dict[[s]], x_dict[[d]]), ei)
      out[[d]] <- if (is.null(out[[d]])) h else out[[d]] + h
      cnt[[d]] <- (cnt[[d]] %||% 0L) + 1L
    }
    if (self$aggr == "mean") for (d in names(out)) out[[d]] <- out[[d]] / cnt[[d]]
    for (t in setdiff(names(x_dict), names(out))) out[[t]] <- x_dict[[t]]
    out
  }
)

#' Heterogeneous graph neural network
#'
#' A per type input layer maps every node type to `hidden_dim` (a linear
#' layer for types with features, a learned embedding table for types
#' without), followed by `num_layers` relation aware layers and an optional
#' head for `target_type`.
#'
#' * `conv = "rgcn"`: relational GCN (Schlichtkrull et al. 2018) on the
#'   homogeneous view; each relation has its own weight, neighbors are
#'   averaged per relation.
#' * `conv = "sage"`, `"graph_conv"`, `"gat"`: one bipartite layer per
#'   relation, combined with [hetero_conv()]. `"gat"` gives relation specific
#'   attention (a heterogeneous GAT in the sense of PyG's `to_hetero` applied
#'   to `GATConv`; it is not HAN or HGT).
#' * `conv = "hgt"`: heterogeneous graph transformer layers, see
#'   [hgt_conv()]; attention is normalized over all incoming relations of a
#'   node.
#'
#' Embedding tables make the model transductive for featureless types: it
#' can only predict for nodes seen during training.
#'
#' @param hg A `hetero_graph_data` object used for its schema (types,
#'   relations, feature sizes, node counts); its data are not stored.
#' @param hidden_dim Hidden size.
#' @param output_dim Output size of the head, or `NULL` for embeddings only.
#' @param target_type Node type that gets the head.
#' @param num_layers Number of relation aware layers.
#' @param conv Layer type.
#' @param heads Attention heads for `"gat"`.
#' @param dropout Dropout probability.
#' @param aggr Neighbor aggregation inside each relation.
#' @param relation_aggr Combination across relations, `"sum"` or `"mean"`.
#' @param num_bases Basis count for `"rgcn"`.
#' @param temporal_encoding For `"hgt"`, use the relative temporal encoding
#'   with the relations' `edge_time`.
#' @return An `nn_module` of class `hetero_gnn`, called as `model(ht)` with
#'   `ht` from [hetero_tensors()]; it returns a named list of per type
#'   outputs.
#' @export
hetero_gnn_model <- function(hg, hidden_dim = 64L, output_dim = NULL, target_type = NULL,
                             num_layers = 2L, conv = c("sage", "graph_conv", "gat", "rgcn", "hgt"),
                             heads = 2L, dropout = 0.5, aggr = "mean",
                             relation_aggr = "sum", num_bases = NULL, temporal_encoding = FALSE) {
  conv <- match.arg(conv)
  if (!is.null(output_dim) && is.null(target_type)) stopf("Give `target_type` with `output_dim`.")
  if (!is.null(target_type) && !target_type %in% hg$node_types) {
    stopf("Unknown target type `%s`.", target_type)
  }
  if (conv %in% c("gat", "hgt") && hidden_dim %% heads != 0L) stopf("`hidden_dim` must be divisible by `heads`.")
  schema <- list(
    node_types = hg$node_types,
    keys = names(hg$edges),
    feature_dims = vapply(hg$nodes, function(nd) if (is.null(nd$x)) 0L else ncol(nd$x), integer(1)),
    num_nodes = num_nodes(hg)
  )
  hetero_gnn(list(schema = schema, hidden_dim = as.integer(hidden_dim),
                  output_dim = output_dim, target_type = target_type,
                  num_layers = as.integer(num_layers), conv = conv, heads = as.integer(heads),
                  dropout = dropout, aggr = aggr, relation_aggr = relation_aggr,
                  num_bases = num_bases, temporal_encoding = temporal_encoding))
}

hetero_gnn <- nn_module(
  "hetero_gnn",
  initialize = function(config) {
    self$config <- config
    sc <- config$schema
    h <- config$hidden_dim
    self$input <- nn_module_dict(lapply(stats::setNames(sc$node_types, sc$node_types), function(t) {
      if (sc$feature_dims[[t]] > 0L) nn_linear(sc$feature_dims[[t]], h)
      else nn_embedding(sc$num_nodes[[t]], h)
    }))
    if (config$conv == "rgcn") {
      self$layers <- nn_module_list(lapply(seq_len(config$num_layers), function(i) {
        rgcn_conv(h, h, num_relations = length(sc$keys), num_bases = config$num_bases)
      }))
    } else if (config$conv == "hgt") {
      self$layers <- nn_module_list(lapply(seq_len(config$num_layers), function(i) {
        hgt_conv(sc$node_types, sc$keys, h, h, heads = config$heads,
                 temporal_encoding = isTRUE(config$temporal_encoding))
      }))
    } else {
      self$layers <- nn_module_list(lapply(seq_len(config$num_layers), function(i) {
        convs <- lapply(stats::setNames(sc$keys, sc$keys), function(k) {
          switch(config$conv,
            sage = sage_conv(c(h, h), h, aggr = config$aggr),
            graph_conv = graph_conv(c(h, h), h, aggr = config$aggr),
            gat = gat_conv(c(h, h), h %/% config$heads, heads = config$heads, add_self_loops = FALSE)
          )
        })
        hetero_conv(convs, aggr = config$relation_aggr)
      }))
    }
    if (!is.null(config$output_dim)) self$head <- nn_linear(h, config$output_dim)
  },
  forward = function(ht, return_embedding = FALSE) {
    sc <- self$config$schema
    h <- list()
    for (t in sc$node_types) {
      inp <- self$input[[t]]
      h[[t]] <- if (sc$feature_dims[[t]] > 0L) {
        inp(ht$x[[t]])
      } else if (!is.null(ht$node_index[[t]])) {
        # Sampled or batched graphs: look up each node's own embedding row.
        inp(ht$node_index[[t]])
      } else {
        if (ht$num_nodes[[t]] != sc$num_nodes[[t]]) {
          stopf("Node type `%s` has no features and %d nodes, but the model has embeddings for %d; give the original node indices in `n_id`.",
                t, ht$num_nodes[[t]], sc$num_nodes[[t]])
        }
        inp$weight
      }
    }
    for (layer in seq_along(self$layers)) {
      conv <- self$layers[[layer]]
      if (self$config$conv == "rgcn") {
        n <- vapply(h, function(v) v$size(1L), numeric(1))
        ptr <- stats::setNames(c(0, cumsum(n))[seq_along(n)], names(h))
        H <- torch_cat(unname(h[sc$node_types]), dim = 1L)
        eis <- list()
        ets <- list()
        for (i in seq_along(sc$keys)) {
          k <- sc$keys[i]
          ei <- ht$edge_index[[k]]
          if (is.null(ei) || ei$size(2L) == 0L) next
          p <- strsplit(k, "__", fixed = TRUE)[[1L]]
          off <- torch_tensor(c(ptr[[p[1L]]], ptr[[p[3L]]]), dtype = torch_long(),
                              device = ei$device)$unsqueeze(2L)
          eis[[length(eis) + 1L]] <- ei + off
          ets[[length(ets) + 1L]] <- torch_full(ei$size(2L), i, dtype = torch_long(), device = ei$device)
        }
        dev <- H$device
        ei_all <- if (length(eis)) torch_cat(eis, dim = 2L)
                  else torch_empty(2L, 0L, dtype = torch_long(), device = dev)
        et_all <- if (length(ets)) torch_cat(ets)
                  else torch_empty(0L, dtype = torch_long(), device = dev)
        H <- conv(H, ei_all, edge_type = et_all)
        h <- lapply(stats::setNames(sc$node_types, sc$node_types), function(t) {
          H$narrow(1L, ptr[[t]] + 1L, n[[t]])
        })
      } else if (self$config$conv == "hgt") {
        h <- conv(h, ht$edge_index, ht$edge_time)
      } else {
        h <- conv(h, ht$edge_index)
      }
      # The last layer stays linear when it gives the embedding, as in gnn_model().
      last <- layer == length(self$layers)
      if (!last || (!return_embedding && !is.null(self$head))) {
        h <- lapply(h, function(v) nnf_dropout(nnf_relu(v), p = self$config$dropout,
                                                training = self$training))
      }
    }
    if (!return_embedding && !is.null(self$head)) {
      tt <- self$config$target_type
      h[[tt]] <- self$head(h[[tt]])
    }
    h
  }
)

hetero_node_runner <- function(model, hg, classif, dev, target_type) {
  if (is.null(target_type)) stopf("Give `target_type` for heterogeneous node tasks.")
  nd <- hg$nodes[[target_type]]
  if (is.null(nd$y) || is.null(nd$train_mask)) {
    stopf("Node type `%s` needs `y` and `train_mask`.", target_type)
  }
  hg$nodes[[target_type]]$y <- prep_labels(nd$y, classif, what = "node labels")
  nd <- hg$nodes[[target_type]]
  ht <- hetero_tensors(hg, dev)
  y <- ht$y[[target_type]]
  tr <- tensor_which(ht$train_mask[[target_type]])
  va <- if (!is.null(nd$val_mask) && any(nd$val_mask)) tensor_which(ht$val_mask[[target_type]])
  metric_fn <- node_metric(classif)
  fwd <- function() model(ht)[[target_type]]
  list(
    default_loss = if (classif) "cross_entropy" else "mse",
    metric_name = if (classif) "accuracy" else "rmse",
    levels = if (classif) levels(nd$y),
    train_epoch = function(epoch, opt, loss_fn, grad_clip) {
      out <- fwd()
      loss <- loss_fn(out$index_select(1L, tr), y$index_select(1L, tr))
      step_optimizer(opt, loss, grad_clip, all_params(opt))
    },
    validate = function(loss_fn) {
      if (is.null(va)) return(list())
      out <- fwd()$index_select(1L, va)
      target <- y$index_select(1L, va)
      list(loss = as.numeric(loss_fn(out, target)$item()), metric = metric_fn(out, target))
    }
  )
}

hetero_link_runner <- function(model, split, dev, decoder, seed) {
  decoder <- decoder %||% link_decoder("dot")
  ht <- hetero_tensors(split$train_graph, dev)
  pt <- function(m) torch_tensor(m, dtype = torch_long(), device = dev)
  st <- split$src_type
  dt <- split$dst_type
  score <- function(z, pairs) decoder(list(z[[st]], z[[dt]]), pairs)
  train_pos <- pt(split$train_pos)
  list(
    default_loss = "bce", metric_name = "roc_auc", decoder = decoder,
    train_epoch = function(epoch, opt, loss_fn, grad_clip) {
      nseed <- if (is.null(seed)) NULL else seed + epoch
      neg <- if (isTRUE(split$same_type)) {
        negative_sampling(split$train_pos, num_nodes = split$num_src,
                          num_samples = ncol(split$train_pos), directed = FALSE, seed = nseed)
      } else {
        negative_sampling(split$train_pos, num_src = split$num_src, num_dst = split$num_dst,
                          num_samples = ncol(split$train_pos), seed = nseed)
      }
      z <- model(ht, return_embedding = TRUE)
      sp <- score(z, train_pos)
      sn <- score(z, pt(neg))
      loss <- loss_fn(torch_cat(list(sp, sn)),
                      torch_cat(list(torch_ones_like(sp), torch_zeros_like(sn))))
      step_optimizer(opt, loss, grad_clip, all_params(opt))
    },
    validate = function(loss_fn) {
      if (!ncol(split$val_pos) || !ncol(split$val_neg)) return(list())
      z <- model(ht, return_embedding = TRUE)
      sp <- score(z, pt(split$val_pos))
      sn <- score(z, pt(split$val_neg))
      s <- torch_cat(list(sp, sn))
      target <- torch_cat(list(torch_ones_like(sp), torch_zeros_like(sn)))
      list(loss = as.numeric(loss_fn(s, target)$item()),
           metric = metric_roc_auc(as_r_array(target) > 0.5, as_r_array(s)))
    }
  )
}

predict_hetero_links <- function(object, new_data, pairs, type) {
  split <- object$data
  g <- new_data %||% split$train_graph
  z <- object$model(hetero_tensors(g, object$device), return_embedding = TRUE)
  if (type == "embedding") return(lapply(z, as_r_array))
  if (is.null(pairs)) stopf("Link prediction needs `pairs` to score.")
  pairs <- normalize_edge_index(pairs)
  s <- object$decoder(list(z[[split$src_type]], z[[split$dst_type]]),
                      torch_tensor(pairs, dtype = torch_long(), device = object$device))
  s <- as.vector(as_r_array(s))
  if (type %in% c("score", "response")) return(s)
  if (type == "prob") return(stats::plogis(s))
  structure(data.frame(.row = seq_along(s), from = pairs[1L, ], to = pairs[2L, ],
                       .score = s, .prob = stats::plogis(s)),
            class = c("graph_predictions", "data.frame"))
}

#' Synthetic heterogeneous drug and protein graph
#'
#' A small simulated graph with `drug` and `protein` nodes for examples and
#' tests. Proteins belong to latent families; drugs belong to latent classes
#' and preferentially target proteins of the family linked to their class.
#' Drug labels are the latent class, so they can be predicted from the
#' targeted proteins. Protein-protein interactions are denser within
#' families. The data are simulated and carry no biological meaning.
#'
#' @param num_drugs,num_proteins Node counts.
#' @param num_classes Latent classes (drug labels and protein families).
#' @param seed Optional integer seed.
#' @return A `hetero_graph_data` with relations `drug__targets__protein` and
#'   `protein__interacts__protein`, drug labels and drug masks.
#' @export
dataset_synthetic_hetero <- function(num_drugs = 60L, num_proteins = 80L, num_classes = 3L,
                                     seed = NULL) {
  with_optional_seed(seed, {
    dcls <- sample(rep_len(seq_len(num_classes), num_drugs))
    pfam <- sample(rep_len(seq_len(num_classes), num_proteins))
    dx <- matrix(stats::rnorm(num_drugs * 4L), num_drugs, 4L)
    px <- cbind(matrix(stats::rnorm(num_proteins * 3L), num_proteins, 3L),
                outer(pfam, seq_len(num_classes), `==`) * 1.0)
    tgt <- do.call(cbind, lapply(seq_len(num_drugs), function(d) {
      same <- which(pfam == dcls[d])
      other <- which(pfam != dcls[d])
      p <- c(sample(same, 4L), sample(other, 1L))
      rbind(d, p)
    }))
    pp <- which(upper.tri(matrix(0, num_proteins, num_proteins)), arr.ind = TRUE)
    prob <- ifelse(pfam[pp[, 1L]] == pfam[pp[, 2L]], 0.08, 0.005)
    pp <- t(pp[stats::runif(nrow(pp)) < prob, , drop = FALSE])
    split_code <- sample(rep_len(1:3, num_drugs))
  })
  y <- factor(paste0("class", dcls), levels = paste0("class", seq_len(num_classes)))
  hetero_graph_data(
    nodes = list(
      drug = list(x = dx, y = y, train_mask = split_code == 1L,
                  val_mask = split_code == 2L, test_mask = split_code == 3L),
      protein = list(x = px)
    ),
    edges = list(drug__targets__protein = tgt,
                 protein__interacts__protein = cbind(pp, pp[2:1, , drop = FALSE])),
    metadata = list(name = "synthetic_drug_protein", source = "simulated", seed = seed)
  )
}
