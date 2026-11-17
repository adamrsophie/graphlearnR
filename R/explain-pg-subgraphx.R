#' Parameterized explainer (PGExplainer)
#'
#' PGExplainer (Luo et al. 2020) learns one small network that predicts an
#' edge mask for **any** instance from the model's node embeddings, instead
#' of optimizing a separate mask per instance as [gnn_explainer()] does.
#' For an edge \eqn{u \to v} the mask logit is
#' \eqn{\omega_{uv} = \mathrm{MLP}([z_u, z_v, z_t])} for a node target
#' \eqn{t} (or \eqn{[z_u, z_v]} for a graph level output), with \eqn{z} the
#' trained model's last hidden representation (`return_embedding = TRUE`),
#' which is not updated.
#'
#' Training samples a relaxed Bernoulli (concrete) mask
#' \eqn{m = \sigma((\log \epsilon - \log(1 - \epsilon) + \omega) / \tau)},
#' \eqn{\epsilon \sim U(0, 1)}, multiplies it into `edge_weight`, and
#' minimizes the negative log probability of the class the model predicts on
#' the full graph (squared change for one output regression), plus
#' `edge_size` times the mask sum and `entropy` times the mask entropy. The
#' temperature \eqn{\tau} decays geometrically from `temperature[1]` to
#' `temperature[2]`. The fitted explainer then explains a new instance with
#' one forward pass, \eqn{m = \sigma(\omega)}.
#'
#' Like [gnn_explainer()], the mask acts through `edge_weight`, so the model
#' must use edge weights; the explanation shows how the model responds to
#' edges, not causal relevance.
#'
#' @param model A trained model or `graph_fit` that accepts
#'   `return_embedding = TRUE` (for example from [gnn_model()]).
#' @param data A `graph_data` object (node level models) or a list of them
#'   (graph level models).
#' @param targets For node level models, the target nodes to train on
#'   (default: training nodes, at most 100).
#' @param hidden MLP hidden size.
#' @param epochs Passes over the instances.
#' @param lr Learning rate.
#' @param edge_size,entropy Mask size and entropy penalties.
#' @param temperature Start and end temperature of the concrete relaxation.
#' @param seed Optional integer seed.
#' @return `pg_explainer()`: an object of class `pg_explainer`;
#'   `predict(explainer, g, target)`: a `graph_explanation` whose `edges`
#'   table holds the mask probabilities.
#' @references Luo et al. (2020) Parameterized explainer for graph neural
#'   network. NeurIPS.
#' @examples
#' if (torch::torch_is_installed()) {
#'   g <- split_nodes(dataset_karate(), seed = 1)
#'   fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g,
#'                            "node_classification", epochs = 20, seed = 1)
#'   pg <- pg_explainer(fit, g, epochs = 3, seed = 1)
#'   ex <- predict(pg, g, target = 1)
#'   head(ex$edges[order(-ex$edges$attribution), ])
#' }
#' @export
pg_explainer <- function(model, data, targets = NULL, hidden = 64L, epochs = 30L, lr = 0.003,
                         edge_size = 0.05, entropy = 1, temperature = c(5, 2), seed = NULL) {
  m <- explain_model(model)
  dev <- m$parameters[[1L]]$device
  if (!is.null(seed)) torch_manual_seed(seed)
  graph_level <- !inherits(data, "graph_data")
  graphs <- if (graph_level) data else list(data)
  if (!length(graphs) || !all(vapply(graphs, inherits, logical(1), "graph_data"))) {
    stopf("`data` must be a graph_data object or a list of them.")
  }
  if (!graph_level) {
    targets <- targets %||% {
      tr <- if (!is.null(data$train_mask)) which(data$train_mask) else seq_len(data$num_nodes)
      utils::head(tr, 100L)
    }
    targets <- as.integer(targets)
    if (!length(targets) || any(targets < 1L | targets > data$num_nodes)) {
      stopf("`targets` must be node indices of `data`.")
    }
  }
  # One instance per target node, or per graph.
  inst <- if (graph_level) lapply(seq_along(graphs), function(i) list(g = i, target = NULL))
          else lapply(targets, function(t) list(g = 1L, target = t))
  tts <- lapply(graphs, graph_tensors, device = dev)
  pre <- lapply(tts, function(tt) {
    with_no_grad({
      full <- forward_with(m, tt)
      z <- forward_with(m, tt, return_embedding = TRUE)
    })
    list(full = full, z = z, w0 = tt$edge_weight %||% torch_ones(tt$edge_index$size(2L), device = dev))
  })
  check_edge_weight_use(m, tts[[1L]], pre[[1L]]$w0, pre[[1L]]$full)
  zdim <- pre[[1L]]$z$size(2L)
  mlp <- nn_sequential(nn_linear((if (graph_level) 2L else 3L) * zdim, hidden), nn_relu(),
                       nn_linear(hidden, 1L))
  mlp$to(device = dev)
  regression <- pre[[1L]]$full$size(2L) == 1L
  logits <- function(i, target) pg_logits(mlp, pre[[i]]$z, tts[[i]]$edge_index, target)
  opt <- optim_adam(mlp$parameters, lr = lr)
  epochs <- check_count(epochs, "epochs", 1L)
  ent <- function(p) (-p * torch_log(p + 1e-15) - (1 - p) * torch_log(1 - p + 1e-15))$mean()
  history <- numeric(epochs)
  # `seed` fixes both the torch draws and the R order of instances, without
  # changing the caller's R random number stream.
  with_optional_seed(seed, for (ep in seq_len(epochs)) {
    tau <- temperature[1L] * (temperature[2L] / temperature[1L])^((ep - 1) / max(1, epochs - 1))
    tot <- 0
    for (ins in inst[sample.int(length(inst))]) {
      i <- ins$g
      opt$zero_grad()
      w <- logits(i, ins$target)
      eps <- torch_rand_like(w)$clamp(1e-6, 1 - 1e-6)
      mask <- torch_sigmoid((torch_log(eps) - torch_log(1 - eps) + w) / tau)
      out <- forward_with(m, tts[[i]], edge_weight = pre[[i]]$w0 * mask)
      orig <- select_output(pre[[i]]$full, ins$target, NULL)
      fit_term <- if (regression) {
        (select_output(out, ins$target, 1L)$value - orig$value)$pow(2)
      } else {
        row <- if (is.null(ins$target)) out[1L, ] else out[ins$target, ]
        -nnf_log_softmax(row, dim = 1L)[orig$index]
      }
      p <- torch_sigmoid(w)
      loss <- fit_term + edge_size * p$sum() + entropy * ent(p)
      loss$backward()
      opt$step()
      tot <- tot + as.numeric(loss$item())
    }
    history[ep] <- tot / length(inst)
  })
  mlp$eval()
  structure(list(mlp = mlp, model = m, graph_level = graph_level, history = history,
                 device = dev),
            class = "pg_explainer")
}

pg_logits <- function(mlp, z, edge_index, target) {
  parts <- list(z$index_select(1L, edge_index[1L, ]), z$index_select(1L, edge_index[2L, ]))
  if (!is.null(target)) parts[[3L]] <- z[target, ]$unsqueeze(1L)$expand(c(edge_index$size(2L), -1L))
  mlp(torch_cat(parts, dim = 2L))$squeeze(-1L)
}

#' @rdname pg_explainer
#' @param object A `pg_explainer`.
#' @param g Graph to explain.
#' @param target Target node (node level models), or `NULL`.
#' @param ... Unused.
#' @export
predict.pg_explainer <- function(object, g, target = NULL, ...) {
  if (!object$graph_level && is.null(target)) stopf("Give `target` for a node level explainer.")
  m <- explain_model(object$model)
  tt <- graph_tensors(g, object$device)
  p <- with_no_grad({
    z <- forward_with(m, tt, return_embedding = TRUE)
    full <- forward_with(m, tt)
    list(mask = torch_sigmoid(pg_logits(object$mlp, z, tt$edge_index, target)),
         output = select_output(full, target, NULL)$index)
  })
  e <- as.vector(as_r_array(p$mask))
  structure(list(method = "pg_explainer", target = target, output = p$output,
                 edges = data.frame(edge = seq_along(e), from = g$edge_index[1L, ],
                                    to = g$edge_index[2L, ], attribution = e)),
            class = "graph_explanation")
}

# Edge masks act through edge_weight; refuse models that ignore it. Two
# probes halve alternating edges (one edge: halve it), since a uniform scale
# is invisible to layers that normalize edge weights. Unchanged outputs
# under both probes are taken as no dependence; a model could still depend
# on weights in ways these probes miss.
check_edge_weight_use <- function(model, tt, w0, full) {
  ne <- tt$edge_index$size(2L)
  if (ne == 0L) return(invisible())
  for (pat in list(c(0.5, 1), c(1, 0.5))) {
    scale <- torch_tensor(rep_len(pat, ne), dtype = w0$dtype, device = w0$device)
    half <- with_no_grad(forward_with(model, tt, edge_weight = w0 * scale))
    if (!as.logical(torch_allclose(full, half))) return(invisible())
  }
  stopf("The model's output does not change when `edge_weight` is scaled, so an edge mask can not explain it.")
}

#' Subgraph explanations by Monte Carlo tree search (SubgraphX)
#'
#' SubgraphX (Yuan et al. 2021) searches for a small connected subgraph that
#' explains a prediction. Each node of the search tree is a node set; its
#' children remove one node and keep the connected component that holds the
#' target (node level) or the largest component (graph level). A subgraph
#' \eqn{S} is scored by its Shapley value as one player among the other
#' nodes within `hops` hops of it,
#' \deqn{\phi(S) \approx \frac{1}{M}\sum_{s=1}^{M} f(C_s \cup S) - f(C_s),}
#' with coalitions \eqn{C_s} drawn as the predecessors of \eqn{S} in a
#' random ordering (a uniform size, then a uniform subset), and \eqn{f} the
#' explained output when only the given nodes are kept: other nodes have
#' their features set to zero and lose their edges (the target node of a
#' node level explanation is always kept). Search follows the paper's
#' upper confidence rule, \eqn{Q + c \, \phi \sqrt{\sum N} / (1 + N)}, over
#' `rollouts` descents, expanding at most `expand_atoms` children per
#' node (the lowest degree nodes first). The best scoring subgraph with at
#' most `max_nodes` nodes is returned.
#'
#' Removing nodes rather than masking edges works with any model, including
#' attention models that ignore `edge_weight`. The cost is many forward
#' passes (about `rollouts` times the search depth times `expand_atoms`
#' times `samples`, with caching), so this suits small graphs and
#' neighborhoods.
#'
#' @param model A trained model or `graph_fit`.
#' @param g A `graph_data` object.
#' @param target Target node, or `NULL` for a graph level output.
#' @param output Output column; default the predicted class (or the single
#'   regression output).
#' @param hops Neighborhood radius: the model's number of message passing
#'   layers.
#' @param max_nodes Largest explanation size.
#' @param rollouts Tree search descents.
#' @param expand_atoms Children expanded per tree node.
#' @param samples Monte Carlo samples per Shapley value.
#' @param c_puct Exploration constant.
#' @param seed Optional integer seed.
#' @return A `graph_explanation` with `subgraph_nodes`, its Shapley `score`,
#'   a `candidates` table of evaluated subgraphs, and an `edges` table with
#'   attribution 1 for edges inside the subgraph and 0 otherwise.
#' @references Yuan, Yu, Wang, Li and Ji (2021) On explainability of graph
#'   neural networks via subgraph explorations. ICML.
#' @examples
#' if (torch::torch_is_installed()) {
#'   g <- split_nodes(dataset_karate(), seed = 1)
#'   fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g,
#'                            "node_classification", epochs = 20, seed = 1)
#'   ex <- subgraphx(fit, g, target = 1, max_nodes = 4, rollouts = 3, samples = 5, seed = 1)
#'   ex$subgraph_nodes
#' }
#' @export
subgraphx <- function(model, g, target = NULL, output = NULL, hops = 2L, max_nodes = 5L,
                      rollouts = 10L, expand_atoms = 4L, samples = 10L, c_puct = 10,
                      seed = NULL) {
  m <- explain_model(model)
  dev <- m$parameters[[1L]]$device
  if (!inherits(g, "graph_data")) stopf("`g` must be a graph_data object.")
  max_nodes <- check_count(max_nodes, "max_nodes", 1L)
  rollouts <- check_count(rollouts, "rollouts", 1L)
  samples <- check_count(samples, "samples", 1L)
  hops <- check_count(hops, "hops", 1L)
  n <- g$num_nodes
  if (!is.null(target) && (!is_scalar_integerish(target) || target < 1L || target > n)) {
    stopf("`target` must be a node of `g`.")
  }
  ei <- g$edge_index
  full <- with_no_grad(forward_with(m, graph_tensors(g, dev)))
  output <- output %||% select_output(full, target, NULL)$index
  # Undirected neighbor lists, for hops and connectivity.
  nbr <- split(c(ei[2L, ], ei[1L, ]), factor(c(ei[1L, ], ei[2L, ]), levels = seq_len(n)))
  within_hops <- function(nodes, k) {
    seen <- nodes
    front <- nodes
    for (i in seq_len(k)) {
      front <- setdiff(unique(unlist(nbr[front], use.names = FALSE)), seen)
      seen <- c(seen, front)
    }
    sort(seen)
  }
  component <- function(nodes) {
    keep <- logical(n)
    keep[nodes] <- TRUE
    comp <- integer(n)
    id <- 0L
    for (s in nodes) {
      if (comp[s]) next
      id <- id + 1L
      stack <- s
      comp[s] <- id
      while (length(stack)) {
        u <- stack[1L]
        stack <- stack[-1L]
        nb <- nbr[[u]]
        nb <- nb[keep[nb] & comp[nb] == 0L]
        comp[nb] <- id
        stack <- c(stack, unique(nb))
      }
    }
    pick <- if (!is.null(target)) comp[target] else as.integer(names(which.max(table(comp[nodes]))))
    sort(nodes[comp[nodes] == pick])
  }
  connected <- function(S) length(component(S)) == length(S)
  value_cache <- new.env(parent = emptyenv())
  value <- function(nodes) {
    nodes <- sort(unique(c(nodes, target)))
    key <- paste0("k", paste(nodes, collapse = ","))
    if (!is.null(value_cache[[key]])) return(value_cache[[key]])
    keep <- logical(n)
    keep[nodes] <- TRUE
    g2 <- edge_subset(g, which(keep[ei[1L, ]] & keep[ei[2L, ]]))
    g2$x <- g$x * keep
    v <- with_no_grad(as.numeric(select_output(forward_with(m, graph_tensors(g2, dev)),
                                               target, output)$value$item()))
    value_cache[[key]] <- v
    v
  }
  score_cache <- new.env(parent = emptyenv())
  shapley <- function(S) {
    key <- paste0("k", paste(S, collapse = ","))
    if (!is.null(score_cache[[key]])) return(score_cache[[key]])
    players <- setdiff(within_hops(S, hops), S)
    s <- mean(vapply(seq_len(samples), function(i) {
      k <- sample.int(length(players) + 1L, 1L) - 1L
      C <- players[sample.int(length(players), k)]
      value(c(C, S)) - value(C)
    }, numeric(1)))
    score_cache[[key]] <- s
    s
  }
  root <- if (!is.null(target)) within_hops(target, hops) else seq_len(n)
  tree <- new.env(parent = emptyenv())
  key_of <- function(S) paste0("k", paste(S, collapse = ","))
  new_node <- function(S) {
    k <- key_of(S)
    if (is.null(tree[[k]])) tree[[k]] <- list(S = S, W = 0, N = 0, P = NA_real_, children = NULL)
    k
  }
  expand <- function(k) {
    nd <- tree[[k]]
    S <- nd$S
    cand <- setdiff(S, target)
    deg <- vapply(cand, function(u) sum(nbr[[u]] %in% S), numeric(1))
    cand <- cand[order(deg)][seq_len(min(expand_atoms, length(cand)))]
    kids <- unique(lapply(cand, function(u) component(setdiff(S, u))))
    kids <- kids[lengths(kids) > 0L]
    keys <- vapply(kids, function(C) {
      ck <- new_node(C)
      if (is.na(tree[[ck]]$P)) {
        e <- tree[[ck]]
        e$P <- shapley(C)
        tree[[ck]] <- e
      }
      ck
    }, character(1))
    nd$children <- keys
    tree[[k]] <- nd
  }
  rk <- new_node(root)
  with_optional_seed(seed, {
    for (r in seq_len(rollouts)) {
      path <- rk
      k <- rk
      # Search until the subgraph is within budget and connected.
      while (length(tree[[k]]$S) > max_nodes || !connected(tree[[k]]$S)) {
        if (is.null(tree[[k]]$children)) expand(k)
        kids <- tree[[k]]$children
        if (!length(kids)) break
        Ns <- vapply(kids, function(c) tree[[c]]$N, numeric(1))
        Q <- vapply(kids, function(c) if (tree[[c]]$N > 0) tree[[c]]$W / tree[[c]]$N else 0, numeric(1))
        P <- vapply(kids, function(c) tree[[c]]$P, numeric(1))
        k <- kids[which.max(Q + c_puct * P * sqrt(sum(Ns)) / (1 + Ns))]
        path <- c(path, k)
      }
      leaf <- tree[[k]]
      sc <- if (is.na(leaf$P)) shapley(leaf$S) else leaf$P
      for (p in path) {
        e <- tree[[p]]
        e$W <- e$W + sc
        e$N <- e$N + 1
        tree[[p]] <- e
      }
    }
  })
  nodes_all <- lapply(ls(tree), function(k) tree[[k]])
  small <- Filter(function(e) length(e$S) <= max_nodes && connected(e$S), nodes_all)
  if (!length(small)) small <- Filter(function(e) connected(e$S), nodes_all)
  if (!length(small)) small <- nodes_all
  scores <- vapply(small, function(e) if (is.na(e$P)) shapley(e$S) else e$P, numeric(1))
  best <- small[[which.max(scores)]]$S
  inside <- ei[1L, ] %in% best & ei[2L, ] %in% best
  cands <- data.frame(size = vapply(small, function(e) length(e$S), integer(1)),
                      nodes = vapply(small, function(e) paste(e$S, collapse = ","), character(1)),
                      score = scores)
  structure(list(method = "subgraphx", target = target, output = output,
                 subgraph_nodes = best, score = max(scores),
                 candidates = cands[order(-cands$score), , drop = FALSE],
                 edges = data.frame(edge = seq_len(ncol(ei)), from = ei[1L, ], to = ei[2L, ],
                                    attribution = as.numeric(inside))),
            class = "graph_explanation")
}
