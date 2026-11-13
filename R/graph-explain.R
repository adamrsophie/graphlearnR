#' Explain node predictions
#'
#' Post hoc explanations of a trained model's output for one target node (or
#' one graph, with `target = NULL` and a graph level model). They describe
#' **how the model's output responds to its inputs**; they are not causal
#' explanations of the data generating process, and attention weights alone
#' are not importance scores.
#'
#' * `explain_saliency()`: gradient of the selected output with respect to
#'   node features (Simonyan et al. 2014), optionally times the input.
#' * `explain_integrated_gradients()`: integrated gradients (Sundararajan et
#'   al. 2017) along a straight path from `baseline` to the features with a
#'   Riemann sum of `steps` points. The result includes `completeness_gap`,
#'   the difference between the summed attributions and `f(x) - f(baseline)`;
#'   a large gap means more steps are needed.
#' * `explain_edges()`: gradient of the output with respect to a multiplicative
#'   mask on edge weights, evaluated at mask 1. Needs a model whose layers
#'   use `edge_weight` (`"gcn"`, `"sage"`, `"graph_conv"`, `"gin"`).
#' * `explain_occlusion()`: change in the output when each feature column, or
#'   each incoming edge of the target, is removed.
#' * `gnn_explainer()`: learns a soft edge mask and feature mask that keep the
#'   prediction while staying sparse (Ying et al. 2019).
#'
#' @param model A trained model (or a `graph_fit`).
#' @param g A `graph_data` object.
#' @param target Target node index, or `NULL` for a graph level output.
#' @param output Output column to explain; default the predicted class.
#' @param times_input Multiply gradients by the input.
#' @return A list with attribution matrices or vectors and metadata, of
#'   class `graph_explanation`.
#' @examples
#' if (torch::torch_is_installed()) {
#'   g <- split_nodes(dataset_karate(), seed = 1)
#'   fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g,
#'                            "node_classification", epochs = 20, seed = 1)
#'   ex <- explain_edges(fit, g, target = 1)
#'   head(ex$edges[order(-abs(ex$edges$attribution)), ])
#' }
#' @name explain
NULL

# The model in evaluation mode with parameters that do not track gradients,
# so explanations take gradients with respect to inputs and masks only and
# never accumulate into parameter gradients. The previous mode and
# `requires_grad` flags are restored when the calling explainer returns.
explain_model <- function(model, envir = parent.frame()) {
  m <- if (inherits(model, "graph_fit")) model$model else model
  params <- m$parameters
  flags <- vapply(params, function(p) isTRUE(p$requires_grad), logical(1))
  was_training <- isTRUE(m$training)
  for (p in params) p$requires_grad_(FALSE)
  m$eval()
  withr::defer({
    for (i in seq_along(params)) params[[i]]$requires_grad_(flags[[i]])
    if (was_training) m$train()
  }, envir = envir)
  m
}

select_output <- function(out, target, output) {
  row <- if (is.null(target)) out[1L, ] else out[target, ]
  idx <- output %||% as.integer(row$argmax()$item())
  list(value = row[idx], index = idx)
}

forward_with <- function(model, tt, x = tt$x, edge_weight = tt$edge_weight) {
  model(x, tt$edge_index, edge_weight = edge_weight, edge_attr = tt$edge_attr,
        batch = tt$batch, num_graphs = tt$num_graphs, edge_type = tt$edge_type)
}

#' @rdname explain
#' @export
explain_saliency <- function(model, g, target = NULL, output = NULL, times_input = FALSE) {
  model <- explain_model(model)
  dev <- model$parameters[[1L]]$device
  tt <- graph_tensors(g, dev)
  x <- tt$x$detach()$requires_grad_(TRUE)
  sel <- select_output(forward_with(model, tt, x), target, output)
  sel$value$backward()
  grad <- as_r_array(x$grad)
  att <- if (times_input) grad * g$x else grad
  structure(list(method = "saliency", target = target, output = sel$index,
                 node_feature = att), class = "graph_explanation")
}

#' @rdname explain
#' @param baseline Baseline features: a number or a matrix like `g$x`.
#' @param steps Riemann sum points.
#' @export
explain_integrated_gradients <- function(model, g, target = NULL, output = NULL,
                                         baseline = 0, steps = 50L) {
  model <- explain_model(model)
  dev <- model$parameters[[1L]]$device
  tt <- graph_tensors(g, dev)
  x <- tt$x
  base <- if (is.matrix(baseline)) torch_tensor(baseline, dtype = x$dtype, device = dev)
          else torch_zeros_like(x) + baseline
  output <- output %||% with_no_grad(select_output(forward_with(model, tt), target, NULL))$index
  total <- torch_zeros_like(x)
  for (s in seq_len(steps)) {
    a <- (s - 0.5) / steps
    xi <- (base + a * (x - base))$detach()$requires_grad_(TRUE)
    select_output(forward_with(model, tt, xi), target, output)$value$backward()
    total <- total + xi$grad
  }
  attr <- as_r_array((x - base) * total / steps)
  f <- function(z) with_no_grad(as.numeric(select_output(forward_with(model, tt, z), target, output)$value$item()))
  gap <- sum(attr) - (f(x) - f(base))
  structure(list(method = "integrated_gradients", target = target, output = output,
                 node_feature = attr, completeness_gap = gap, steps = steps),
            class = "graph_explanation")
}

#' @rdname explain
#' @export
explain_edges <- function(model, g, target = NULL, output = NULL) {
  model <- explain_model(model)
  dev <- model$parameters[[1L]]$device
  tt <- graph_tensors(g, dev)
  w0 <- tt$edge_weight %||% torch_ones(tt$edge_index$size(2L), device = dev)
  mask <- torch_ones_like(w0)$requires_grad_(TRUE)
  out <- forward_with(model, tt, edge_weight = w0 * mask)
  sel <- select_output(out, target, output)
  sel$value$backward()
  grad <- if (is.null(mask$grad)) rep(NA_real_, num_edges(g)) else as.vector(as_r_array(mask$grad))
  if (all(grad == 0, na.rm = TRUE)) {
    warnf("All edge gradients are zero; the model may ignore `edge_weight`.")
  }
  structure(list(method = "edge_gradient", target = target, output = sel$index,
                 edges = data.frame(edge = seq_along(grad), from = g$edge_index[1L, ],
                                    to = g$edge_index[2L, ], attribution = grad)),
            class = "graph_explanation")
}

#' @rdname explain
#' @export
explain_occlusion <- function(model, g, target = NULL, output = NULL) {
  model <- explain_model(model)
  dev <- model$parameters[[1L]]$device
  tt <- graph_tensors(g, dev)
  with_no_grad({
    sel <- select_output(forward_with(model, tt), target, output)
    ref <- as.numeric(sel$value$item())
    feat <- vapply(seq_len(ncol(g$x)), function(j) {
      x2 <- tt$x$clone()
      x2[, j] <- 0
      ref - as.numeric(select_output(forward_with(model, tt, x2), target, sel$index)$value$item())
    }, numeric(1))
    cand <- if (is.null(target)) seq_len(num_edges(g)) else which(g$edge_index[2L, ] == target)
    edge <- vapply(cand, function(e) {
      g2 <- edge_subset(g, setdiff(seq_len(num_edges(g)), e))
      t2 <- graph_tensors(g2, dev)
      ref - as.numeric(select_output(forward_with(model, t2), target, sel$index)$value$item())
    }, numeric(1))
  })
  structure(list(method = "occlusion", target = target, output = sel$index,
                 feature = feat,
                 edges = data.frame(edge = cand, from = g$edge_index[1L, cand],
                                    to = g$edge_index[2L, cand], attribution = edge)),
            class = "graph_explanation")
}

#' @rdname explain
#' @param epochs Optimization steps.
#' @param lr Learning rate.
#' @param edge_size,feature_size Sparsity penalties.
#' @param entropy Mask entropy penalty, pushing mask values towards 0 or 1.
#' @param seed Optional integer seed.
#' @export
gnn_explainer <- function(model, g, target = NULL, output = NULL, epochs = 100L, lr = 0.01,
                          edge_size = 0.005, feature_size = 1, entropy = 1, seed = NULL) {
  model <- explain_model(model)
  dev <- model$parameters[[1L]]$device
  if (!is.null(seed)) torch_manual_seed(seed)
  tt <- graph_tensors(g, dev)
  w0 <- tt$edge_weight %||% torch_ones(tt$edge_index$size(2L), device = dev)
  full <- with_no_grad(forward_with(model, tt))
  # The edge mask acts through `edge_weight`; a model that ignores edge
  # weights (attention layers, for example) can not be explained this way.
  if (tt$edge_index$size(2L) > 0L) {
    # Halve every other edge: a uniform scale would miss layers that
    # normalize edge weights.
    ne <- tt$edge_index$size(2L)
    scale <- torch_tensor(rep_len(c(1, 0.5), ne), dtype = w0$dtype, device = dev)
    half <- with_no_grad(forward_with(model, tt, edge_weight = w0 * scale))
    if (as.logical(torch_allclose(full, half))) {
      stopf("The model's output does not depend on `edge_weight`, so an edge mask can not explain it.")
    }
  }
  regression <- full$size(2L) == 1L
  output <- output %||% select_output(full, target, NULL)$index
  orig <- select_output(full, target, output)$value
  em <- (torch_randn_like(w0) * 0.1 + 1)$requires_grad_(TRUE)
  fm <- (torch_randn(ncol(g$x), device = dev) * 0.1 + 1)$requires_grad_(TRUE)
  opt <- optim_adam(list(em, fm), lr = lr)
  ent <- function(m) (-m * torch_log(m + 1e-15) - (1 - m) * torch_log(1 - m + 1e-15))$mean()
  for (i in seq_len(epochs)) {
    opt$zero_grad()
    e <- torch_sigmoid(em)
    f <- torch_sigmoid(fm)
    out <- forward_with(model, tt, tt$x * f, w0 * e)
    # Classification: keep the explained class likely. Regression (one
    # output): keep the prediction close to the full graph prediction.
    fit_term <- if (regression) {
      (select_output(out, target, output)$value - orig)$pow(2)
    } else {
      row <- if (is.null(target)) out[1L, ] else out[target, ]
      -nnf_log_softmax(row, dim = 1L)[output]
    }
    loss <- fit_term + edge_size * e$sum() + entropy * ent(e) + feature_size * f$mean() + entropy * ent(f)
    loss$backward()
    opt$step()
  }
  e <- as.vector(as_r_array(torch_sigmoid(em)))
  structure(list(method = "gnn_explainer", target = target, output = output,
                 edges = data.frame(edge = seq_along(e), from = g$edge_index[1L, ],
                                    to = g$edge_index[2L, ], attribution = e),
                 feature = as.vector(as_r_array(torch_sigmoid(fm)))),
            class = "graph_explanation")
}

#' Fidelity and stability of an explanation
#'
#' `explanation_fidelity()` removes the `k` edges with the largest absolute
#' attribution (\eqn{fid^{+}}, the drop in the explained output; large is
#' good) and, separately, keeps only those `k` edges (\eqn{fid^{-}}, the drop
#' when everything else is removed; small is good), following Yuan et al.
#' (2022). `explanation_stability()` reruns an explainer on graphs with small
#' Gaussian feature noise and reports the mean Spearman correlation of edge
#' attributions with the original.
#'
#' @param model A trained model or `graph_fit`.
#' @param g A `graph_data` object.
#' @param explanation A `graph_explanation` with an `edges` table.
#' @param k Number of top edges.
#' @return A named numeric vector.
#' @examples
#' if (torch::torch_is_installed()) {
#'   g <- split_nodes(dataset_karate(), seed = 1)
#'   fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g, "node_classification",
#'                            epochs = 20, seed = 1)
#'   explanation_fidelity(fit, g, explain_edges(fit, g, target = 1), k = 3)
#' }
#' @export
explanation_fidelity <- function(model, g, explanation, k = 5L) {
  model <- explain_model(model)
  dev <- model$parameters[[1L]]$device
  ed <- explanation$edges
  top <- ed$edge[order(-abs(ed$attribution))][seq_len(min(k, nrow(ed)))]
  val <- function(gg) {
    with_no_grad(as.numeric(select_output(forward_with(model, graph_tensors(gg, dev)),
                                          explanation$target, explanation$output)$value$item()))
  }
  ref <- val(g)
  all_e <- seq_len(num_edges(g))
  c(fidelity_plus = ref - val(edge_subset(g, setdiff(all_e, top))),
    fidelity_minus = ref - val(edge_subset(g, sort(top))))
}

#' @rdname explanation_fidelity
#' @param explainer Function `(model, g)` returning a `graph_explanation`.
#' @param noise Standard deviation of feature noise.
#' @param reps Number of noisy repetitions.
#' @param seed Optional integer seed.
#' @export
explanation_stability <- function(model, g, explainer, noise = 0.01, reps = 5L, seed = NULL) {
  base <- explainer(model, g)$edges$attribution
  cors <- with_optional_seed(seed, vapply(seq_len(reps), function(r) {
    g2 <- g
    g2$x <- g$x + matrix(stats::rnorm(length(g$x), sd = noise), nrow(g$x))
    stats::cor(base, explainer(model, g2)$edges$attribution, method = "spearman")
  }, numeric(1)))
  c(mean_spearman = mean(cors), min_spearman = min(cors))
}

#' Explanation subgraph
#'
#' Turns an edge attribution into a subgraph explanation: the `k` edges with
#' the largest absolute attribution (and, for undirected graphs, their
#' reverse directions), the nodes they touch, and the target node. The
#' result is a `graph_data` object with local 1-based indices; its
#' `node_ids` hold the original node identifiers (`g$node_ids`, which are
#' the node indices unless the graph has external identifiers) so the
#' subgraph can be mapped back, and `edge_original` the original edge
#' positions.
#'
#' @param g The `graph_data` object that was explained.
#' @param explanation A `graph_explanation` with an `edges` table, for
#'   example from [explain_edges()] or [gnn_explainer()].
#' @param k Number of top edges (directions of one undirected pair count
#'   once).
#' @return A `graph_data` object.
#' @examples
#' if (torch::torch_is_installed()) {
#'   g <- split_nodes(dataset_karate(), seed = 1)
#'   fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g,
#'                            "node_classification", epochs = 20, seed = 1)
#'   sub <- explanation_subgraph(g, explain_edges(fit, g, target = 1), k = 5)
#'   sub$node_ids
#' }
#' @export
explanation_subgraph <- function(g, explanation, k = 5L) {
  ed <- explanation$edges
  if (is.null(ed)) stopf("`explanation` has no edge attributions.")
  k <- check_count(k, "k", 1L)
  key <- if (!is.null(g$edge_pair)) g$edge_pair[ed$edge] else ed$edge
  # Rank pairs by the largest absolute attribution of either direction.
  score <- tapply(abs(ed$attribution), key, max)
  top_keys <- names(sort(score, decreasing = TRUE))[seq_len(min(k, length(score)))]
  edges <- sort(ed$edge[as.character(key) %in% top_keys])
  nodes <- sort(unique(c(explanation$target, g$edge_index[, edges])))
  sub <- edge_subset(g, edges)
  ids <- g$node_ids %||% seq_len(g$num_nodes)
  out <- induced_subgraph(sub, nodes)
  out$node_ids <- ids[nodes]
  out$edge_original <- edges
  out
}

#' Attention weights of attention layers
#'
#' Collects the attention coefficients stored by attention layers
#' (`gat_conv`, `gatv2_conv`, `transformer_conv`) during the last forward
#' pass. Run a forward pass (for example [predict()]) first. Attention shows
#' how a layer weighted neighbors; it is not a feature importance measure.
#'
#' @param model A model or `graph_fit`.
#' @return A list, one data frame per attention layer, with `from`, `to` and
#'   one column per head.
#' @examples
#' if (torch::torch_is_installed()) {
#'   g <- split_nodes(dataset_karate(), seed = 1)
#'   fit <- train_graph_model(gnn_model("gat", 34, 8, 2, heads = 2), g,
#'                            "node_classification", epochs = 5, seed = 1)
#'   invisible(predict(fit, g))
#'   head(attention_weights(fit)[[1]])
#' }
#' @export
attention_weights <- function(model) {
  model <- if (inherits(model, "graph_fit")) model$model else model
  out <- list()
  for (m in model$modules) {
    la <- tryCatch(m$last_attention, error = function(e) NULL)
    # Only layers that store an edge list with coefficients (gat_conv,
    # gatv2_conv, transformer_conv); other attention modules keep different
    # structures and are skipped.
    if (is.null(la) || !is.list(la) || is.null(la$edge_index) || is.null(la$alpha)) next
    ei <- as_r_array(la$edge_index)
    a <- as_r_array(la$alpha)
    if (!is.matrix(a)) a <- matrix(a, ncol = 1L)
    df <- data.frame(from = ei[1L, ], to = ei[2L, ])
    df <- cbind(df, stats::setNames(as.data.frame(a), paste0("head", seq_len(ncol(a)))))
    out[[length(out) + 1L]] <- df
  }
  out
}

#' @export
print.graph_explanation <- function(x, ...) {
  cat(sprintf("<graph_explanation> %s for %s, output %s\n", x$method,
              if (is.null(x$target)) "the graph" else paste("node", x$target), x$output))
  invisible(x)
}
