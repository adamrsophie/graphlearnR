explain_setup <- function() {
  g <- split_nodes(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("gcn", 34, 8, 2, dropout = 0), g, "node_classification",
                           epochs = 30, seed = 1)
  list(g = g, fit = fit)
}

test_that("saliency equals a finite difference of the selected output", {
  skip_if_no_torch()
  s <- explain_setup()
  ex <- explain_saliency(s$fit, s$g, target = 1)
  expect_s3_class(ex, "graph_explanation")
  expect_equal(dim(ex$node_feature), c(34, 34))
  m <- s$fit$model
  f <- function(x) {
    g2 <- s$g
    g2$x <- x
    torch::with_no_grad(as.numeric(graphlearnR:::model_forward(m, graph_tensors(g2))[1, ex$output]$item()))
  }
  xp <- s$g$x; xp[2, 2] <- xp[2, 2] + 1e-2
  xm <- s$g$x; xm[2, 2] <- xm[2, 2] - 1e-2
  # ReLU kinks and float32 make the central difference approximate.
  expect_equal(ex$node_feature[2, 2], (f(xp) - f(xm)) / 2e-2, tolerance = 0.05)
  expect_output(print(ex), "saliency for node 1")
})

test_that("integrated gradients satisfy completeness", {
  skip_if_no_torch()
  s <- explain_setup()
  ig <- explain_integrated_gradients(s$fit, s$g, target = 3, steps = 64)
  ref <- abs(sum(ig$node_feature))
  expect_lt(abs(ig$completeness_gap), 0.05 * max(ref, 1e-3) + 1e-3)
})

test_that("edge gradients, occlusion and GNNExplainer return edge scores", {
  skip_if_no_torch()
  s <- explain_setup()
  eg <- explain_edges(s$fit, s$g, target = 1)
  expect_equal(nrow(eg$edges), num_edges(s$g))
  into <- eg$edges$to == 1
  expect_true(any(eg$edges$attribution[into] != 0))
  oc <- explain_occlusion(s$fit, s$g, target = 1)
  expect_true(all(oc$edges$to == 1))
  expect_length(oc$feature, 34)
  gx <- gnn_explainer(s$fit, s$g, target = 1, epochs = 20, seed = 1)
  expect_true(all(gx$edges$attribution >= 0 & gx$edges$attribution <= 1))
  expect_true(all(vapply(s$fit$model$parameters, function(p) p$requires_grad, logical(1))))
  fid <- explanation_fidelity(s$fit, s$g, eg, k = 5)
  expect_named(fid, c("fidelity_plus", "fidelity_minus"))
  st <- explanation_stability(s$fit, s$g, function(m, g) explain_edges(m, g, target = 1),
                              reps = 2, seed = 1)
  expect_gt(st[["mean_spearman"]], 0.5)
})

test_that("attention weights are collected from attention layers", {
  skip_if_no_torch()
  g <- split_nodes(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("gat", 34, 8, 2, heads = 2), g, "node_classification",
                           epochs = 3, seed = 1)
  invisible(predict(fit, g))
  aw <- attention_weights(fit)
  expect_length(aw, 2)
  expect_true(all(c("from", "to", "head1", "head2") %in% names(aw[[1]])))
  expect_equal(as.vector(tapply(aw[[1]]$head1, aw[[1]]$to, sum)), rep(1, 34), tolerance = 1e-5)
})
