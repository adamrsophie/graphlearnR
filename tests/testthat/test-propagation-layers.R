und_graph <- function(n, m, f, seed) {
  withr::with_seed(seed, {
    pairs <- utils::combn(n, 2)[, sample.int(n * (n - 1) / 2, m)]
    x <- matrix(stats::rnorm(n * f), n, f)
    w <- stats::runif(m, 0.5, 1.5)
  })
  graph_data(x = x, edge_index = pairs, edge_weight = w, num_nodes = n, directed = FALSE)
}

dense_in_adj <- function(g, weighted = TRUE) {
  # M[i, j] = total weight of edges j -> i
  n <- g$num_nodes
  w <- if (weighted && !is.null(g$edge_weight)) g$edge_weight else rep(1, ncol(g$edge_index))
  M <- matrix(0, n, n)
  for (e in seq_len(ncol(g$edge_index))) {
    M[g$edge_index[2, e], g$edge_index[1, e]] <- M[g$edge_index[2, e], g$edge_index[1, e]] + w[e]
  }
  M
}

test_that("cheb_conv matches the dense Chebyshev recursion", {
  skip_if_no_torch()
  torch::torch_manual_seed(1)
  g <- und_graph(8, 14, 3, seed = 2)
  layer <- cheb_conv(3, 2, K = 4)
  tt <- graph_tensors(g)
  out <- as_mat(layer(tt$x, tt$edge_index, tt$edge_weight))
  A <- dense_in_adj(g)
  diag(A) <- 0
  d <- rowSums(A)
  dinv <- ifelse(d > 0, 1 / sqrt(d), 0)
  L <- -diag(dinv) %*% A %*% diag(dinv)
  Tk <- list(g$x, L %*% g$x)
  for (k in 3:4) Tk[[k]] <- 2 * L %*% Tk[[k - 1]] - Tk[[k - 2]]
  ref <- Reduce(`+`, lapply(1:4, function(k) Tk[[k]] %*% t(as_mat(layer$lins[[k]]$weight))))
  ref <- sweep(ref, 2, as.numeric(as.array(layer$bias$detach())), `+`)
  expect_equal(out, ref, tolerance = 1e-5)
  # K = 1 is a plain linear layer
  l1 <- cheb_conv(3, 2, K = 1, bias = FALSE)
  expect_equal(as_mat(l1(tt$x, tt$edge_index)), g$x %*% t(as_mat(l1$lins[[1]]$weight)), tolerance = 1e-6)
})

test_that("appnp converges to the personalized PageRank solution", {
  skip_if_no_torch()
  g <- und_graph(7, 12, 2, seed = 3)
  g$edge_weight <- NULL
  tt <- graph_tensors(g, dtype = torch::torch_double())
  nrm <- gcn_norm(tt$edge_index, 7, dtype = torch::torch_double())
  ei <- as_mat(nrm$edge_index)
  w <- as.numeric(as.array(nrm$edge_weight))
  Ahat <- matrix(0, 7, 7)
  for (e in seq_along(w)) Ahat[ei[2, e], ei[1, e]] <- Ahat[ei[2, e], ei[1, e]] + w[e]
  a <- 0.2
  exact <- a * solve(diag(7) - (1 - a) * Ahat, g$x)
  got <- as_mat(appnp(K = 200, alpha = a)(tt$x, tt$edge_index))
  expect_equal(got, exact, tolerance = 1e-8)
  # K = 1 is one step of the iteration
  one <- as_mat(appnp(K = 1, alpha = a)(tt$x, tt$edge_index))
  expect_equal(one, (1 - a) * Ahat %*% g$x + a * g$x, tolerance = 1e-10)
  expect_error(appnp(alpha = 2), "alpha")
})

test_that("appnp_model trains on a node task", {
  skip_if_no_torch()
  testthat::skip_on_cran()  # trains for many epochs
  g <- split_nodes(dataset_sbm(c(40, 40), seed = 4), seed = 4)
  torch::torch_manual_seed(1)
  fit <- train_graph_model(appnp_model(ncol(g$x), 16, 2, K = 5), g, "node_classification",
                           epochs = 60, lr = 0.01, seed = 1)
  ev <- evaluate_graph_model(fit, g)
  expect_gt(ev$value[ev$metric == "accuracy"], 0.8)
})

test_that("pna_conv matches an explicit per node computation", {
  skip_if_no_torch()
  torch::torch_manual_seed(2)
  g <- random_graph(6, 12, 2, seed = 5)
  ea <- matrix(withr::with_seed(6, stats::rnorm(24)), 12, 2)
  deg <- pna_degree(g)
  expect_equal(sum(deg), 6L)
  expect_equal(sum((seq_along(deg) - 1) * deg), 12)
  aggs <- c("mean", "min", "max", "std", "sum")
  scs <- c("identity", "amplification", "attenuation", "linear", "inverse_linear")
  layer <- pna_conv(2, 3, deg = deg, aggregators = aggs, scalers = scs, edge_dim = 2)
  tt <- graph_tensors(g)
  out <- as_mat(layer(tt$x, tt$edge_index, edge_attr = torch::torch_tensor(ea)))
  lin <- function(l, v) as.vector(as_mat(l$weight) %*% v + as.numeric(as.array(l$bias$detach())))
  bins <- seq_along(deg) - 1
  avg_lin <- sum(bins * deg) / sum(deg)
  avg_log <- sum(log(bins + 1) * deg) / sum(deg)
  ref <- t(vapply(1:6, function(i) {
    inc <- which(g$edge_index[2, ] == i)
    msgs <- vapply(inc, function(e) {
      j <- g$edge_index[1, e]
      lin(layer$pre, c(g$x[i, ], g$x[j, ], lin(layer$edge_encoder, ea[e, ])))
    }, numeric(2))
    msgs <- matrix(msgs, nrow = 2)
    ag <- if (length(inc)) {
      c(rowMeans(msgs), apply(msgs, 1, min), apply(msgs, 1, max),
        {sd <- sqrt(pmax(rowMeans(msgs^2) - rowMeans(msgs)^2, 1e-5)); ifelse(sd <= sqrt(1e-5), 0, sd)}, rowSums(msgs))
    } else {
      c(rep(0, 10))
    }
    d <- max(length(inc), 1)
    sc <- c(ag, ag * log(d + 1) / avg_log, ag * avg_log / log(d + 1), ag * d / avg_lin, ag * avg_lin / d)
    lin(layer$lin, lin(layer$post, c(g$x[i, ], sc)))
  }, numeric(3)))
  expect_equal(out, ref, tolerance = 1e-4)
  expect_error(pna_conv(2, 3, deg = deg, aggregators = "median"), "aggregators")
  expect_error(pna_conv(2, 3, deg = deg, edge_dim = 2)(tt$x, tt$edge_index), "edge_attr")
})

test_that("sag_pool keeps the top scored nodes per graph using neighborhood scores", {
  skip_if_no_torch()
  torch::torch_manual_seed(3)
  gs <- list(random_graph(5, 8, 3, seed = 1), random_graph(4, 6, 3, seed = 2))
  b <- graph_batch(gs)
  tt <- graph_tensors(b)
  pool <- sag_pool(3, ratio = 0.5)
  r <- pool(tt$x, tt$edge_index, tt$batch)
  score <- as.numeric(as.array(pool$gnn(tt$x, tt$edge_index)$detach()))
  expect_equal(as.numeric(as.array(r$score$detach())), score, tolerance = 1e-6)
  keep1 <- order(-score[1:5])[1:3]
  keep2 <- 5 + order(-score[6:9])[1:2]
  expect_equal(r$perm, c(keep1, keep2))
  expect_equal(as_mat(r$x), b$x[r$perm, ] * tanh(score[r$perm]), tolerance = 1e-5)
  expect_equal(as.integer(as.array(r$batch)), c(1L, 1L, 1L, 2L, 2L))
  kept <- as_mat(r$edge_index)
  old <- matrix(r$perm[kept], nrow = 2)
  ok <- b$edge_index[1, ] %in% r$perm & b$edge_index[2, ] %in% r$perm
  expect_equal(ncol(old), sum(ok))
  # scores depend on neighbors, unlike topk_pool
  for (gnn in c("gcn", "sage")) expect_length(sag_pool(3, gnn = gnn)(tt$x, tt$edge_index, tt$batch)$perm, 5)
  # the pooled features are differentiable with respect to the scoring layer
  r$x$sum()$backward()
  expect_true(!is.null(pool$gnn$lin_rel$weight$grad))
})

test_that("pna_conv rejects a degree histogram without edges", {
  skip_if_no_torch()
  expect_error(pna_conv(2, 3, deg = c(5, 0, 0)), "degree zero")
})
