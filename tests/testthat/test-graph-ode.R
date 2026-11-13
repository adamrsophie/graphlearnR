test_that("gcn diffusion block converges to the matrix exponential", {
  skip_if_no_torch()
  g <- random_graph(6, 8, 3, seed = 1)
  g$edge_weight <- NULL
  g <- as_undirected(g)
  tt <- graph_tensors(g, dtype = torch::torch_double())
  A <- matrix(0, 6, 6)
  A[cbind(g$edge_index[2, ], g$edge_index[1, ])] <- 1
  diag(A) <- 1
  d <- rowSums(A)
  M <- diag(1 / sqrt(d)) %*% A %*% diag(1 / sqrt(d)) - diag(6)
  e <- eigen(M, symmetric = TRUE)
  ref <- e$vectors %*% (exp(1.5 * e$values) * crossprod(e$vectors, g$x))
  rk <- graph_ode_block(3, t_end = 1.5, steps = 20)
  eu <- graph_ode_block(3, t_end = 1.5, steps = 20, method = "euler")
  err_rk <- max(abs(as_mat(rk(tt$x, tt$edge_index)) - ref))
  err_eu <- max(abs(as_mat(eu(tt$x, tt$edge_index)) - ref))
  expect_lt(err_rk, 1e-5)
  expect_lt(err_eu, 0.05)
  expect_lt(err_rk, err_eu)
})

test_that("attention diffusion is row stochastic, equivariant and differentiable", {
  skip_if_no_torch()
  torch::torch_manual_seed(2)
  g <- random_graph(7, 14, 4, seed = 2)
  b <- graph_ode_block(4, diffusion = "attention", t_end = 1, steps = 3)
  b$to(dtype = torch::torch_double())
  tt <- graph_tensors(g, dtype = torch::torch_double())
  op <- b$operator(tt$x, tt$edge_index)
  rs <- as.vector(as.array(scatter_sum(op$edge_weight, op$edge_index[2, ], 7)$detach()))
  expect_equal(rs, rep(1, 7), tolerance = 1e-10)
  # constant features are a fixed point of (A - I) X with row stochastic A
  one <- torch::torch_ones(7, 4, dtype = torch::torch_double())
  expect_equal(as_mat(b(one, tt$edge_index)), matrix(1, 7, 4), tolerance = 1e-10)
  perm <- withr::with_seed(3, sample.int(7))
  gp <- permute_graph(g, perm)
  o1 <- as_mat(b(tt$x, tt$edge_index))
  tp <- graph_tensors(gp, dtype = torch::torch_double())
  expect_equal(as_mat(b(tp$x, tp$edge_index)), o1[perm, ], tolerance = 1e-8)
  x <- tt$x$clone()$requires_grad_(TRUE)
  b(x, tt$edge_index)$pow(2)$sum()$backward()
  base <- g$x
  num <- vapply(c(1, 9, 20), function(k) {
    p <- base; m <- base
    p[k] <- p[k] + 1e-6; m[k] <- m[k] - 1e-6
    f <- function(v) as.numeric(b(torch::torch_tensor(v, dtype = torch::torch_double()),
                                    tt$edge_index)$pow(2)$sum()$item())
    (f(p) - f(m)) / 2e-6
  }, numeric(1))
  expect_equal(as.vector(as_mat(x$grad))[c(1, 9, 20)], num, tolerance = 1e-6)
})

test_that("graph_ode_model trains on a node task", {
  skip_if_no_torch()
  testthat::skip_on_cran()  # trains for many epochs
  g <- split_nodes(dataset_sbm(c(50, 50), seed = 5), seed = 5)
  torch::torch_manual_seed(1)
  m <- graph_ode_model(ncol(g$x), 16, 2, t_end = 2, steps = 4)
  fit <- train_graph_model(m, g, "node_classification", epochs = 60, seed = 1)
  ev <- evaluate_graph_model(fit, g)
  expect_gt(ev$value[ev$metric == "accuracy"], 0.8)
})
