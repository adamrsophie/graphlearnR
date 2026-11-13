ref_aggregate <- function(src, index, n, fun, empty = 0) {
  t(vapply(seq_len(n), function(i) {
    rows <- src[index == i, , drop = FALSE]
    if (!nrow(rows)) rep(empty, ncol(src)) else apply(rows, 2, fun)
  }, numeric(ncol(src))))
}

test_that("scatter kernels match reference loops", {
  skip_if_no_torch()
  withr::local_seed(3)
  src <- matrix(rnorm(24), 8, 3)
  index <- c(1L, 1L, 3L, 3L, 3L, 5L, 1L, 5L)
  n <- 6L
  s <- torch::torch_tensor(src)
  expect_equal(as_mat(scatter_sum(s, index, n)), ref_aggregate(src, index, n, sum), tolerance = 1e-5)
  expect_equal(as_mat(scatter_mean(s, index, n)), ref_aggregate(src, index, n, mean), tolerance = 1e-5)
  expect_equal(as_mat(scatter_max(s, index, n)), ref_aggregate(src, index, n, max), tolerance = 1e-5)
  expect_equal(as_mat(scatter_min(s, index, n)), ref_aggregate(src, index, n, min), tolerance = 1e-5)
  sdp <- function(v) sqrt(mean((v - mean(v))^2) + 1e-5)
  expect_equal(as_mat(scatter_std(s, index, n)), ref_aggregate(src, index, n, sdp, sqrt(1e-5)),
               tolerance = 1e-4)
  both <- scatter_aggregate(s, index, n, c("sum", "max"))
  expect_equal(dim(both), c(6, 6))
  expect_equal(as.numeric(as.array(scatter_count(index, n))), c(3, 0, 3, 0, 2, 0))
  expect_error(scatter_aggregate(s, index, n, "median"), "Unknown aggregation")
})

test_that("scatter_softmax normalizes within groups", {
  skip_if_no_torch()
  s <- torch::torch_tensor(matrix(c(1, 2, 3, 100, 101, 0), 6, 1))
  index <- c(1L, 1L, 1L, 2L, 2L, 3L)
  p <- as_mat(scatter_softmax(s, index, 3))[, 1]
  expect_equal(as.vector(tapply(p, index, sum)), c(1, 1, 1), tolerance = 1e-6)
  expect_equal(p[1:3], exp(1:3) / sum(exp(1:3)), tolerance = 1e-6)
  expect_equal(p[4:5], exp(c(0, 1)) / sum(exp(c(0, 1))), tolerance = 1e-6)
})

test_that("message passing handles isolated nodes, repeats, loops and weights", {
  skip_if_no_torch()
  x <- matrix(c(1, 10, 100, 1000), 4, 1)
  ei <- rbind(c(1, 1, 2, 3, 3), c(2, 2, 2, 3, 1))
  w <- c(1, 2, 0.5, 1, 3)
  xt <- torch::torch_tensor(x)
  et <- torch::torch_tensor(ei, dtype = torch::torch_long())
  out_sum <- as_mat(message_passing(xt, et, aggr = "sum"))[, 1]
  expect_equal(out_sum, c(100, 1 + 1 + 10, 100, 0))
  out_w <- as_mat(message_passing(xt, et, aggr = "sum", edge_weight = torch::torch_tensor(w)))[, 1]
  expect_equal(out_w, c(300, 1 + 2 + 5, 100, 0))
  out_mean <- as_mat(message_passing(xt, et, aggr = "mean"))[, 1]
  expect_equal(out_mean, c(100, 4, 100, 0))
  out_max <- as_mat(message_passing(xt, et, aggr = "max"))[, 1]
  expect_equal(out_max, c(100, 10, 100, 0))
})

test_that("bipartite message passing indexes source and target separately", {
  skip_if_no_torch()
  xs <- torch::torch_tensor(matrix(c(1, 2, 3), 3, 1))
  xd <- torch::torch_zeros(2, 1)
  ei <- torch::torch_tensor(rbind(c(1, 3, 2), c(2, 2, 1)), dtype = torch::torch_long())
  out <- as_mat(message_passing(list(xs, xd), ei, aggr = "sum"))[, 1]
  expect_equal(out, c(2, 4))
})

test_that("gradients through scatter sum and mean are analytic", {
  skip_if_no_torch()
  src <- torch::torch_randn(5, 2, requires_grad = TRUE)
  index <- c(1L, 1L, 2L, 2L, 2L)
  scatter_mean(src, index, 3)$sum()$backward()
  expect_equal(as_mat(src$grad), matrix(c(0.5, 0.5, 1 / 3, 1 / 3, 1 / 3), 5, 2), tolerance = 1e-6)
  src2 <- torch::torch_randn(5, 2, requires_grad = TRUE)
  scatter_sum(src2, index, 3)$sum()$backward()
  expect_equal(as_mat(src2$grad), matrix(1, 5, 2))
  src3 <- torch::torch_tensor(matrix(c(1, 5, 2, 7, 3), 5, 1), requires_grad = TRUE)
  scatter_max(src3, index, 2)$sum()$backward()
  expect_equal(as_mat(src3$grad)[, 1], c(0, 1, 0, 1, 0))
})

test_that("gcn_norm matches the dense symmetric normalization", {
  skip_if_no_torch()
  g <- dataset_karate(features = "none")
  A <- as.matrix(adjacency_from_graph(g)) + diag(34)
  d <- rowSums(A)
  N <- diag(1 / sqrt(d)) %*% A %*% diag(1 / sqrt(d))
  tt <- graph_tensors(g)
  nrm <- gcn_norm(tt$edge_index, 34)
  ei <- as_mat(nrm$edge_index)
  w <- as.numeric(as.array(nrm$edge_weight))
  M <- matrix(0, 34, 34)
  M[cbind(ei[2, ], ei[1, ])] <- w
  expect_equal(M, N, tolerance = 1e-5)
})

test_that("self loop helpers add remaining loops only", {
  skip_if_no_torch()
  ei <- torch::torch_tensor(rbind(c(1, 2, 2), c(2, 2, 3)), dtype = torch::torch_long())
  sl <- add_self_loops_tensor(ei, 3, remaining = TRUE)
  m <- as_mat(sl$edge_index)
  expect_equal(sum(m[1, ] == m[2, ]), 3)
  expect_equal(ncol(m), 5)
  rl <- remove_self_loops_tensor(sl$edge_index)
  expect_equal(ncol(as_mat(rl$edge_index)), 2)
})
