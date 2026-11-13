layer_cases <- function(f_in, f_out) {
  list(
    sage = function() sage_conv(f_in, f_out),
    sage_max = function() sage_conv(f_in, f_out, aggr = "max"),
    gcn = function() gcn_conv(f_in, f_out),
    graph_conv = function() graph_conv(f_in, f_out),
    gin = function() gin_conv(graph_mlp(c(f_in, 4, f_out))),
    gat = function() gat_conv(f_in, f_out, heads = 2),
    gatv2 = function() gatv2_conv(f_in, f_out, heads = 2, concat = FALSE),
    transformer = function() transformer_conv(f_in, f_out, heads = 2),
    cheb = function() cheb_conv(f_in, f_out, K = 3),
    pna = function() pna_conv(f_in, f_out, deg = c(1, 3, 4, 2),
                              scalers = c("identity", "amplification", "attenuation", "linear"))
  )
}

test_that("all layers are permutation equivariant", {
  skip_if_no_torch()
  torch::torch_manual_seed(1)
  g <- random_graph(n = 9, m = 20, f = 3, seed = 4)
  perm <- withr::with_seed(5, sample.int(9))
  gp <- permute_graph(g, perm)
  for (nm in names(layer_cases(3, 4))) {
    layer <- layer_cases(3, 4)[[nm]]()
    layer$eval()
    t1 <- graph_tensors(g)
    t2 <- graph_tensors(gp)
    o1 <- as_mat(layer(t1$x, t1$edge_index))
    o2 <- as_mat(layer(t2$x, t2$edge_index))
    expect_equal(o2, o1[perm, , drop = FALSE], tolerance = 1e-5, info = nm)
  }
})

test_that("edge feature and relation layers are permutation equivariant", {
  skip_if_no_torch()
  torch::torch_manual_seed(2)
  g <- random_graph(n = 7, m = 15, f = 3, seed = 6)
  g$edge_attr <- matrix(withr::with_seed(1, rnorm(30)), 15, 2)
  g$edge_type <- withr::with_seed(2, sample(1:3, 15, TRUE))
  perm <- withr::with_seed(7, sample.int(7))
  gp <- permute_graph(g, perm)
  t1 <- graph_tensors(g)
  t2 <- graph_tensors(gp)
  layers <- list(
    nn = nn_conv(3, 4, edge_dim = 2),
    gine = gin_conv(graph_mlp(c(3, 4)), edge_dim = 2, in_features = 3),
    gat_e = gat_conv(3, 2, heads = 2, edge_dim = 2),
    tr_e = transformer_conv(3, 2, heads = 2, edge_dim = 2)
  )
  for (nm in names(layers)) {
    l <- layers[[nm]]
    o1 <- as_mat(l(t1$x, t1$edge_index, edge_attr = t1$edge_attr))
    o2 <- as_mat(l(t2$x, t2$edge_index, edge_attr = t2$edge_attr))
    expect_equal(o2, o1[perm, , drop = FALSE], tolerance = 1e-5, info = nm)
  }
  r <- rgcn_conv(3, 4, num_relations = 3, num_bases = 2)
  o1 <- as_mat(r(t1$x, t1$edge_index, edge_type = t1$edge_type))
  o2 <- as_mat(r(t2$x, t2$edge_index, edge_type = t2$edge_type))
  expect_equal(o2, o1[perm, , drop = FALSE], tolerance = 1e-5)
})

test_that("gcn_conv equals the dense Kipf and Welling formula", {
  skip_if_no_torch()
  g <- random_graph(n = 6, m = 10, f = 3, seed = 9)
  g$edge_weight <- NULL
  g <- coalesce_edges(remove_self_loops(g))
  layer <- gcn_conv(3, 2)
  tt <- graph_tensors(g)
  W <- t(as_mat(layer$lin$weight))
  A <- t(as.matrix(adjacency_from_graph(g, weighted = FALSE))) + diag(6)
  d <- rowSums(A)
  ref <- diag(1 / sqrt(d)) %*% A %*% diag(1 / sqrt(d)) %*% g$x %*% W
  expect_equal(as_mat(layer(tt$x, tt$edge_index)), ref, tolerance = 1e-5)
})

test_that("sage_conv equals mean of neighbors plus root", {
  skip_if_no_torch()
  g <- random_graph(n = 6, m = 12, f = 2, seed = 10)
  layer <- sage_conv(2, 3)
  tt <- graph_tensors(g)
  Wl <- t(as_mat(layer$lin_l$weight))
  bl <- as.numeric(as.array(layer$lin_l$bias))
  Wr <- t(as_mat(layer$lin_r$weight))
  agg <- t(vapply(1:6, function(i) {
    src <- g$edge_index[1, g$edge_index[2, ] == i]
    if (!length(src)) c(0, 0) else colMeans(g$x[src, , drop = FALSE])
  }, numeric(2)))
  ref <- sweep(agg %*% Wl, 2, bl, "+") + g$x %*% Wr
  expect_equal(as_mat(layer(tt$x, tt$edge_index)), ref, tolerance = 1e-5)
})

test_that("gin_conv equals the MLP of (1 + eps) x plus neighbor sum", {
  skip_if_no_torch()
  g <- random_graph(n = 5, m = 8, f = 2, seed = 11)
  lin <- torch::nn_linear(2, 2)
  layer <- gin_conv(lin, eps = 0.3)
  tt <- graph_tensors(g)
  s <- t(vapply(1:5, function(i) {
    src <- g$edge_index[1, g$edge_index[2, ] == i]
    colSums(g$x[src, , drop = FALSE])
  }, numeric(2)))
  h <- 1.3 * g$x + s
  ref <- sweep(h %*% t(as_mat(lin$weight)), 2, as.numeric(as.array(lin$bias)), "+")
  expect_equal(as_mat(layer(tt$x, tt$edge_index)), ref, tolerance = 1e-5)
})

test_that("attention coefficients sum to one over incoming edges", {
  skip_if_no_torch()
  g <- random_graph(n = 6, m = 14, f = 3, seed = 12)
  tt <- graph_tensors(g)
  for (layer in list(gat_conv(3, 2, heads = 3), gatv2_conv(3, 2, heads = 3),
                     transformer_conv(3, 2, heads = 3))) {
    layer(tt$x, tt$edge_index)
    a <- as_mat(layer$last_attention$alpha)
    dst <- as_mat(layer$last_attention$edge_index)[2, ]
    sums <- rowsum(a, dst)
    expect_equal(unname(sums), matrix(1, nrow(sums), 3), tolerance = 1e-5)
  }
})

test_that("rgcn_conv equals per relation means plus root", {
  skip_if_no_torch()
  g <- random_graph(n = 5, m = 12, f = 2, seed = 13)
  g$edge_type <- withr::with_seed(3, sample(1:2, 12, TRUE))
  layer <- rgcn_conv(2, 3, num_relations = 2)
  tt <- graph_tensors(g)
  W <- as.array(layer$weight$detach())
  ref <- g$x %*% t(as_mat(layer$lin_root$weight)) + matrix(as.numeric(as.array(layer$bias)), 5, 3, byrow = TRUE)
  for (r in 1:2) {
    agg <- t(vapply(1:5, function(i) {
      src <- g$edge_index[1, g$edge_index[2, ] == i & g$edge_type == r]
      if (!length(src)) c(0, 0) else colMeans(g$x[src, , drop = FALSE])
    }, numeric(2)))
    ref <- ref + agg %*% W[r, , ]
  }
  expect_equal(as_mat(layer(tt$x, tt$edge_index, edge_type = tt$edge_type)), ref, tolerance = 1e-5)
})

numeric_grad_check <- function(make_layer, g, n_check = 6, eps = 1e-6) {
  torch::torch_manual_seed(21)
  layer <- make_layer()
  layer$to(dtype = torch::torch_double())
  tt <- graph_tensors(g, dtype = torch::torch_double())
  R <- torch::torch_randn(g$num_nodes, 3, dtype = torch::torch_double())
  x <- tt$x$clone()$requires_grad_(TRUE)
  loss <- function(xx) (layer(xx, tt$edge_index, edge_weight = tt$edge_weight) * R)$sum()
  loss(x)$backward()
  analytic <- as_mat(x$grad)
  base <- as_mat(tt$x)
  idx <- withr::with_seed(8, sample(length(base), n_check))
  numeric <- vapply(idx, function(k) {
    xp <- base
    xm <- base
    xp[k] <- xp[k] + eps
    xm[k] <- xm[k] - eps
    f <- function(m) as.numeric(loss(torch::torch_tensor(m, dtype = torch::torch_double()))$item())
    (f(xp) - f(xm)) / (2 * eps)
  }, numeric(1))
  list(analytic = analytic[idx], numeric = numeric, layer = layer, tt = tt, R = R, loss = loss)
}

test_that("input gradients match central finite differences", {
  skip_if_no_torch()
  g <- random_graph(n = 7, m = 14, f = 2, seed = 14)
  makers <- list(
    sage = function() sage_conv(2, 3),
    gcn = function() gcn_conv(2, 3),
    graph_conv = function() graph_conv(2, 3),
    gat = function() gat_conv(2, 3, heads = 1),
    gin = function() gin_conv(graph_mlp(c(2, 3), activation = torch::nn_tanh)),
    cheb = function() cheb_conv(2, 3, K = 3),
    pna = function() pna_conv(2, 3, deg = c(1, 3, 4, 2), aggregators = c("mean", "sum", "std"))
  )
  for (nm in names(makers)) {
    r <- numeric_grad_check(makers[[nm]], g)
    expect_equal(r$analytic, r$numeric, tolerance = 1e-6, info = nm)
  }
})

test_that("parameter gradients of gcn_conv match the analytic formula", {
  skip_if_no_torch()
  g <- random_graph(n = 6, m = 10, f = 2, seed = 15)
  g$edge_weight <- NULL
  layer <- gcn_conv(2, 3)
  layer$to(dtype = torch::torch_double())
  tt <- graph_tensors(g, dtype = torch::torch_double())
  R <- torch::torch_randn(6, 3, dtype = torch::torch_double())
  (layer(tt$x, tt$edge_index) * R)$sum()$backward()
  nrm <- gcn_norm(tt$edge_index, 6)
  ei <- as_mat(nrm$edge_index)
  M <- matrix(0, 6, 6)
  for (j in seq_len(ncol(ei))) M[ei[2, j], ei[1, j]] <- M[ei[2, j], ei[1, j]] + as.numeric(as.array(nrm$edge_weight))[j]
  # out = M X W^T + b, so dL/dW = R^T M X and dL/db = colSums(R)
  expect_equal(as_mat(layer$lin$weight$grad), t(as_mat(R)) %*% M %*% g$x, tolerance = 1e-8)
  expect_equal(as.numeric(as.array(layer$bias$grad)), colSums(as_mat(R)), tolerance = 1e-10)
})

test_that("layers accept bipartite input with different feature sizes", {
  skip_if_no_torch()
  xs <- torch::torch_randn(4, 3)
  xd <- torch::torch_randn(2, 5)
  ei <- torch::torch_tensor(rbind(c(1, 2, 4), c(1, 2, 2)), dtype = torch::torch_long())
  expect_equal(dim(sage_conv(c(3, 5), 6)(list(xs, xd), ei)), c(2, 6))
  expect_equal(dim(graph_conv(c(3, 5), 6)(list(xs, xd), ei)), c(2, 6))
  expect_equal(dim(gat_conv(c(3, 5), 2, heads = 2)(list(xs, xd), ei)), c(2, 4))
  expect_equal(dim(gatv2_conv(c(3, 5), 2, heads = 2)(list(xs, xd), ei)), c(2, 4))
  expect_equal(dim(transformer_conv(c(3, 5), 2, heads = 2)(list(xs, xd), ei)), c(2, 4))
})

test_that("layers report missing required inputs", {
  skip_if_no_torch()
  x <- torch::torch_randn(3, 2)
  ei <- torch::torch_tensor(rbind(1:2, 2:3), dtype = torch::torch_long())
  expect_error(nn_conv(2, 2, edge_dim = 1)(x, ei), "edge_attr")
  expect_error(rgcn_conv(2, 2, 2)(x, ei), "edge_type")
  expect_error(nn_conv(2, 2), "edge_nn")
  expect_error(gin_conv(graph_mlp(c(2, 2)), edge_dim = 1), "in_features")
})
