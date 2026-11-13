test_that("Laplacian eigenvectors satisfy the eigen equation and are orthonormal", {
  g <- dataset_karate(features = "none")
  pe <- laplacian_pe(g, 4)
  A <- as.matrix(adjacency_from_graph(g))
  d <- rowSums(A)
  L <- diag(34) - diag(1 / sqrt(d)) %*% A %*% diag(1 / sqrt(d))
  lam <- colSums(pe * (L %*% pe))
  expect_equal(L %*% pe, sweep(pe, 2, lam, "*"), tolerance = 1e-6)
  expect_equal(crossprod(pe), diag(4), tolerance = 1e-6)
  expect_true(all(lam > 1e-8))
  small <- laplacian_pe(graph_data(edge_index = rbind(1, 2), directed = FALSE), 3)
  expect_equal(dim(small), c(2, 3))
  expect_equal(small[, 2:3], matrix(0, 2, 2))
})

test_that("random walk encodings are permutation equivariant", {
  g <- random_graph(10, 25, 1, seed = 1)
  perm <- withr::with_seed(2, sample.int(10))
  gp <- permute_graph(g, perm)
  expect_equal(random_walk_pe(gp, 4), random_walk_pe(g, 4)[perm, ], tolerance = 1e-10)
  D <- shortest_path_distances(g, 3)
  expect_equal(shortest_path_distances(gp, 3), D[perm, perm])
  ring <- graph_data(edge_index = rbind(1:6, c(2:6, 1)), directed = FALSE)
  expect_equal(shortest_path_distances(ring)[1, ], c(0, 1, 2, 3, 2, 1))
  iso <- graph_data(edge_index = rbind(1, 2), num_nodes = 3, directed = FALSE)
  expect_equal(shortest_path_distances(iso, max_dist = 4)[1, 3], 5L)
  aug <- add_positional_encoding(ring, "random_walk", 3)
  expect_equal(ncol(aug$x), 3)
  flipped <- random_sign_flip(laplacian_pe(ring, 2), seed = 1)
  expect_equal(abs(flipped), abs(laplacian_pe(ring, 2)))
})

test_that("dense batch layout round trips", {
  skip_if_no_torch()
  x <- torch::torch_randn(7, 3)
  batch <- c(1L, 1L, 2L, 2L, 2L, 2L, 3L)
  d <- to_dense_batch(x, batch)
  expect_equal(dim(d$x), c(3, 4, 3))
  expect_equal(as.logical(as.array(d$mask$sum(2) )), c(TRUE, TRUE, TRUE))
  expect_equal(as_mat(from_dense_batch(d)), as_mat(x))
})

test_that("graph transformer layers are equivariant and never mix graphs", {
  skip_if_no_torch()
  torch::torch_manual_seed(1)
  g <- random_graph(8, 16, 4, seed = 3)
  perm <- withr::with_seed(4, sample.int(8))
  gp <- permute_graph(g, perm)
  m <- graph_transformer_model(4, 8, 2, num_layers = 2, heads = 2, pool = NULL, dropout = 0)
  m$eval()
  o1 <- as_mat(graphlearnR:::model_forward(m, graph_tensors(g)))
  o2 <- as_mat(graphlearnR:::model_forward(m, graph_tensors(gp)))
  expect_equal(o2, o1[perm, ], tolerance = 1e-4)
  gs <- list(g, random_graph(5, 8, 4, seed = 5), random_graph(3, 4, 4, seed = 6))
  b <- graph_batch(gs)
  os <- do.call(rbind, lapply(gs, function(x) as_mat(graphlearnR:::model_forward(m, graph_tensors(x)))))
  ob <- as_mat(graphlearnR:::model_forward(m, graph_tensors(b)))
  expect_equal(ob, os, tolerance = 1e-4)
  att <- m$layers[[1]]$last_attention
  expect_equal(dim(att), c(3, 2, 8, 8))
  expect_equal(as.numeric(as.array(att[3, 1, 1, 4:8]$sum())), 0, tolerance = 1e-6)
})

test_that("pooled graph transformers are invariant and train on graph classes", {
  skip_if_no_torch()
  torch::torch_manual_seed(2)
  g <- random_graph(7, 14, 1, seed = 7)
  perm <- withr::with_seed(8, sample.int(7))
  m <- graph_transformer_model(1, 8, 2, heads = 2, dropout = 0)
  m$eval()
  expect_equal(as_mat(graphlearnR:::model_forward(m, graph_tensors(permute_graph(g, perm)))),
               as_mat(graphlearnR:::model_forward(m, graph_tensors(g))), tolerance = 1e-4)
  gs <- dataset_synthetic_graphs(40, seed = 9)
  fit <- train_graph_model(graph_transformer_model(1, 16, 2, heads = 2, dropout = 0), gs,
                           "graph_classification", epochs = 15, batch_size = 10, seed = 9)
  ev <- evaluate_graph_model(fit, gs)
  expect_gt(ev$value[ev$metric == "accuracy"], 0.8)
})

test_that("laplacian_pe skips every zero eigenvalue of a disconnected graph", {
  g <- graph_data(edge_index = rbind(c(1, 3, 5), c(2, 4, 6)), num_nodes = 6, directed = FALSE)
  pe <- laplacian_pe(g, 2)
  A <- as.matrix(adjacency_from_graph(g, weighted = FALSE))
  L <- diag(6) - A  # every degree is 1
  for (j in 1:2) {
    lam <- sum(pe[, j] * (L %*% pe[, j]))
    expect_gt(lam, 1e-6)
  }
})

test_that("edge biased attention shifts only the target's attention row by w'e", {
  skip_if_no_torch()
  torch::torch_manual_seed(1)
  g <- random_graph(n = 6, m = 10, f = 4, seed = 5)
  E <- ncol(g$edge_index)
  ea <- matrix(rnorm(E * 2), E, 2)
  layer <- graph_transformer_layer(4, heads = 1, dropout = 0, edge_dim = 2)
  layer$eval()
  tt <- graph_tensors(g)
  batch <- torch::torch_ones(6, dtype = torch::torch_long())
  dist <- graphlearnR:::batch_distances(tt$edge_index, batch, 5L, tt$x$device)
  run <- function(a) {
    out <- as_mat(layer(tt$x, batch, dist, tt$edge_index, torch::torch_tensor(a)))
    list(out = out, att = as.array(layer$last_attention)[1, 1, , ])
  }
  r0 <- run(ea)
  # change the features of one edge j -> i (a non duplicated pair)
  key <- paste(g$edge_index[1, ], g$edge_index[2, ])
  e <- which(!duplicated(key) & !duplicated(key, fromLast = TRUE) &
               g$edge_index[1, ] != g$edge_index[2, ])[1]
  j <- g$edge_index[1, e]; i <- g$edge_index[2, e]
  ea2 <- ea; ea2[e, ] <- ea[e, ] + c(1, -2)
  r1 <- run(ea2)
  expect_equal(r1$out[-i, ], r0$out[-i, ], tolerance = 1e-6)
  expect_false(isTRUE(all.equal(r1$out[i, ], r0$out[i, ])))
  w <- as.numeric(as.array(layer$edge_bias$weight$detach()))
  k <- setdiff(seq_len(6), j)[1]
  ratio <- (r1$att[i, j] / r1$att[i, k]) / (r0$att[i, j] / r0$att[i, k])
  expect_equal(ratio, exp(sum(w * c(1, -2))), tolerance = 1e-4)
  expect_error(layer(tt$x, batch, dist), "edge_dim")
})

test_that("graph_transformer_model uses edge features when edge_dim is set", {
  skip_if_no_torch()
  gs <- dataset_synthetic_graphs(12, seed = 1)
  gs <- lapply(gs, function(g) { g$edge_attr <- matrix(1, num_edges(g), 3); g })
  torch::torch_manual_seed(1)
  m <- graph_transformer_model(ncol(gs[[1]]$x), 8, 2, heads = 2, edge_dim = 3)
  fit <- train_graph_model(m, gs, "graph_classification", epochs = 2, batch_size = 6, seed = 1)
  expect_s3_class(fit, "graph_fit")
  bt <- graph_tensors(graph_batch(gs[1:2]))
  m$eval()
  o1 <- as_mat(graphlearnR:::model_forward(m, bt))
  bt$edge_attr <- bt$edge_attr * 2
  o2 <- as_mat(graphlearnR:::model_forward(m, bt))
  expect_false(isTRUE(all.equal(o1, o2)))
})
