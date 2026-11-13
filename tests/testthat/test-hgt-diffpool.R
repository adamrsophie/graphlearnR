tiny_hetero <- function() {
  hetero_graph_data(
    nodes = list(a = matrix(c(1, 0, 0.5, -1, 2, 1), 3, 2), b = matrix(c(0.2, -0.3, 1, 0.4), 2, 2)),
    edges = list(a__r1__b = rbind(c(1, 2, 3), c(1, 1, 2)), b__r2__b = rbind(2, 1),
                 b__r3__a = rbind(c(1, 2), c(3, 3)))
  )
}

test_that("hgt_conv equals an explicit loop over incoming edges", {
  skip_if_no_torch()
  torch::torch_manual_seed(1)
  hg <- tiny_hetero()
  rels <- names(hg$edges)
  layer <- hgt_conv(hg$node_types, rels, 2, 2, heads = 1)
  layer$eval()
  ht <- hetero_tensors(hg)
  out <- layer(ht$x, ht$edge_index)
  W <- function(lin) list(w = as_mat(lin$weight), b = as.numeric(as.array(lin$bias$detach())))
  ap <- function(p, v) as.vector(p$w %*% v + p$b)
  gelu <- function(v) v * stats::pnorm(v)
  watt <- as.array(layer$w_att$detach())
  wmsg <- as.array(layer$w_msg$detach())
  mu <- as_mat(layer$mu)
  skip <- as.numeric(as.array(layer$skip$detach()))
  for (ti in seq_along(hg$node_types)) {
    d <- hg$node_types[ti]
    xd <- hg$nodes[[d]]$x
    ref <- t(vapply(seq_len(nrow(xd)), function(t) {
      sc <- numeric(0)
      msgs <- list()
      for (r in seq_along(rels)) {
        p <- strsplit(rels[r], "__")[[1]]
        if (p[3] != d) next
        ei <- hg$edges[[rels[r]]]$edge_index
        for (e in which(ei[2, ] == t)) {
          xs <- hg$nodes[[p[1]]]$x[ei[1, e], ]
          k <- ap(W(layer$k_lin[[p[1]]]), xs) %*% watt[r, 1, , ]
          q <- ap(W(layer$q_lin[[d]]), xd[t, ])
          sc <- c(sc, sum(k * q) * mu[r, 1] / sqrt(2))
          msgs[[length(msgs) + 1]] <- as.vector(ap(W(layer$v_lin[[p[1]]]), xs) %*% wmsg[r, 1, , ])
        }
      }
      agg <- if (length(sc)) {
        al <- exp(sc - max(sc)) / sum(exp(sc - max(sc)))
        Reduce(`+`, Map(`*`, al, msgs))
      } else c(0, 0)
      a <- ap(W(layer$a_lin[[d]]), gelu(agg))
      beta <- 1 / (1 + exp(-skip[ti]))
      beta * a + (1 - beta) * xd[t, ]
    }, numeric(2)))
    expect_equal(as_mat(out[[d]]), ref, tolerance = 1e-5, info = d)
  }
})

test_that("hgt attention sums to one per target across relations", {
  skip_if_no_torch()
  hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 2))
  ht <- hetero_tensors(hg)
  layer <- hgt_conv(hg$node_types, names(hg$edges), 6, 6, heads = 3)
  x <- list(drug = torch::torch_randn(60, 6), protein = torch::torch_randn(80, 6))
  invisible(layer(x, ht$edge_index))
  a <- layer$last_attention$protein
  s <- as_mat(scatter_sum(a$alpha, a$index, 80))
  has_in <- rowSums(s) > 0
  expect_equal(unname(s[has_in, ]), matrix(1, sum(has_in), 3), tolerance = 1e-5)
  expect_error(hgt_conv("a", character(), 4, 5, heads = 2), "divisible")
})

test_that("hgt_conv is equivariant to permuting nodes of one type", {
  skip_if_no_torch()
  torch::torch_manual_seed(3)
  hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 3))
  layer <- hgt_conv(hg$node_types, names(hg$edges), 4, 4, heads = 2)
  x <- list(drug = torch::torch_randn(60, 4), protein = torch::torch_randn(80, 4))
  o1 <- layer(x, hetero_tensors(hg)$edge_index)
  perm <- withr::with_seed(1, sample.int(80))
  inv <- order(perm)
  hp <- hg
  for (k in names(hp$edges)) {
    e <- hp$edges[[k]]$edge_index
    if (hp$edges[[k]]$src == "protein") e[1, ] <- inv[e[1, ]]
    if (hp$edges[[k]]$dst == "protein") e[2, ] <- inv[e[2, ]]
    hp$edges[[k]]$edge_index <- e
  }
  xp <- x
  xp$protein <- x$protein[perm, ]
  o2 <- layer(xp, hetero_tensors(hp)$edge_index)
  expect_equal(as_mat(o2$protein), as_mat(o1$protein)[perm, ], tolerance = 1e-5)
  expect_equal(as_mat(o2$drug), as_mat(o1$drug), tolerance = 1e-5)
})

test_that("hgt input gradients match finite differences", {
  skip_if_no_torch()
  torch::torch_manual_seed(4)
  hg <- tiny_hetero()
  layer <- hgt_conv(hg$node_types, names(hg$edges), 2, 2, heads = 2)
  layer$to(dtype = torch::torch_double())
  ei <- hetero_tensors(hg)$edge_index
  xb <- torch::torch_tensor(hg$nodes$b$x, dtype = torch::torch_double())
  f <- function(xa) {
    o <- layer(list(a = xa, b = xb), ei)
    o$a$sum() + (o$b * o$b)$sum()
  }
  base <- hg$nodes$a$x
  xa <- torch::torch_tensor(base, dtype = torch::torch_double(), requires_grad = TRUE)
  f(xa)$backward()
  g <- as_mat(xa$grad)
  num <- vapply(seq_along(base), function(k) {
    p <- base; m <- base
    p[k] <- p[k] + 1e-6; m[k] <- m[k] - 1e-6
    tt <- function(v) torch::torch_tensor(v, dtype = torch::torch_double())
    (as.numeric(f(tt(p))$item()) - as.numeric(f(tt(m))$item())) / 2e-6
  }, numeric(1))
  expect_equal(as.vector(g), num, tolerance = 1e-6)
})

test_that("hetero models with HGT layers train", {
  skip_if_no_torch()
  hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 1))
  torch::torch_manual_seed(1)
  m <- hetero_gnn_model(hg, hidden_dim = 16, output_dim = 3, target_type = "drug", conv = "hgt",
                        dropout = 0.1)
  fit <- train_graph_model(m, hg, "node_classification", target_type = "drug", epochs = 60,
                           lr = 0.01, seed = 1)
  ev <- evaluate_graph_model(fit, hg)
  expect_gt(ev$value[ev$metric == "accuracy"], 0.4)
  h <- fit$history$train_loss
  expect_lt(tail(h, 1), h[1])
})

test_that("to_dense_adj matches per graph adjacency matrices", {
  skip_if_no_torch()
  gs <- list(random_graph(4, 7, 1, seed = 1), random_graph(3, 5, 1, seed = 2))
  for (k in 1:2) gs[[k]]$edge_weight <- NULL
  b <- graph_batch(gs)
  adj <- as.array(to_dense_adj(graph_tensors(b)$edge_index, b$batch))
  for (k in 1:2) {
    n <- gs[[k]]$num_nodes
    ref <- t(as.matrix(adjacency_from_graph(gs[[k]], weighted = FALSE)))
    expect_equal(adj[k, 1:n, 1:n], unname(ref), info = k)
  }
  expect_equal(adj[2, 4, ], rep(0, 4))
})

test_that("dense_diff_pool with hard assignments sums clusters", {
  skip_if_no_torch()
  x <- torch::torch_tensor(array(c(1, 2, 3, 4), c(1, 4, 1)))
  A <- matrix(0, 4, 4)
  A[2, 1] <- A[1, 2] <- 1
  A[3, 2] <- A[2, 3] <- 1
  A[4, 3] <- A[3, 4] <- 1
  adj <- torch::torch_tensor(array(A, c(1, 4, 4)))
  S <- matrix(c(1, 1, 0, 0, 0, 0, 1, 1), 4, 2)
  s <- torch::torch_tensor(array(S * 50, c(1, 4, 2)))
  p <- dense_diff_pool(x, adj, s)
  expect_equal(as.vector(as.array(p$x)), c(3, 7), tolerance = 1e-6)
  expect_equal(as.array(p$adj)[1, , ], t(S) %*% A %*% S, tolerance = 1e-6)
  expect_lt(as.numeric(p$entropy_loss$item()), 1e-6)
  mask <- torch::torch_tensor(matrix(c(TRUE, TRUE, TRUE, FALSE), 1))
  pm <- dense_diff_pool(x, adj, s, mask)
  expect_equal(as.vector(as.array(pm$x)), c(3, 3), tolerance = 1e-6)
})

test_that("diffpool_model is permutation invariant and trains", {
  skip_if_no_torch()
  torch::torch_manual_seed(5)
  g <- random_graph(8, 16, 1, seed = 5)
  g$edge_weight <- NULL
  m <- diffpool_model(1, 8, 2, num_clusters = 3)
  m$eval()
  perm <- withr::with_seed(6, sample.int(8))
  o1 <- as_mat(graphlearnR:::model_forward(m, graph_tensors(g)))
  o2 <- as_mat(graphlearnR:::model_forward(m, graph_tensors(permute_graph(g, perm))))
  expect_equal(o2, o1, tolerance = 1e-5)
  expect_true(inherits(m$aux_loss, "torch_tensor"))
  gs <- dataset_synthetic_graphs(80, seed = 1)
  sp <- split_graphs(80, labels = vapply(gs, function(h) as.character(h$graph_y), ""), seed = 1)
  torch::torch_manual_seed(1)
  fit <- train_graph_model(diffpool_model(1, 16, 2, num_clusters = 4), gs[sp$train],
                           "graph_classification", epochs = 30, batch_size = 16, seed = 1)
  ev <- evaluate_graph_model(fit, gs[sp$test])
  expect_gt(ev$value[ev$metric == "accuracy"], 0.8)
  path <- withr::local_tempfile(fileext = ".rds")
  save_graph_fit(fit, path)
  back <- load_graph_fit(path)
  expect_equal(predict(back, gs[sp$test])$.pred_class, predict(fit, gs[sp$test])$.pred_class)
})
