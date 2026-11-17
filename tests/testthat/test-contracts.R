# Regression tests for data, sampling, metric and numerical contracts.

test_that("an existing exact reverse relation is paired and split together", {
  ei <- rbind(c(1, 2, 3), c(2, 3, 4))
  hg <- hetero_graph_data(nodes = list(a = 4, b = 5),
                          edges = list(a__r__b = ei, b__rev_r__a = ei[2:1, c(3, 1, 2), drop = FALSE]))
  hg <- add_reverse_relations(hg)
  expect_equal(hg$edges$a__r__b$reverse, "b__rev_r__a")
  expect_equal(hg$edges$b__rev_r__a$reverse_of, "a__r__b")
  expect_identical(add_reverse_relations(hg)$edges, hg$edges)
  sp <- split_edges_hetero(hg, "a__r__b", val = 1, test = 1, seed = 1)
  held <- cbind(sp$val_pos, sp$test_pos)
  rev <- sp$train_graph$edges$b__rev_r__a$edge_index
  expect_false(any(paste(held[2, ], held[1, ]) %in% paste(rev[1, ], rev[2, ])))
  bad <- hetero_graph_data(nodes = list(a = 4, b = 5),
                           edges = list(a__r__b = ei, b__rev_r__a = rbind(1, 1)))
  expect_error(add_reverse_relations(bad), "not the exact reverse")
})

test_that("to_homogeneous keeps matrix targets and mixed weights", {
  hg <- hetero_graph_data(nodes = list(a = list(x = matrix(1, 2, 1), y = matrix(1:4, 2, 2,
                                         dimnames = list(NULL, c("u", "v")))),
                                       b = matrix(1, 2, 1)),
                          edges = list(a__r__b = rbind(c(1, 2), c(1, 2))))
  g <- to_homogeneous(hg, target_type = "a")
  expect_equal(dim(g$y), c(4L, 2L))
  expect_equal(colnames(g$y), c("u", "v"))
  expect_equal(g$y[1:2, ], matrix(1:4, 2, 2, dimnames = list(NULL, c("u", "v"))))
  expect_true(all(is.na(g$y[3:4, ])))
  hg <- hetero_graph_data(nodes = list(a = 3, b = 3),
                          edges = list(a__r__b = list(edge_index = rbind(1, 2), edge_weight = 10),
                                       b__s__a = rbind(1, 3)))
  expect_equal(to_homogeneous(hg)$edge_weight, c(10, 1))
})

test_that("transforms honor fill weights, keep edge masks and reject batches", {
  g <- graph_data(x = matrix(1, 2, 1), edge_index = rbind(1, 2))
  g2 <- add_self_loops(g, fill_weight = 3)
  expect_equal(g2$edge_weight, c(1, 3, 3))
  g <- graph_data(x = matrix(1, 4, 1), edge_index = rbind(c(1, 2, 3), c(2, 3, 4)), edge_y = 1:3)
  g <- split_edge_labels(g, train = 0.34, val = 0.33, test = 0.33, seed = 1)
  u <- as_undirected(g)
  expect_equal(u$edge_train_mask, g$edge_train_mask)
  expect_equal(u$edge_test_mask, g$edge_test_mask)
  b <- graph_batch(list(graph_data(x = matrix(1, 2, 1), edge_index = rbind(1, 2)),
                        graph_data(x = matrix(1, 3, 1), edge_index = rbind(c(1, 2), c(2, 3)))))
  expect_error(induced_subgraph(b, c(1, 3)), "graph_batch")
  expect_error(add_self_loops(b), "graph_batch")
  g1 <- graph_data(x = matrix(1, 2, 1), graph_attr = c(1, 2))
  g3 <- graph_data(x = matrix(1, 2, 1), graph_attr = 3)
  expect_error(graph_batch(list(g1, g3)), "graph_attr")
})

test_that("undirected negative sampling with loops is uniform", {
  x <- negative_sampling(NULL, num_nodes = 2, num_samples = 1, directed = FALSE,
                         allow_self_loops = TRUE, seed = 1)
  expect_equal(dim(x), c(2L, 1L))
  set.seed(717)
  draws <- vapply(seq_len(6000), function(i) {
    p <- negative_sampling(NULL, num_nodes = 2, num_samples = 1, directed = FALSE,
                           allow_self_loops = TRUE)
    paste(p[, 1], collapse = "-")
  }, character(1))
  prop <- prop.table(table(draws))
  # each of the three pairs has probability 1/3; 4 standard errors is ~0.024
  expect_equal(sort(names(prop)), c("1-1", "1-2", "2-2"))
  expect_true(all(abs(prop - 1 / 3) < 0.025))
})

test_that("splits validate proportions and give every evaluation split negatives", {
  expect_error(split_graphs(10, train = -0.5, val = 0.1, test = 99, method = "group",
                            groups = 1:10, seed = 1), "proportions")
  g <- graph_data(x = matrix(1, 5, 1), edge_index = rbind(c(1, 2, 3), c(2, 3, 4)))
  sp <- split_edges(g, val = 1, test = 1, neg_ratio = 0.5, seed = 1)
  expect_equal(c(ncol(sp$val_neg), ncol(sp$test_neg)), c(1L, 1L))
})

test_that("synthetic graphs honor a fixed size and loaders report their length", {
  gs <- dataset_synthetic_graphs(10, min_nodes = 5, max_nodes = 5, seed = 1)
  sizes <- vapply(gs, function(g) g$num_nodes - (as.character(g$graph_y) == "wheel"), numeric(1))
  expect_true(all(sizes == 5))
  expect_error(dataset_synthetic_graphs(2, min_nodes = 6, max_nodes = 5), "max_nodes")
  g <- graph_data(x = matrix(1, 8, 1), edge_index = rbind(1:7, 2:8), train_mask = rep(TRUE, 8))
  cl <- cluster_loader(g, num_parts = 4, clusters_per_batch = 2, method = "random", seed = 1)
  sl <- saint_loader(g, batch_size = 2, num_steps = 3, sample_coverage = 0, seed = 1)
  ll <- ladies_loader(g, batch_size = 2, layer_sizes = 2, seed = 1)
  expect_equal(length(cl), length(loader_batches(cl)))
  expect_equal(length(sl), length(loader_batches(sl)))
  expect_equal(length(ll), length(loader_batches(ll)))
})

test_that("classification metrics refuse mismatched or missing inputs", {
  expect_error(metric_accuracy(c(1, 1, 2, 2), c(1, 1)), "values")
  expect_error(metric_f1(c(1, 2), c(1, NA)), "missing")
  expect_error(metric_roc_auc(c(1, 0, 1), c(0.2, 0.4)), "rows")
  truth <- factor(c("a", "b", "a"))
  p <- cbind(b = c(0.1, 0.8, 0.3), a = c(0.9, 0.2, 0.7))
  expect_equal(metric_roc_auc(truth, p), metric_roc_auc(truth, p[, c("a", "b")]))
})

test_that("spatiotemporal signals check shapes and training handles one origin", {
  g <- graph_data(x = matrix(1, 2, 1), edge_index = rbind(1, 2), num_nodes = 2)
  f <- array(rnorm(5 * 2), c(5, 2, 1))
  expect_error(spatiotemporal_signal(g, f, targets = matrix(0, 2, 2)), "targets")
  expect_error(spatiotemporal_signal(g, f, times = 1:2), "times")
  expect_error(spatiotemporal_signal(g, f, times = c(1, 3, 2, 4, 5)), "times")
})

test_that("snapshot sequences require one node order", {
  a <- graph_data(edge_index = rbind(1, 2), num_nodes = 3, node_ids = c("x", "y", "z"))
  b <- graph_data(edge_index = rbind(1, 2), num_nodes = 3, node_ids = c("y", "x", "z"))
  expect_error(snapshot_sequence(list(a, b)), "node_ids")
  expect_s3_class(snapshot_sequence(list(a, a)), "snapshot_sequence")
})

test_that("scatter reductions keep shape, precision and gradients", {
  skip_if_no_torch()
  idx <- torch::torch_tensor(c(1L, 1L, 2L, 2L), dtype = torch::torch_long())
  v <- torch::torch_tensor(c(1, 2, 3, 5))
  expect_equal(scatter_aggregate(v, idx, 2, c("sum", "max"))$shape, c(2, 2))
  x <- torch::torch_tensor(matrix(c(10000, 10001), 2, 1), dtype = torch::torch_float())
  s <- scatter_std(x, c(1L, 1L), 1)
  expect_equal(as.numeric(s$item()), 0.5, tolerance = 1e-4)
  for (a in c("sum", "mean", "max", "std")) {
    e <- torch::torch_zeros(0, 3, requires_grad = TRUE)
    out <- scatter_aggregate(e, torch::torch_tensor(integer(), dtype = torch::torch_long()), 2, a)
    out$sum()$backward()
    expect_equal(e$grad$shape, c(0, 3))
  }
})

test_that("FAVOR attention is independent across graphs in a batch", {
  skip_if_no_torch()
  torch::torch_manual_seed(1)
  attn <- linear_attention(4, heads = 2, num_features = 8)
  attn$eval()
  a <- torch::torch_randn(3, 4)
  big <- torch::torch_randn(4, 4) * 20
  alone <- attn(a)
  both <- attn(torch::torch_cat(list(a, big)), batch = c(1, 1, 1, 2, 2, 2, 2))
  expect_equal(as_r_array(both[1:3, ]), as_r_array(alone), tolerance = 1e-5)
})

test_that("explainers accept a one edge graph and pg_explainer seeds the R stream", {
  skip_if_no_torch()
  g <- graph_data(x = matrix(c(1, 2), 2, 1), edge_index = rbind(1, 2), num_nodes = 2)
  m <- torch::nn_module(
    initialize = function() self$w <- torch::nn_parameter(torch::torch_ones(1)),
    forward = function(x, edge_index, edge_weight = NULL, return_embedding = FALSE, ...) {
      h <- message_passing(x, edge_index, edge_weight = edge_weight) + x
      if (return_embedding) return(h)
      torch::torch_cat(list(h * self$w, -h * self$w), dim = 2)
    }
  )()
  expect_s3_class(gnn_explainer(m, g, target = 2, epochs = 2), "graph_explanation")
  set.seed(99)
  h1 <- pg_explainer(m, g, targets = 1:2, epochs = 3, seed = 7)$history
  after <- runif(1)
  set.seed(99)
  invisible(runif(1))
  set.seed(100)
  h2 <- pg_explainer(m, g, targets = 1:2, epochs = 3, seed = 7)$history
  expect_equal(h1, h2)
  set.seed(99)
  pg_explainer(m, g, targets = 1:2, epochs = 3, seed = 7)
  expect_equal(runif(1), after)
})

test_that("TGN scores one pair and refuses a time equal to the last event", {
  skip_if_no_torch()
  tg <- dataset_synthetic_temporal(num_events = 200, seed = 2)
  torch::torch_manual_seed(1)
  fit <- train_tgn(tgn_model(tg$num_nodes, 8, 4, 8, embedding = "graph_sum"), split_temporal(tg),
                   epochs = 1, batch_size = 50, seed = 1)
  pr <- rbind(c(1, 2, 3), c(4, 5, 6))
  t1 <- fit$last_time + 1
  all3 <- predict(fit, pairs = pr, time = t1)
  expect_equal(predict(fit, pairs = pr[, 2, drop = FALSE], time = t1), all3[2], tolerance = 1e-6)
  expect_error(predict(fit, pairs = pr, time = fit$last_time), "strictly after")
  same <- temporal_graph_data(src = 1, dst = 2, t = fit$last_time, num_nodes = tg$num_nodes)
  expect_error(predict(fit, new_data = same, pairs = pr, time = t1), "strictly after")
})

test_that("a subgraph loader built on another graph of the same size is refused", {
  skip_if_no_torch()
  g <- split_nodes(graph_data(x = diag(6), edge_index = rbind(1:5, 2:6), y = factor(rep(1:2, 3))),
                   seed = 1)
  other <- g
  other$edge_index <- rbind(2:6, 1:5)
  cl <- cluster_loader(other, num_parts = 2, method = "random", seed = 1)
  expect_error(train_graph_model(gnn_model("gcn", 6, 4, 2), g, "node_classification",
                                 epochs = 1, loader = cl), "different graph")
})

test_that("spatiotemporal training works with a single training origin", {
  skip_if_no_torch()
  sig <- dataset_synthetic_signal(seed = 1)
  sig$features <- sig$features[1:7, , , drop = FALSE]
  sig$targets <- sig$targets[1:7, , drop = FALSE]
  sig$times <- sig$times[1:7]
  fit <- train_spatiotemporal_model(sig, lags = 3, train = 0.3, val = 0.3, hidden_dim = 4,
                                    epochs = 1, seed = 1)
  expect_equal(nrow(fit$history), 1L)
  expect_error(train_spatiotemporal_model(sig, lags = 0), "lags")
})

test_that("dynamic link prediction refuses snapshots with another node order", {
  skip_if_no_torch()
  snaps <- temporal_snapshots(dataset_synthetic_temporal(seed = 1), breaks = 4)
  fit <- train_dynamic_link_model(snaps, hidden_dim = 4, epochs = 1, seed = 1)
  ok <- predict(fit, snaps, pairs = rbind(1, 2), step = 2)
  expect_length(ok, 1L)
  perm <- snaps
  perm$graphs <- lapply(perm$graphs, function(g) { g$node_ids <- rev(g$node_ids); g })
  expect_error(predict(fit, perm, pairs = rbind(1, 2), step = 2), "node_ids")
})

test_that("GCN on an all neighbor sample matches the full graph at the seeds", {
  skip_if_no_torch()
  set.seed(3)
  g <- graph_data(x = matrix(rnorm(12 * 3), 12, 3),
                  edge_index = rbind(c(1:11, 3, 7), c(2:12, 9, 1)), directed = FALSE,
                  edge_weight = runif(13) + 0.5)
  torch::torch_manual_seed(1)
  m <- gnn_model("gcn", 3, 4, 2, num_layers = 2, dropout = 0)
  m$eval()
  full <- as_r_array(torch::with_no_grad(model_forward(m, graph_tensors(g))))
  seeds <- c(5, 1)
  sg <- sample_neighbors(g, seeds, c(-1, -1))
  sub <- as_r_array(torch::with_no_grad(model_forward(m, graph_tensors(sg))))
  expect_equal(sub[seq_along(seeds), ], full[seeds, ], tolerance = 1e-5)
  expect_null(sample_neighbors(g, seeds, c(2, 2), seed = 1)$node_degree)
})

test_that("shared ranking and loader caches give the reference results", {
  ref_mrr <- function(pos, neg) {
    neg <- matrix(neg, length(pos), length(neg), byrow = TRUE)
    mean(1 / (1 + rowSums(neg > pos) + 0.5 * rowSums(neg == pos)))
  }
  set.seed(4)
  for (i in 1:50) {
    p <- round(rnorm(sample(1:10, 1)), 1)
    n <- round(rnorm(sample(1:15, 1)), 1)
    expect_equal(metric_mrr(p, n), ref_mrr(p, n))
  }
  expect_equal(metric_mrr(1, c(1, 1)), 0.5)
  expect_equal(metric_hits_at_k(1, c(1, 1), k = 2), 0)
  expect_error(metric_mrr(c(1, NA), 1), "missing")
  g <- dataset_karate()
  ld <- node_loader(g, nodes = 1:20, batch_size = 7, fanouts = c(3, 2), seed = 5)
  b <- loader_batches(ld, epoch = 2)
  s <- sub_seeds(ld$seed + 2, length(b))
  parts <- batch_split(20, 7, TRUE, ld$seed + 2)
  for (i in seq_along(b)) {
    expect_identical(b[[i]], sample_neighbors(g, (1:20)[parts[[i]]], c(3, 2), seed = s[i]))
  }
})

test_that("a GCN model and graph GRU equal their layer by layer computation", {
  skip_if_no_torch()
  g <- dataset_karate()
  tt <- graph_tensors(g)
  torch::torch_manual_seed(2)
  m <- gnn_model("gcn", 34, 8, 2, num_layers = 3, dropout = 0)
  m$eval()
  out <- m(tt$x, tt$edge_index)
  h <- tt$x
  for (i in 1:3) h <- torch::nnf_relu(m$convs[[i]](h, tt$edge_index)) + if (i > 1 && m$config$residual) h else 0
  expect_equal(as_r_array(out), as_r_array(m$head(h)), tolerance = 1e-5)
  cell <- graph_gru_cell(34, 4)
  hh <- cell(tt$x, tt$edge_index)
  z <- torch::torch_sigmoid(cell$xz(tt$x, tt$edge_index))
  ht <- torch::torch_tanh(cell$xh(tt$x, tt$edge_index))
  expect_equal(as_r_array(hh), as_r_array((1 - z) * ht + z * torch::torch_zeros_like(ht)),
               tolerance = 1e-5)
})
