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
  # Each stored direction carries the split of its input pair, so the
  # labels in each split are those of the original split, and no pair is in
  # two splits.
  expect_equal(length(u$edge_train_mask), num_edges(u))
  for (f in c("edge_train_mask", "edge_val_mask", "edge_test_mask")) {
    expect_equal(u[[f]], g[[f]][u$edge_pair])
    expect_setequal(u$edge_y[u[[f]]], g$edge_y[g[[f]]])
  }
  pair_split <- tapply(u$edge_train_mask + 2 * u$edge_val_mask + 3 * u$edge_test_mask,
                       u$edge_pair, function(z) length(unique(z)))
  expect_true(all(pair_split == 1))
  # loops, reciprocal input and duplicates
  h <- graph_data(x = matrix(1, 3, 1), edge_index = rbind(c(1, 2, 3, 1), c(2, 1, 3, 2)),
                  edge_y = 1:4)
  h$edge_train_mask <- c(TRUE, TRUE, FALSE, TRUE)
  h$edge_test_mask <- !h$edge_train_mask
  hu <- as_undirected(h)
  expect_equal(num_edges(hu), 3L)
  expect_equal(hu$edge_train_mask, c(TRUE, TRUE, FALSE))
  expect_equal(hu$edge_test_mask, c(FALSE, FALSE, TRUE))
  e0 <- graph_data(x = matrix(1, 2, 1), edge_index = matrix(integer(), 2, 0))
  e0$edge_train_mask <- logical()
  expect_equal(as_undirected(e0)$edge_train_mask, logical())
  bad <- g
  bad$edge_train_mask <- c(TRUE, FALSE)
  expect_error(validate_graph(bad), "stored edges")
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

test_that("sampled training validates on exact subgraphs without the full graph", {
  skip_if_no_torch()
  g <- split_nodes(dataset_karate(), seed = 1)
  torch::torch_manual_seed(1)
  m <- gnn_model("gcn", 34, 8, 2, dropout = 0)
  fit <- train_graph_model(m, g, "node_classification", epochs = 2, fanouts = c(3, 3),
                           batch_size = 8, seed = 1)
  m$eval()
  full <- torch::with_no_grad(model_forward(m, graph_tensors(g)))
  v <- which(g$val_mask)
  pred <- as.integer(as_r_array(full$argmax(dim = 2)))[v]
  acc <- mean(pred == as.integer(g$y)[v])
  expect_equal(utils::tail(fit$history$val_metric, 1), acc, tolerance = 1e-6)
})

test_that("the vectorized TGN neighbor store matches event by event insertion", {
  ref_add <- function(store, src, dst, t, edge_attr = NULL) {
    k <- store$k
    for (i in seq_along(src)) {
      for (pr in list(c(src[i], dst[i]), c(dst[i], src[i]))) {
        a <- pr[1L]
        store$nb[a, ] <- c(pr[2L], store$nb[a, ])[seq_len(k)]
        store$t[a, ] <- c(t[i], store$t[a, ])[seq_len(k)]
        if (!is.null(store$e)) {
          d <- dim(store$e)[3L]
          cur <- matrix(store$e[a, , ], k, d)
          store$e[a, , ] <- rbind(edge_attr[i, ], cur)[seq_len(k), , drop = FALSE]
        }
      }
    }
    store
  }
  set.seed(8)
  for (k in c(1L, 3L)) {
    s1 <- s2 <- list(k = k, nb = matrix(0L, 6, k), t = matrix(0, 6, k), e = array(0, c(6, k, 2)))
    for (b in 1:4) {
      m <- sample(1:5, 1)
      src <- sample(6, m, TRUE); dst <- sample(6, m, TRUE); tm <- b + seq_len(m) / 10
      ea <- matrix(rnorm(2 * m), m, 2)
      s1 <- tgn_store_add(s1, src, dst, tm, ea)
      s2 <- ref_add(s2, src, dst, tm, ea)
    }
    expect_equal(s1, s2)
  }
})

test_that("transforms carry every edge field", {
  g <- graph_data(x = matrix(1:6, 3, 2), edge_index = rbind(c(1, 1, 2), c(2, 2, 3)),
                  edge_attr = matrix(c(1, 3, 5, 2, 4, 6), 3, 2, dimnames = list(NULL, c("a", "b"))),
                  edge_y = matrix(1:6, 3, 2), edge_type = c(1L, 1L, 2L))
  g$edge_train_mask <- c(TRUE, TRUE, FALSE)
  l <- add_self_loops(g, loop_type = 3L)
  expect_equal(num_edges(l), 6L)
  expect_equal(l$edge_attr[4:6, ], matrix(0, 3, 2, dimnames = list(NULL, c("a", "b"))))
  expect_true(all(is.na(l$edge_y[4:6, ])))
  expect_equal(l$edge_type[4:6], rep(3L, 3))
  expect_equal(l$edge_train_mask, c(TRUE, TRUE, FALSE, FALSE, FALSE, FALSE))
  expect_identical(add_self_loops(l), l)
  cg <- coalesce_edges(g)
  expect_equal(num_edges(cg), 2L)
  expect_equal(unname(cg$edge_attr[1, ]), c(2, 3))
  expect_equal(colnames(cg$edge_attr), c("a", "b"))
  e <- graph_data(x = matrix(1, 2, 1), edge_index = matrix(integer(), 2, 0), num_nodes = 2)
  expect_identical(coalesce_edges(e), e)
  u <- graph_data(x = matrix(1, 2, 1), edge_index = rbind(1, 2), directed = FALSE)
  expect_identical(as_undirected(u), u)
  bad <- graph_data(x = matrix(1, 2, 1), edge_index = rbind(c(1, 2), c(2, 1)))
  bad$edge_train_mask <- c(TRUE, FALSE)
  expect_error(as_undirected(bad), "differs between the two directions")
  expect_error(normalize_features(graph_data(edge_index = rbind(1, 2), num_nodes = 2)), "no node features")
  d <- add_degree_features(graph_data(x = matrix(1, 3, 1), edge_index = rbind(c(1, 2), c(2, 3))), log = FALSE)
  expect_equal(d$x[, 2], c(0, 1, 1))
})

test_that("temporal predictions validate their inputs", {
  skip_if_no_torch()
  tg <- dataset_synthetic_temporal(num_events = 120, seed = 3)
  fit <- train_tgn(tgn_model(tg$num_nodes, 8, 4, 8, embedding = "graph_sum"), split_temporal(tg),
                   epochs = 1, batch_size = 40, seed = 1)
  t1 <- fit$last_time + 1
  expect_error(predict(fit, pairs = rbind(1, 2)), "pairs` and `time")
  expect_error(predict(fit, pairs = rbind(c(1, 2), c(3, 4)), time = c(t1, t1, t1)), "one per pair")
  expect_error(predict(fit, pairs = rbind(1, 10^6), time = t1), "outside the model")
  expect_error(predict(fit, new_data = list(), pairs = rbind(1, 2), time = t1), "temporal_graph_data")
  snaps <- temporal_snapshots(tg, breaks = 4)
  dfit <- train_dynamic_link_model(snaps, hidden_dim = 4, epochs = 1, seed = 1)
  expect_error(predict(dfit, list(), pairs = rbind(1, 2), step = 2), "snapshot_sequence")
  expect_error(predict(dfit, snaps, step = 2), "need `pairs`")
  small <- snapshot_sequence(lapply(snaps$graphs, function(g) induced_subgraph(g, 1:5)))
  expect_error(predict(dfit, small, pairs = rbind(1, 2), step = 2), "number of nodes")
  sig <- dataset_synthetic_signal(seed = 1)
  sfit <- train_spatiotemporal_model(sig, lags = 3, hidden_dim = 4, epochs = 1, seed = 1)
  expect_error(predict(sfit, list()), "spatiotemporal_signal")
  expect_equal(nrow(predict(sfit, sig)), 1L)
  expect_error(predict(sfit, sig, ends = 1), "Forecast origins")
})

test_that("sampled validation follows the model's receptive depth and full degrees", {
  skip_if_no_torch()
  g <- graph_data(x = matrix(c(0, 0, 7), 3, 1), edge_index = rbind(c(3L, 2L), c(2L, 1L)),
                  y = rep(0, 3), train_mask = c(FALSE, FALSE, TRUE),
                  val_mask = c(TRUE, FALSE, FALSE), test_mask = c(FALSE, TRUE, FALSE))
  m <- gnn_model("graph_conv", 1, 1, num_layers = 2, dropout = 0)
  with_no_grad(for (i in 1:2) {
    m$convs[[i]]$lin_rel$weight$fill_(1); m$convs[[i]]$lin_rel$bias$zero_()
    m$convs[[i]]$lin_root$weight$zero_()
  })
  # Node 1 sees 7 after two hops: full graph validation MSE is 49.
  for (fan in list(1L, c(1L, 1L))) {
    fit <- train_graph_model(m, g, "node_regression", epochs = 1, lr = 0,
                             fanouts = fan, batch_size = 1, seed = 7)
    expect_equal(fit$history$val_loss, 49, tolerance = 1e-6)
  }
  g4 <- graph_data(x = matrix(c(0, 0, 7, 0), 4, 1), edge_index = rbind(c(4L, 3L, 2L), c(3L, 2L, 1L)),
                   y = rep(0, 4), train_mask = c(FALSE, FALSE, FALSE, TRUE),
                   val_mask = c(TRUE, FALSE, FALSE, FALSE))
  a <- appnp_model(1, 1, 1, K = 2, alpha = 0.1, dropout = 0)
  with_no_grad({
    a$lin1$weight$fill_(1); a$lin1$bias$zero_(); a$lin2$weight$fill_(1); a$lin2$bias$zero_()
  })
  a$eval()
  full <- as.numeric(as_r_array(with_no_grad(model_forward(a, graph_tensors(g4))))[1])
  sg <- sample_neighbors(g4, 1L, c(-1L, -1L))
  expect_equal(as.numeric(as_r_array(with_no_grad(model_forward(a, graph_tensors(sg))))[1]), full,
               tolerance = 1e-6)
  fit <- train_graph_model(a, g4, "node_regression", epochs = 1, lr = 0, fanouts = 1L,
                           batch_size = 1, seed = 7)
  expect_equal(fit$history$val_loss, full^2, tolerance = 1e-5)
  # A custom model without a declared depth validates on the full graph.
  probe <- nn_module(initialize = function() self$w <- nn_parameter(torch_ones(1)),
                     forward = function(x, edge_index, ...) {
                       h <- scatter_sum(x$index_select(1L, edge_index[1L, ]), edge_index[2L, ], x$size(1L))
                       scatter_sum(h$index_select(1L, edge_index[1L, ]), edge_index[2L, ], x$size(1L)) * self$w
                     })
  fit <- train_graph_model(probe(), g, "node_regression", epochs = 1, lr = 0, fanouts = 1L,
                           batch_size = 1, seed = 7)
  expect_equal(fit$history$val_loss, 49, tolerance = 1e-6)
})

test_that("an empty validation list means no validation", {
  skip_if_no_torch()
  g <- graph_data(x = matrix(c(1, 2), 2, 1), edge_index = rbind(1L, 2L), graph_y = 0)
  fits <- lapply(list(NULL, list()), function(vl) {
    m <- gnn_model("sage", 1, 2, 1, num_layers = 1, dropout = 0, pool = "mean")
    train_graph_model(m, list(g, g), "graph_regression", val_data = vl, epochs = 1,
                      batch_size = 1, lr = 0, seed = 1)
  })
  expect_identical(fits[[1]]$history$val_loss, fits[[2]]$history$val_loss)
})

test_that("FAVOR+ attention has exact gradients and stays independent across graphs", {
  skip_if_no_torch()
  torch_manual_seed(3)
  attn <- linear_attention(4, heads = 2, num_features = 6)
  attn$to(dtype = torch_double())
  f <- function(x, batch = NULL) attn(x, batch)$sum()
  for (scale in c(1, 6)) {
    x0 <- matrix(rnorm(20) * scale, 5, 4)
    x <- torch_tensor(x0, dtype = torch_double(), requires_grad = TRUE)
    f(x)$backward()
    g <- as_r_array(x$grad)
    fd <- x0
    for (i in seq_along(x0)) {
      e <- x0; e[i] <- e[i] + 1e-6
      d <- x0; d[i] <- d[i] - 1e-6
      fd[i] <- (as.numeric(f(torch_tensor(e, dtype = torch_double()))) -
                  as.numeric(f(torch_tensor(d, dtype = torch_double())))) / 2e-6
    }
    expect_equal(g, fd, tolerance = 1e-5)
  }
  # two graphs in one batch give the same outputs and gradients as separately
  x0 <- matrix(rnorm(28) * 4, 7, 4)
  b <- c(1, 1, 1, 2, 2, 2, 2)
  xb <- torch_tensor(x0, dtype = torch_double(), requires_grad = TRUE)
  ob <- attn(xb, b)
  ob$sum()$backward()
  for (k in 1:2) {
    xs <- torch_tensor(x0[b == k, ], dtype = torch_double(), requires_grad = TRUE)
    os <- attn(xs)
    os$sum()$backward()
    expect_equal(as_r_array(ob)[b == k, ], as_r_array(os), tolerance = 1e-10)
    expect_equal(as_r_array(xb$grad)[b == k, ], as_r_array(xs$grad), tolerance = 1e-10)
  }
})

test_that("probability metrics keep predicted classes absent from the truth levels", {
  truth <- factor(c("a", "b"))
  p <- rbind(c(.2, .1, .7), c(.1, .2, .7))
  colnames(p) <- c("a", "b", "c")
  expect_equal(metric_brier(truth, p), 1.14)
  expect_equal(metric_ece(truth, p), 0.7)
  q <- p[, c("c", "b", "a")]
  expect_equal(metric_brier(truth, q), 1.14)
  expect_equal(metric_ece(truth, q), 0.7)
  expect_equal(metric_brier(factor(c("a", "b"), levels = c("a", "b", "c")), p), 1.14)
  expect_equal(metric_brier(c(1L, 2L), unname(p)), 1.14)
})

test_that("edgeless RGCN without root or bias gives zero gradients", {
  skip_if_no_torch()
  for (nb in list(NULL, 2L)) {
    conv <- rgcn_conv(2, 2, 3, num_bases = nb, root_weight = FALSE, bias = FALSE)
    x <- torch_randn(4, 2, requires_grad = TRUE)
    out <- conv(x, torch_zeros(2, 0, dtype = torch_long()), edge_type = integer())
    expect_true(out$requires_grad)
    out$sum()$backward()
    expect_equal(as_r_array(x$grad), matrix(0, 4, 2))
    for (p in conv$parameters) expect_equal(sum(abs(as_r_array(p$grad))), 0)
  }
})

test_that("reverse pairing ignores relation order and negative rounding matches", {
  ei <- rbind(c(1, 2, 3), c(2, 3, 4))
  f <- add_reverse_relations(hetero_graph_data(nodes = list(a = 4, b = 5),
    edges = list(a__r__b = ei, b__rev_r__a = ei[2:1, 3:1, drop = FALSE])))
  r <- add_reverse_relations(hetero_graph_data(nodes = list(a = 4, b = 5),
    edges = list(b__rev_r__a = ei[2:1, , drop = FALSE], a__r__b = ei)))
  expect_setequal(names(r$edges), c("a__r__b", "b__rev_r__a"))
  expect_equal(r$edges$a__r__b$reverse, "b__rev_r__a")
  expect_equal(r$edges$b__rev_r__a$reverse_of, "a__r__b")
  expect_identical(add_reverse_relations(r), r)
  expect_error(add_reverse_relations(hetero_graph_data(nodes = list(a = 4, b = 5),
    edges = list(b__rev_r__a = ei[2:1, 1:2, drop = FALSE], a__r__b = ei))), "not the exact reverse")
  g <- graph_data(num_nodes = 5, edge_index = ei)
  hg <- hetero_graph_data(nodes = list(a = 4, b = 5), edges = list(a__r__b = ei))
  a <- split_edges(g, val = 1, test = 1, neg_ratio = .5, seed = 1)
  b <- split_edges_hetero(hg, "a__r__b", val = 1, test = 1, neg_ratio = .5, seed = 1)
  expect_equal(c(ncol(b$val_neg), ncol(b$test_neg)), c(ncol(a$val_neg), ncol(a$test_neg)))
  expect_equal(c(ncol(b$val_neg), ncol(b$test_neg)), c(1L, 1L))
})

test_that("TGN rejects periods that share a timestamp", {
  skip_if_no_torch()
  tg <- dataset_synthetic_temporal(num_events = 60, seed = 2)
  sp <- split_temporal(tg)
  expect_s3_class(train_tgn(tgn_model(tg$num_nodes, 4, 2, 4), sp, epochs = 1, seed = 1), "temporal_fit")
  bad <- sp
  bad$val$t[1] <- max(sp$train$t)
  expect_error(train_tgn(tgn_model(tg$num_nodes, 4, 2, 4), bad, epochs = 1), "strictly after")
})

test_that("the full graph degree cache follows transforms and batching", {
  skip_if_no_torch()
  gcn_out <- function(g) {
    torch_manual_seed(1)
    m <- gnn_model("gcn", 1, 4, 1, num_layers = 2, dropout = 0)
    as.numeric(as_r_array(with_no_grad(model_forward(m, graph_tensors(g)))))
  }
  g <- graph_data(x = matrix(1:5, 5, 1), edge_index = rbind(c(1, 1, 2, 3, 4, 5), c(2, 3, 1, 1, 2, 4)),
                  edge_weight = c(2, 3, 4, 1, 2, 5))
  sg <- sample_neighbors(g, seeds = 1, fanouts = c(-1L, -1L))
  p <- rev(seq_len(sg$num_nodes))
  ps <- induced_subgraph(sg, p)
  expect_equal(ps$node_degree, sg$node_degree[p])
  expect_equal(ps$n_id, sg$n_id[p])
  expect_equal(ps$seed_index, which(p == 1L))
  expect_equal(gcn_out(ps), gcn_out(sg)[p], tolerance = 1e-6)
  small <- induced_subgraph(sg, 1:2)
  expect_length(small$node_degree, 2L)
  expect_silent(validate_graph(small))
  # edge edits move the cache by the change in local degrees
  local_in <- tabulate(sg$edge_index[2, ], sg$num_nodes)
  has_loop <- tabulate(sg$edge_index[2, sg$edge_index[1, ] == sg$edge_index[2, ]], sg$num_nodes) > 0
  expect_equal(add_self_loops(sg)$node_degree, sg$node_degree + !has_loop)
  expect_equal(coalesce_edges(sg)$node_degree, sg$node_degree)
  # one sampled graph in a batch keeps its seed output; unbatch restores it
  b <- graph_batch(list(sg))
  expect_equal(gcn_out(b), gcn_out(sg), tolerance = 1e-6)
  b2 <- graph_batch(list(sg, g))
  expect_equal(gcn_out(b2)[seq_len(sg$num_nodes)], gcn_out(sg), tolerance = 1e-6)
  expect_equal(gcn_out(b2)[-seq_len(sg$num_nodes)], gcn_out(g), tolerance = 1e-6)
  u <- unbatch(b2)
  expect_equal(u[[1]]$node_degree, sg$node_degree)
  expect_null(u[[2]]$node_degree)
  # editing edges by hand makes the cache refuse to be used
  bad <- sg
  bad$edge_weight <- bad$edge_weight * 2
  expect_error(graph_tensors(bad), "no longer matches")
  expect_error(validate_graph(bad), "no longer matches")
})

test_that("feature explanations of a sampled graph use the prediction's normalization", {
  skip_if_no_torch()
  g <- graph_data(x = matrix(1, 3, 1), edge_index = rbind(c(1L, 2L), c(2L, 3L)), directed = FALSE)
  sg <- sample_neighbors(g, seeds = 1L, fanouts = -1L)
  m <- gnn_model("gcn", 1L, 1L, 1L, num_layers = 1L, dropout = 0)
  with_no_grad({
    m$convs[[1L]]$lin$weight$fill_(1); m$convs[[1L]]$bias$zero_()
    m$head$weight$fill_(1); m$head$bias$zero_()
  })
  m$eval()
  fit <- structure(list(model = m, data = sg, device = torch_device("cpu"), task = "node_regression"),
                   class = "graph_fit")
  pred <- as.numeric(predict(fit, type = "response"))[1]
  full <- as.numeric(as_r_array(with_no_grad(model_forward(m, graph_tensors(g)))))[1]
  expect_equal(pred, full, tolerance = 1e-6)
  tt <- graph_tensors(sg)
  tt$x <- tt$x$detach()$requires_grad_(TRUE)
  model_forward(m, tt)[1, 1]$backward()
  sal <- explain_saliency(fit, sg, target = 1, output = 1)
  expect_equal(as.vector(sal$node_feature), as.vector(as_r_array(tt$x$grad)), tolerance = 1e-6)
  ig <- explain_integrated_gradients(fit, sg, target = 1, output = 1, baseline = 0, steps = 10)
  expect_equal(sum(ig$node_feature), pred, tolerance = 1e-5)
})

test_that("loaders follow replaced graphs and current edge inputs", {
  g <- graph_data(x = matrix(1:3, 3, 1), edge_index = rbind(c(2, 3), c(1, 2)), edge_weight = c(2, 3))
  ld <- node_loader(g, nodes = 1, fanouts = -1, shuffle = FALSE, seed = 1)
  ld$graph <- graph_data(x = g$x, edge_index = rbind(c(2, 3), c(3, 1)), edge_weight = c(5, 7))
  expected <- sample_neighbors(ld$graph, seeds = 1, fanouts = -1)
  actual <- loader_batches(ld)[[1]]
  expect_identical(actual$edge_index, expected$edge_index)
  expect_identical(actual$n_id, expected$n_id)
  expect_identical(actual$node_degree, expected$node_degree)
  b <- list()
  for_each_batch(ld, 1L, function(x) b[[length(b) + 1L]] <<- x)
  expect_identical(b[[1]]$edge_index, expected$edge_index)
  skip_if_no_torch()
  g <- graph_data(x = matrix(1, 3, 1), edge_index = rbind(c(1L, 2L), c(2L, 3L)),
                  edge_attr = matrix(1, 2, 1), y = rep(0, 3),
                  train_mask = c(FALSE, TRUE, FALSE), val_mask = c(FALSE, FALSE, TRUE))
  old <- cluster_loader(g, num_parts = 1L, method = "random", seed = 1)
  g$edge_attr <- matrix(10, 2, 1)
  edge_probe <- nn_module(initialize = function() self$scale <- nn_parameter(torch_tensor(1)),
                          forward = function(x, edge_index, edge_attr = NULL, ...) {
                            scatter_sum(edge_attr * self$scale, edge_index[2L, ], x$size(1L))
                          })
  losses <- lapply(list(old, cluster_loader(g, num_parts = 1L, method = "random", seed = 1)), function(l) {
    fit <- train_graph_model(edge_probe(), g, "node_regression", epochs = 1, lr = 0, loader = l, seed = 1)
    c(fit$history$train_loss, fit$history$val_loss)
  })
  expect_equal(losses[[1]], losses[[2]])
  expect_equal(losses[[1]], c(100, 100))
})

test_that("SAINT and LADIES warn for models whose aggregation they do not correct", {
  skip_if_no_torch()
  g <- split_nodes(dataset_sbm(c(10, 10), seed = 1), seed = 1)
  sl <- saint_loader(g, 5, 2, 2, seed = 1)
  ld <- ladies_loader(g, batch_size = 5, layer_sizes = c(5, 5), seed = 1)
  for (L in list(sl, ld)) {
    expect_warning(train_graph_model(gnn_model("sage", ncol(g$x), 4, 2), g, "node_classification",
                                     epochs = 1, loader = L, seed = 1), "sums weighted messages")
    expect_no_warning(train_graph_model(gnn_model("graph_conv", ncol(g$x), 4, 2), g,
                                        "node_classification", epochs = 1, loader = L, seed = 1))
  }
  expect_no_warning(train_graph_model(gnn_model("sage", ncol(g$x), 4, 2), g, "node_classification",
                                      epochs = 1, loader = cluster_loader(g, 2, seed = 1), seed = 1))
})

test_that("score metrics reject missing scores", {
  expect_error(metric_roc_auc(c(1, 0), c(.2, NA_real_)), "missing values")
})

test_that("degree caches are exact, follow edge edits and nest", {
  skip_if_no_torch()
  # loader key collision: same weighted index sums, different edges
  g <- graph_data(x = matrix(1, 4, 1), edge_index = rbind(c(1, 3), c(2, 2)))
  ld <- node_loader(g, nodes = 2, fanouts = -1, shuffle = FALSE)
  ld$graph <- graph_data(x = matrix(1, 4, 1), edge_index = rbind(c(1, 3), c(4, 1)))
  expect_identical(loader_batches(ld)[[1]]$edge_index,
                   sample_neighbors(ld$graph, 2, -1)$edge_index)
  # degree cache key collision
  g <- graph_data(x = matrix(1, 4, 1), edge_index = rbind(c(1, 2), c(2, 4)))
  sg <- sample_neighbors(g, 1:4, -1)
  bad <- sg
  bad$edge_index[2, ] <- c(3L, 3L)
  expect_error(graph_tensors(bad), "no longer matches")
  # nested all neighbor sampling keeps the first sample's degrees
  gcn1 <- function(h) {
    m <- gcn_conv(1, 1, bias = FALSE)
    with_no_grad(m$lin$weight$fill_(1))
    tt <- graph_tensors(h)
    as.numeric(as_r_array(with_no_grad(m(tt$x, tt$edge_index, tt$edge_weight,
                                           node_degree = tt$node_degree))))[1]
  }
  ch <- graph_data(x = matrix(c(0, 1, 0), 3, 1), edge_index = rbind(c(3, 2), c(2, 1)))
  s1 <- sample_neighbors(ch, 1, -1)
  s2 <- sample_neighbors(s1, 1, -1)
  expect_equal(gcn1(s1), 0.5, tolerance = 1e-6)
  expect_equal(gcn1(s2), 0.5, tolerance = 1e-6)
  # identity edits keep the cache; real edits move it by the local change
  expect_identical(edge_subset(s1, seq_len(num_edges(s1)))$node_degree, s1$node_degree)
  expect_equal(remove_self_loops(add_self_loops(s1))$node_degree, s1$node_degree)
  expect_equal(edge_subset(s1, integer())$node_degree, s1$node_degree - c(1, 0))
})

test_that("an identity edge intervention reproduces the prediction", {
  skip_if_no_torch()
  ch <- graph_data(x = matrix(c(1, 2, 3), 3, 1), edge_index = rbind(c(3, 2), c(2, 1)))
  sg <- sample_neighbors(ch, 1, -1)
  torch_manual_seed(1)
  m <- gnn_model("gcn", 1, 4, 1, num_layers = 1, dropout = 0)
  fit <- structure(list(model = m, data = sg, device = torch_device("cpu"), task = "node_regression"),
                   class = "graph_fit")
  ex <- explain_edges(fit, sg, target = 1, output = 1)
  expect_equal(unname(explanation_fidelity(fit, sg, ex, k = 0)[["fidelity_plus"]]), 0, tolerance = 1e-6)
  expect_equal(unname(explanation_fidelity(fit, sg, ex, k = 99)[["fidelity_minus"]]), 0, tolerance = 1e-6)
})

test_that("metrics handle large sizes, named columns for any truth type and bad inputs", {
  expect_equal(metric_roc_auc(rep(c(FALSE, TRUE), each = 50000), rep(c(0, 1), each = 50000)), 1)
  p <- rbind(c(b = .1, a = .9), c(b = .9, a = .1))
  expect_equal(metric_roc_auc(c("a", "b"), p), 1)
  expect_equal(metric_brier(c("a", "b"), p), .02)
  expect_equal(metric_ece(c("a", "b"), p), .1)
  q <- rbind(c(`1` = .1, `0` = .9), c(`1` = .9, `0` = .1))
  expect_equal(metric_brier(c(0, 1), q), .02)
  expect_error(metric_hits_at_k(NA_real_, numeric(), 1), "missing")
  expect_error(metric_hits_at_k(0, c(NA_real_, 1), 2), "missing")
  expect_error(metric_ari(1, 1), "at least two")
})

test_that("scatter extremes, weighted vector messages and GCN normalization edge cases", {
  skip_if_no_torch()
  for (r in list(list(f = scatter_max, v = c(0, -1)), list(f = scatter_min, v = c(0, 1)))) {
    s <- torch_tensor(r$v, requires_grad = TRUE)
    out <- r$f(s, c(1L, 1L), 2)
    expect_equal(as.numeric(as_r_array(out)), c(0, 0))
    out$sum()$backward()
    expect_equal(as.numeric(as_r_array(s$grad)), c(1, 0))
  }
  x <- torch_tensor(c(2, 3, 0))
  ei <- torch_tensor(rbind(c(1L, 2L), c(3L, 3L)), dtype = torch_long())
  expect_equal(as.numeric(as_r_array(message_passing(x, ei, edge_weight = torch_tensor(c(5, 7))))),
               c(0, 0, 31))
  w <- torch_tensor(-2, requires_grad = TRUE)
  n1 <- gcn_norm(torch_tensor(rbind(1L, 2L), dtype = torch_long()), 2, w)
  expect_false(any(is.nan(as_r_array(n1$edge_weight))))
  w0 <- torch_tensor(0, requires_grad = TRUE)
  n0 <- gcn_norm(torch_tensor(rbind(1L, 2L), dtype = torch_long()), 2, w0, add_self_loops = FALSE)
  n0$edge_weight$sum()$backward()
  expect_false(is.nan(as.numeric(w0$grad)))
})

test_that("training negatives never include evaluation pairs", {
  skip_if_no_torch()
  g <- graph_data(x = diag(6), edge_index = rbind(c(1, 2, 3, 4, 5, 1, 2), c(2, 3, 4, 5, 6, 3, 5)))
  sp <- split_edges(g, val = 2, test = 2, seed = 3)
  key <- function(m) paste(m[1, ], m[2, ])
  ev <- key(cbind(sp$val_pos, sp$val_neg, sp$test_pos, sp$test_neg))
  seen <- new.env()
  seen$train <- character()
  rec <- nn_module(initialize = function() self$b <- nn_parameter(torch_zeros(1)),
                   forward = function(z, pairs) {
                     if (z$requires_grad) {
                       seen$train <- c(seen$train, key(as_r_array(pairs$cpu())))
                     }
                     (z$index_select(1L, pairs[1L, ]) * z$index_select(1L, pairs[2L, ]))$sum(2L) + self$b
                   })
  train_graph_model(gnn_model("sage", 6, 4, num_layers = 1), sp, "link_prediction",
                    decoder = rec(), epochs = 30, seed = 1)
  expect_gt(length(seen$train), 0)
  expect_false(any(seen$train %in% ev))
})

test_that("graph evaluation never scores training graphs as held out data", {
  skip_if_no_torch()
  gs <- lapply(1:4, function(i) graph_data(x = matrix(i, 2, 1), edge_index = rbind(1L, 2L), graph_y = i))
  fit <- train_graph_model(gnn_model("sage", 1, 2, 1, num_layers = 1, pool = "mean"), gs[1:2],
                           "graph_regression", epochs = 1, seed = 1)
  expect_error(evaluate_graph_model(fit), "need `new_data`")
  expect_error(evaluate_graph_model(fit, split = "val"), "validation graphs")
  expect_s3_class(evaluate_graph_model(fit, split = "train"), "data.frame")
  expect_s3_class(evaluate_graph_model(fit, gs[3:4]), "data.frame")
})

test_that("training rejects overlapping masks set after construction", {
  skip_if_no_torch()
  g <- split_nodes(dataset_karate(), seed = 1)
  g$val_mask <- g$train_mask
  expect_error(train_graph_model(gnn_model("sage", 34, 4, 2), g, "node_classification", epochs = 1),
               "overlap")
})

test_that("corrupted tails stay filtered on large graphs", {
  p <- rbind(50000L, 1L)
  out <- corrupt_tails(p, 50000L, 49997L, known = cbind(p, rbind(50000L, 2L)), seed = 1)
  expect_false(any(out %in% c(1L, 2L, 50000L)))
  expect_error(corrupt_tails(p, 50000L, 49998L, known = cbind(p, rbind(50000L, 2L)), seed = 1),
               "Too few")
})

test_that("TGN negatives avoid every partner of the source at that time", {
  for (s in 1:20) {
    neg <- tgn_negatives(4L, c(1L, 1L), c(2L, 3L), s, t = c(5, 5))
    expect_true(all(neg == 4L))
  }
  expect_error(tgn_negatives(3L, c(1L, 1L), c(2L, 3L), 1, t = c(5, 5)), "no negative destination")
  # different times do not exclude each other
  neg <- tgn_negatives(3L, c(1L, 1L), c(2L, 3L), 1, t = c(5, 6))
  expect_equal(neg, c(3L, 2L))
})

test_that("temporal data check lengths and TGN keeps node identities", {
  expect_error(temporal_graph_data(c(1, 1), c(2, 2), 1:2, y = 17), "one entry per event")
  expect_error(temporal_graph_data(c(1, 1), c(2, 2), 1:2, edge_attr = matrix(1, 3, 1)), "one row per event")
  expect_error(temporal_graph_data(1, 2, 1, node_ids = c("a", "a")), "unique")
  skip_if_no_torch()
  tg <- dataset_synthetic_temporal(num_events = 60, seed = 2)
  tg$node_ids <- paste0("n", seq_len(tg$num_nodes))
  sp <- split_temporal(tg)
  sp$val$node_ids <- rev(sp$val$node_ids)
  expect_error(train_tgn(tgn_model(tg$num_nodes, 4, 2, 4), sp, epochs = 1), "lists its nodes")
  sp <- split_temporal(tg)
  fit <- train_tgn(tgn_model(tg$num_nodes, 4, 2, 4), sp, epochs = 1, seed = 1)
  later <- temporal_graph_data(1, 2, fit$last_time + 1, num_nodes = tg$num_nodes,
                               node_ids = rev(tg$node_ids))
  expect_error(predict(fit, later, pairs = rbind(1, 2), time = fit$last_time + 2), "lists its nodes")
})

test_that("zero sized temporal periods stay empty", {
  tg <- temporal_graph_data(c(1, 2, 3, 1), c(2, 3, 1, 3), 1:4)
  sp <- split_temporal(tg, val = 0, test = 0)
  expect_equal(c(length(sp$train$t), length(sp$val$t), length(sp$test$t)), c(4L, 0L, 0L))
  sp <- split_temporal(tg, val = 0.25, test = 0)
  expect_equal(length(sp$test$t), 0L)
})

test_that("one node forecasts keep one row per origin", {
  skip_if_no_torch()
  sig <- dataset_synthetic_signal(num_nodes = 1, num_steps = 12, seed = 1)
  fit <- train_spatiotemporal_model(sig, lags = 1, hidden_dim = 2, epochs = 1, seed = 1)
  p <- predict(fit, sig, ends = 9:10)
  expect_equal(dim(p), c(2L, 1L))
})

test_that("snapshot sequences and dynamic link models share one causal, consistent protocol", {
  g <- graph_data(x = diag(3), edge_index = rbind(1, 2))
  expect_error(snapshot_sequence(list(g, g, g), start = c(0, 1, 2), end = c(10, 2, 3)), "before the next")
  expect_error(snapshot_sequence(list(g, g), start = c(0, 1), end = c(1)), "one non missing")
  expect_error(snapshot_sequence(list(g, g), start = c(0, 2), end = c(1, 1)), "end at or after")
  # interaction counts drive the history baseline
  tg <- temporal_graph_data(c(rep(1, 10), 3, 1, 3), c(rep(2, 10), 4, 2, 4),
                            c(seq(0.1, 1, length.out = 10), 0.5, 1.5, 2.5), num_nodes = 4)
  sn <- temporal_snapshots(tg, breaks = c(0, 1.2, 2, 3))
  expect_equal(max(sn$graphs[[1]]$edge_weight), 10)
  # undirected positives are counted once per pair
  ug <- graph_data(num_nodes = 3, edge_index = rbind(c(1, 2), c(2, 3)), directed = FALSE)
  pr <- snapshot_link_pairs(ug, 3, 0.5, 1)
  expect_equal(ncol(pr$pos), 2L)
  expect_equal(ncol(pr$neg), 1L)
  skip_if_no_torch()
  # mixed feature presence is rejected before training
  e <- graph_data(edge_index = rbind(1, 2), num_nodes = 4)
  f <- graph_data(x = diag(4), edge_index = rbind(c(1, 3), c(2, 4)))
  expect_error(train_dynamic_link_model(snapshot_sequence(list(f, e, f)), hidden_dim = 2, epochs = 1),
               "Either every snapshot")
  # an edgeless evaluation snapshot is skipped by model and baseline alike
  h <- graph_data(x = diag(4), edge_index = matrix(integer(), 2, 0), num_nodes = 4)
  fit <- train_dynamic_link_model(snapshot_sequence(list(f, f, h)), train_steps = 2, hidden_dim = 2,
                                  epochs = 1, seed = 1)
  expect_true(all(is.na(fit$baseline_metrics)))
})
