path_graph <- function() graph_data(edge_index = rbind(1:3, 2:4), num_nodes = 4, directed = FALSE)

test_that("graph_diffusion matches the matrix exponential solution", {
  skip_if_not_installed("modelcompR")
  g <- path_graph()
  u0 <- c(1, 0, 0, 2)
  times <- c(0, 0.5, 2)
  u <- graph_diffusion(g, u0, times, rate = 0.7, abstol = 1e-10, reltol = 1e-10)
  L <- as.matrix(graphlearnR:::graph_laplacian(g, "laplacian"))
  e <- eigen(L, symmetric = TRUE)
  ref <- t(vapply(times, function(t) {
    as.vector(e$vectors %*% (exp(-0.7 * t * e$values) * crossprod(e$vectors, u0)))
  }, numeric(4)))
  expect_equal(unname(u), ref, tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(rowSums(u), rep(sum(u0), 3), tolerance = 1e-8)
  expect_equal(attr(u, "times"), times)
  un <- graph_diffusion(g, u0, c(0, 50), operator = "normalized_laplacian")
  expect_equal(dim(un), c(2L, 4L))
  expect_error(graph_diffusion(g, 1, times), "one value per node")
  expect_error(graph_diffusion(g, u0, c(1, 0)), "increasing")
})

test_that("graph_operator_symbolic is exact and matches gcn_conv", {
  skip_if_not_installed("symcompR")
  skip_if_no_torch()
  g <- path_graph()
  P <- graph_operator_symbolic(g, "gcn")
  expect_equal(format(P[1, 2]), format(sqrt(symcompR::Integer(6)) / 6))
  layer <- gcn_conv(4, 4, bias = FALSE)
  torch::with_no_grad(layer$lin$weight$copy_(torch::torch_eye(4)))
  out <- as_mat(layer(torch::torch_eye(4), graph_tensors(g)$edge_index))
  expect_equal(out, as.matrix(P), tolerance = 1e-6)
  Lsym <- as.matrix(graph_operator_symbolic(g, "laplacian"))
  expect_equal(Lsym, as.matrix(graphlearnR:::graph_laplacian(g, "laplacian")), ignore_attr = TRUE)
  expect_equal(as.matrix(graph_operator_symbolic(g, "adjacency")),
               unname(t(as.matrix(adjacency_from_graph(g, weighted = FALSE)))))
})

test_that("consistent_labels finds the best labeling that obeys the rules", {
  skip_if_not_installed("z3r")
  set.seed(1)
  n <- 5
  p <- matrix(stats::runif(n * 3), n, 3)
  p <- p / rowSums(p)
  colnames(p) <- c("a", "b", "c")
  p[, "a"] <- p[, "a"] + 1
  p <- p / rowSums(p)
  ei <- rbind(c(1, 2, 2, 3, 3, 4, 4, 5), c(2, 1, 3, 2, 4, 3, 5, 4))
  forb <- rbind(c("a", "a"), c("b", "c"))
  res <- consistent_labels(p, ei, forb)
  expect_equal(res$status, "sat")
  # brute force over all 3^5 labelings
  grid <- as.matrix(expand.grid(rep(list(1:3), n)))
  ok <- apply(grid, 1, function(l) {
    all(!(l[ei[1, ]] == 1 & l[ei[2, ]] == 1)) && all(!(l[ei[1, ]] == 2 & l[ei[2, ]] == 3))
  })
  score <- apply(grid, 1, function(l) sum(log(p[cbind(1:n, l)])))
  best <- grid[ok, , drop = FALSE][which.max(score[ok]), ]
  expect_equal(as.integer(res$labels), unname(best))
  expect_equal(res$log_prob, max(score[ok]), tolerance = 1e-9)
  expect_true(all(max.col(p) == 1))
  expect_gt(length(res$changed), 0)
  fixed <- consistent_labels(p, ei, forb, fixed = c("2" = "a"))
  expect_equal(as.character(fixed$labels[2]), "a")
  expect_false(any(fixed$labels[c(1, 3)] == "a"))
  bad <- consistent_labels(p[1:2, ], rbind(1, 2), forbidden = cbind(c("a", "a", "a", "b", "b", "b", "c", "c", "c"),
                                                                  c("a", "b", "c", "a", "b", "c", "a", "b", "c")))
  expect_equal(bad$status, "unsat")
  expect_error(consistent_labels(unname(p), ei, forb), "column names")
  expect_error(consistent_labels(p, ei, cbind("a", "z")), "class names")
})

test_that("select_links solves the budgeted selection exactly", {
  skip_if_not_installed("optcompR")
  set.seed(2)
  pairs <- utils::combn(6, 2)
  s <- round(stats::runif(ncol(pairs)), 3)
  res <- select_links(pairs, s, budget = 4, max_new_degree = 1)
  expect_equal(res$status, "optimal")
  # brute force: all subsets of at most 4 pairs that form a matching
  best <- 0
  for (k in 1:4) for (cmb in utils::combn(ncol(pairs), k, simplify = FALSE)) {
    if (!anyDuplicated(as.vector(pairs[, cmb]))) best <- max(best, sum(s[cmb]))
  }
  expect_equal(res$objective, best, tolerance = 1e-9)
  expect_false(anyDuplicated(as.vector(res$pairs)) > 0)
  free <- select_links(pairs, s, budget = 3)
  expect_equal(sort(free$selected), sort(order(-s)[1:3]))
  expect_length(select_links(pairs, s, budget = 3, min_score = 2)$selected, 0)
  expect_error(select_links(pairs, s[-1], 2), "one value per pair")
})

test_that("aggregate_jax matches torch aggregation and its gradient", {
  skip_if_not_installed("jaxR")
  skip_if_not(isTRUE(tryCatch(jaxR:::backend_available(), error = function(e) FALSE)),
              "no jaxR XLA runtime")
  skip_if_no_torch()
  g <- random_graph(6, 15, 3, seed = 3)
  w <- seq(0.5, 2, length.out = 15)
  for (a in c("sum", "mean")) {
    got <- as.matrix(aggregate_jax(g$x, g$edge_index, aggr = a, edge_weight = w))
    tt <- graph_tensors(g)
    ref <- as_mat(message_passing(tt$x, tt$edge_index, aggr = a,
                                  edge_weight = torch::torch_tensor(w)))
    expect_equal(unname(got), ref, tolerance = 1e-5, info = a)
  }
  f <- function(x) jaxR::jnp_sum(aggregate_jax(x, g$edge_index, num_nodes = 6L, aggr = "mean")^2)
  gj <- as.matrix(jaxR::grad(f)(jaxR::as_jax(g$x)))
  xt <- torch::torch_tensor(g$x, requires_grad = TRUE)
  message_passing(xt, graph_tensors(g)$edge_index, aggr = "mean")$pow(2)$sum()$backward()
  expect_equal(unname(gj), as_mat(xt$grad), tolerance = 1e-5)
})

test_that("luz trains a node model on the masked nodes", {
  skip_if_not_installed("luz")
  skip_if_no_torch()
  g <- split_nodes(dataset_sbm(c(40, 40), seed = 6), seed = 6)
  dl <- luz_node_dataloader(g, "train")
  b <- torch::dataloader_next(torch::dataloader_make_iter(dl))
  expect_equal(as.integer(as.array(b[[1]]$idx)), which(g$train_mask))
  expect_equal(length(b[[2]]), sum(g$train_mask))
  torch::torch_manual_seed(1)
  fitted <- luz::fit(
    luz::set_opt_hparams(luz::set_hparams(
      luz::setup(luz_node_module, loss = torch::nn_cross_entropy_loss(), optimizer = torch::optim_adam,
                 metrics = list(luz::luz_metric_accuracy())),
      model = gnn_model("sage", ncol(g$x), 16, 2)), lr = 0.01),
    dl, epochs = 40, valid_data = luz_node_dataloader(g, "val"), verbose = FALSE,
    accelerator = luz::accelerator(cpu = TRUE))
  met <- luz::get_metrics(fitted)
  acc <- met$value[met$set == "valid" & met$metric == "acc"]
  expect_gt(tail(acc, 1), 0.8)
  expect_error(luz_node_dataloader(dataset_karate(), "train"), "split_nodes")
})

test_that("augment and glance summarize fits", {
  skip_if_not_installed("generics")
  skip_if_no_torch()
  g <- split_nodes(dataset_karate(), seed = 1)
  fit <- train_graph_model(gnn_model("gcn", 34, 8, 2), g, "node_classification", epochs = 10, seed = 1)
  aug <- generics::augment(fit)
  expect_equal(aug$.truth, g$y)
  expect_equal(sum(aug$.split == "test", na.rm = TRUE), sum(g$test_mask))
  expect_equal(aug$.pred_class, predict(fit, g)$.pred_class)
  gl <- generics::glance(fit)
  expect_equal(nrow(gl), 1L)
  expect_equal(gl$epochs, 10L)
  expect_gt(gl$parameters, 0)
  gs <- dataset_synthetic_graphs(20, seed = 1)
  fg <- train_graph_model(gnn_model("gin", 1, 8, 2, pool = "sum"), gs, "graph_classification",
                          epochs = 2, seed = 1)
  ag <- generics::augment(fg)
  expect_equal(nrow(ag), 20L)
  expect_true(is.factor(ag$.truth))
})
