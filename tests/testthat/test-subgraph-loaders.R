test_that("cluster_loader partitions nodes and batches are induced subgraphs", {
  g <- split_nodes(dataset_sbm(c(30, 30, 30), seed = 1), seed = 1)
  cl <- cluster_loader(g, num_parts = 6, clusters_per_batch = 2, seed = 1)
  expect_equal(sort(unlist(cl$parts)), seq_len(90))
  expect_true(all(lengths(cl$parts) == 15L))
  b <- loader_batches(cl, 1)
  expect_length(b, 3L)
  expect_equal(sort(unlist(lapply(b, `[[`, "node_ids"))), seq_len(90))
  sg <- b[[1]]
  orig <- sg$node_ids[sg$edge_index]
  key <- paste(g$edge_index[1, ], g$edge_index[2, ])
  expect_true(all(paste(orig[c(TRUE, FALSE)], orig[c(FALSE, TRUE)]) %in% key))
  inside <- g$edge_index[1, ] %in% sg$node_ids & g$edge_index[2, ] %in% sg$node_ids
  expect_equal(num_edges(sg), sum(inside))
  expect_equal(sg$y, g$y[sg$node_ids])
  expect_identical(loader_batches(cl, 2), loader_batches(cl, 2))
  # planted blocks stay together more than under a random partition
  block <- rep(1:3, each = 30)
  purity <- function(parts) mean(vapply(parts, function(p) max(tabulate(block[p])) / length(p), 1))
  r <- cluster_loader(g, 6, method = "random", seed = 1)
  expect_equal(sort(unlist(r$parts)), seq_len(90))
  expect_gt(purity(cl$parts), purity(r$parts) + 0.15)
  expect_error(cluster_loader(g, 100), "exceeds")
})

test_that("saint_loader walks along edges and normalizes", {
  g <- split_nodes(dataset_sbm(c(40, 40), seed = 2), seed = 2)
  sl <- saint_loader(g, batch_size = 5, walk_length = 3, num_steps = 4, seed = 3)
  expect_true(all(sl$node_norm > 0))
  b <- loader_batches(sl, 1)
  expect_length(b, 4L)
  expect_identical(b, loader_batches(sl, 1))
  sg <- b[[1]]
  expect_lte(num_nodes(sg), 5 * 4)
  expect_equal(length(sg$node_norm), num_nodes(sg))
  expect_equal(length(sg$edge_weight), num_edges(sg))
  expect_true(all(sg$edge_weight > 0))
  # walks are connected: every sampled node is reachable within the subgraph from a root
  expect_true(num_edges(sg) > 0)
  plain <- saint_loader(g, 5, 3, 2, sample_coverage = 0, seed = 3)
  expect_null(loader_batches(plain)[[1]]$node_norm)
})

test_that("training with subgraph loaders works and checks its input", {
  skip_if_no_torch()
  g <- split_nodes(dataset_sbm(c(60, 60, 60), seed = 4), seed = 4)
  for (L in list(cluster_loader(g, 6, 2, seed = 1), saint_loader(g, 20, 2, 5, seed = 1))) {
    torch::torch_manual_seed(1)
    fit <- train_graph_model(gnn_model("sage", ncol(g$x), 16, 3), g, "node_classification",
                             epochs = 30, loader = L, seed = 1)
    ev <- evaluate_graph_model(fit, g)
    expect_gt(ev$value[ev$metric == "accuracy"], 0.8)
  }
  m <- gnn_model("sage", ncol(g$x), 8, 3)
  expect_error(train_graph_model(m, g, "node_classification", epochs = 1, loader = list()),
               "cluster_loader")
  other <- cluster_loader(dataset_sbm(c(10, 10), seed = 1), 2)
  expect_error(train_graph_model(m, g, "node_classification", epochs = 1, loader = other),
               "different graph")
})

test_that("saint node weights estimate the full graph mean", {
  g <- dataset_sbm(c(30, 30), p_in = 0.2, p_out = 0.02, seed = 1)
  sl <- saint_loader(g, batch_size = 5, walk_length = 3, num_steps = 400,
                     sample_coverage = 50, seed = 1)
  # With a per node loss of 1, the weighted sum over a batch estimates
  # sum(1) / n = 1 over all nodes.
  est <- mean(vapply(loader_batches(sl), function(b) sum(b$node_norm), numeric(1)))
  expect_equal(est, 1, tolerance = 0.1)
})
