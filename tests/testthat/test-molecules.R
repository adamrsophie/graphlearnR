test_that("molecule_graph encodes atoms and bonds", {
  g <- molecule_graph(c("C", "N", "Xe"), data.frame(from = c(1, 2), to = c(2, 3), order = c(2, 1)))
  expect_equal(num_nodes(g), 3L)
  expect_equal(num_edges(g, TRUE), 2L)
  expect_equal(unname(g$x[, "C"]), c(1, 0, 0))
  expect_equal(unname(g$x[, "other"]), c(0, 0, 1))
  expect_equal(rowSums(g$x), rep(1, 3))
  # each bond appears in both directions with the same features
  ea <- g$edge_attr
  ei <- g$edge_index
  k <- which(ei[1, ] == 2 & ei[2, ] == 1)
  expect_equal(unname(ea[k, "bond_2"]), 1)
  expect_equal(rowSums(ea), rep(1, ncol(ei)))
  expect_error(molecule_graph("C", data.frame(from = 1, to = 1, order = 4)), "Unknown bond order")
  expect_error(molecule_graph(character(), matrix(0, 0, 2)), "empty")
})

test_that("synthetic molecules support scaffold splits without overlap", {
  m <- dataset_synthetic_molecules(60, seed = 1)
  expect_identical(m, dataset_synthetic_molecules(60, seed = 1))
  sc <- attr(m, "scaffold")
  expect_length(sc, 60)
  sp <- split_graphs(60, method = "scaffold", groups = sc)
  expect_length(intersect(sc[sp$train], sc[sp$test]), 0)
  expect_length(intersect(sc[sp$train], sc[sp$val]), 0)
  y <- vapply(m, function(g) as.character(g$graph_y), "")
  expect_true(all(y[sc == "acyclic"] == "chain"))
})

test_that("an edge aware GIN learns ring versus chain", {
  skip_if_no_torch()
  m <- dataset_synthetic_molecules(120, seed = 2)
  sp <- split_graphs(120, labels = vapply(m, function(g) as.character(g$graph_y), ""), seed = 2)
  torch::torch_manual_seed(1)
  mod <- gnn_model("gin", 10, 32, 2, pool = "sum", edge_dim = 4)
  fit <- train_graph_model(mod, m[sp$train], "graph_classification", epochs = 40,
                           batch_size = 16, seed = 1)
  ev <- evaluate_graph_model(fit, m[sp$test])
  expect_gt(ev$value[ev$metric == "accuracy"], 0.75)
})
