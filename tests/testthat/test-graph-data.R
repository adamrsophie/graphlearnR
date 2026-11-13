test_that("graph_data stores a directed graph as given", {
  g <- graph_data(x = diag(3), edge_index = rbind(c(1, 2), c(2, 3)), y = factor(c("a", "b", "a")))
  expect_s3_class(g, "graph_data")
  expect_equal(num_nodes(g), 3L)
  expect_equal(num_edges(g), 2L)
  expect_true(is_directed_graph(g))
  expect_equal(g$edge_index, rbind(c(1L, 2L), c(2L, 3L)))
  expect_equal(num_node_features(g), 3L)
})

test_that("undirected input is stored in both directions with pair ids", {
  g <- graph_data(edge_index = rbind(c(1, 2, 3), c(2, 3, 3)), edge_weight = c(1, 2, 3),
                  edge_attr = matrix(1:3, 3, 1), directed = FALSE)
  expect_equal(num_edges(g), 5L)
  expect_equal(num_edges(g, unique_pairs = TRUE), 3L)
  expect_equal(g$edge_index, rbind(c(1L, 2L, 2L, 3L, 3L), c(2L, 1L, 3L, 2L, 3L)))
  expect_equal(g$edge_pair, c(1L, 1L, 2L, 2L, 3L))
  expect_equal(g$edge_weight, c(1, 1, 2, 2, 3))
  expect_equal(g$edge_attr[, 1], c(1, 1, 2, 2, 3))
  expect_true(has_self_loops(g))
})

test_that("reciprocal pairs in undirected input warn", {
  expect_warning(graph_data(edge_index = rbind(c(1, 2), c(2, 1)), directed = FALSE),
                 "reciprocal")
})

test_that("accepted edge index layouts agree", {
  a <- graph_data(edge_index = rbind(c(1, 2, 3), c(2, 3, 1)))
  b <- graph_data(edge_index = cbind(c(1, 2, 3), c(2, 3, 1)))
  d <- graph_data(edge_index = data.frame(from = c(1, 2, 3), to = c(2, 3, 1)))
  expect_equal(a$edge_index, b$edge_index)
  expect_equal(a$edge_index, d$edge_index)
})

test_that("invalid inputs are rejected with clear errors", {
  expect_error(graph_data(edge_index = rbind(1, 5), num_nodes = 3), "outside 1:3")
  expect_error(graph_data(edge_index = rbind(0, 1)), "outside")
  expect_error(graph_data(edge_index = rbind(1.5, 2)), "whole numbers")
  expect_error(graph_data(edge_index = rbind(NA, 2)), "missing")
  expect_error(graph_data(x = diag(3), edge_index = rbind(1, 4)), "x")
  expect_error(graph_data(x = diag(2), y = 1:3), "nodes")
  expect_error(graph_data(edge_index = rbind(1:2, 2:3), edge_weight = 1), "input edges")
  expect_error(graph_data(x = diag(2), train_mask = c(TRUE, NA)), "train_mask")
  expect_error(graph_data(x = diag(2), train_mask = c(TRUE, FALSE), test_mask = c(TRUE, FALSE)),
               "overlap")
  expect_error(graph_data(x = matrix(c(1, NA), 2, 1)), "missing values")
  expect_error(graph_data(x = diag(2), node_ids = c("a", "a")), "unique")
})

test_that("empty graphs, isolated nodes, self loops and duplicates are handled", {
  e <- graph_data(num_nodes = 0)
  expect_equal(num_nodes(e), 0L)
  expect_equal(num_edges(e), 0L)
  g <- graph_data(edge_index = rbind(c(1, 1, 2), c(2, 2, 2)), num_nodes = 4)
  expect_true(has_isolated_nodes(g))
  expect_true(has_duplicate_edges(g))
  expect_true(has_self_loops(g))
  expect_equal(degree(g, "in"), c(0L, 3L, 0L, 0L))
  expect_equal(degree(g, "out"), c(2L, 1L, 0L, 0L))
  s <- summary(g)
  expect_equal(s$isolated_nodes, 2L)
  expect_output(print(s), "isolated nodes: 2")
})

test_that("weighted degree sums weights", {
  g <- graph_data(edge_index = rbind(c(1, 2), c(3, 3)), edge_weight = c(0.5, 2))
  expect_equal(degree(g, "in", weighted = TRUE), c(0, 0, 2.5))
})

test_that("print shows the main fields", {
  g <- split_nodes(dataset_karate(), seed = 1)
  expect_output(print(g), "34 nodes, 156 edges \\(78 undirected pairs\\)")
  expect_output(print(g), "masks: train=")
})

test_that("zero based conversion is the single explicit offset", {
  m <- rbind(c(1L, 2L), c(3L, 4L))
  expect_equal(to_zero_based(m), m - 1L)
  expect_equal(from_zero_based(to_zero_based(m)), m)
  expect_error(to_zero_based(0L), ">= 1")
  expect_error(from_zero_based(-1L), ">= 0")
})

test_that("index checks reject fractional and missing node indices", {
  expect_error(to_zero_based(c(1, 2.5)), "whole numbers")
  expect_error(temporal_graph_data(src = c(1, 2.5), dst = c(2, 3), t = 1:2), "whole numbers")
  e <- data.frame(from = factor(c("b", "a")), to = factor(c("a", "c")))
  g <- graph_from_edgelist(e)
  expect_equal(g$node_ids, c("b", "a", "c"))
})
