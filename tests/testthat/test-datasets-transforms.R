test_that("karate club has the documented structure", {
  g <- dataset_karate()
  expect_equal(num_nodes(g), 34L)
  expect_equal(num_edges(g, TRUE), 78L)
  expect_equal(as.vector(table(g$y)), c(17, 17))
  expect_equal(g$y[c(1, 34)], factor(c("Mr. Hi", "Officer"), levels = c("Mr. Hi", "Officer")))
  expect_equal(dim(g$x), c(34, 34))
  expect_equal(ncol(dataset_karate("degree")$x), 1)
  expect_null(dataset_karate("none")$x)
})

test_that("synthetic datasets are reproducible", {
  expect_equal(dataset_sbm(c(10, 10), seed = 1), dataset_sbm(c(10, 10), seed = 1))
  gs <- dataset_synthetic_graphs(6, seed = 2, task = "regression")
  expect_true(all(vapply(gs, function(g) is.numeric(g$graph_y), logical(1))))
  hg <- dataset_synthetic_hetero(seed = 1)
  expect_equal(unname(num_nodes(hg)), c(60L, 80L))
})

test_that("graph datasets apply transforms lazily and subset", {
  gs <- dataset_synthetic_graphs(5, seed = 1)
  ds <- graph_dataset(gs, transform = add_degree_features, metadata = list(name = "toy"))
  expect_length(ds, 5)
  expect_equal(ncol(ds[[1]]$x), 2)
  expect_equal(ncol(ds$graphs[[1]]$x), 1)
  expect_length(ds[2:3], 2)
  expect_output(print(ds), "toy")
  expect_error(graph_dataset(list(1)), "graph_data")
})

test_that("on disk datasets round trip", {
  dir <- withr::local_tempdir()
  gs <- dataset_synthetic_graphs(4, seed = 3)
  write_graph_dataset(gs, dir, metadata = list(name = "disk"))
  ds <- graph_dataset_disk(dir)
  expect_length(ds, 4)
  expect_equal(ds[[3]], gs[[3]])
  expect_equal(ds$metadata$name, "disk")
  expect_length(ds[c(1, 4)], 2)
  expect_output(print(ds), "on disk")
  expect_error(graph_dataset_disk(withr::local_tempdir()), "No graph files")
})

test_that("TU and LINQS readers parse the file formats", {
  tu <- read_tu_dataset(system.file("extdata", "TOY", package = "graphlearnR"), "TOY")
  expect_length(tu, 2)
  expect_equal(num_nodes(tu[[1]]), 3L)
  expect_equal(tu[[2]]$edge_index, rbind(c(1L, 2L), c(2L, 1L)))
  expect_equal(ncol(tu[[1]]$x), 2)
  expect_equal(as.character(tu[[2]]$graph_y), "-1")
  cit <- read_linqs_citation(system.file("extdata", "toycite", package = "graphlearnR"), "toycite")
  expect_equal(num_nodes(cit), 3L)
  expect_equal(cit$node_ids, c("p1", "p2", "p3"))
  expect_equal(cit$edge_index, rbind(c(2L, 3L), c(1L, 1L)))
  expect_equal(cit$metadata$dropped_edges, 1L)
  expect_equal(levels(cit$y), c("A", "B"))
})

test_that("transforms add loops, coalesce, subset and normalize", {
  g <- graph_data(x = matrix(c(1, 2, 3, 4, 5, 6), 3, 2), edge_index = rbind(c(1, 1, 2, 3), c(2, 2, 3, 3)),
                  edge_weight = c(1, 2, 3, 4))
  c1 <- coalesce_edges(g)
  expect_equal(num_edges(c1), 3L)
  expect_equal(c1$edge_weight, c(3, 3, 4))
  l <- add_self_loops(c1)
  expect_equal(sum(l$edge_index[1, ] == l$edge_index[2, ]), 3L)
  expect_equal(num_edges(remove_self_loops(l)), 2L)
  s <- induced_subgraph(g, c(3, 2))
  expect_equal(s$edge_index, rbind(c(2L, 1L), c(1L, 1L)))
  expect_equal(s$x, g$x[c(3, 2), ])
  n <- normalize_features(g, stats_mask = c(TRUE, TRUE, FALSE))
  expect_equal(colMeans(n$x[1:2, ]), c(0, 0))
  r <- normalize_features(g, "row")
  expect_equal(rowSums(r$x), c(1, 1, 1))
  u <- as_undirected(graph_data(edge_index = rbind(c(1, 2, 2), c(2, 1, 3))))
  expect_false(u$directed)
  expect_equal(num_edges(u, TRUE), 2L)
  d <- add_degree_features(dataset_karate("none"), one_hot = TRUE, max_degree = 5)
  expect_equal(ncol(d$x), 6)
  tf <- compose_transforms(add_self_loops, coalesce_edges)
  expect_s3_class(tf(g), "graph_data")
  expect_error(induced_subgraph(g, c(1, 1)), "unique")
})

test_that("datasets reject graphs with different feature schemas", {
  gs <- dataset_synthetic_graphs(3, seed = 1)
  expect_silent(graph_dataset(gs))
  bad <- gs
  bad[[2]]$x <- cbind(bad[[2]]$x, 1)
  expect_error(graph_dataset(bad), "Graph 2 .*x")
  bad <- gs
  bad[[3]]$graph_y <- NULL
  expect_error(graph_dataset(bad), "graph_y")
  expect_equal(graph_schema(gs[[1]])$graph_y, graph_schema(gs[[2]])$graph_y)
})

test_that("read_linqs_pubmed parses the sparse word weight format", {
  dir <- withr::local_tempdir()
  writeLines(c("NODE\tpaper",
               "cat=1,2,3:label\tnumeric:w-a:0.0\tnumeric:w-b:0.0\tstring:summary",
               "11\tlabel=1\tw-a=0.5\tsummary=w-a",
               "22\tlabel=3\tw-b=0.25\tw-a=0.1\tsummary=w-b,w-a",
               "33\tlabel=2\tsummary="),
             file.path(dir, "Pubmed-Diabetes.NODE.paper.tab"))
  writeLines(c("DIRECTED\tcites", "NO_FEATURES",
               "1\tpaper:22\t|\tpaper:11", "2\tpaper:33\t|\tpaper:11", "3\tpaper:44\t|\tpaper:11"),
             file.path(dir, "Pubmed-Diabetes.DIRECTED.cites.tab"))
  g <- read_linqs_pubmed(dir)
  expect_equal(unname(g$x), rbind(c(0.5, 0), c(0.1, 0.25), c(0, 0)))
  expect_equal(colnames(g$x), c("w-a", "w-b"))
  expect_equal(as.character(g$y), c("1", "3", "2"))
  expect_equal(g$edge_index, rbind(c(2L, 3L), c(1L, 1L)))
  expect_equal(g$metadata$dropped_edges, 1L)
})
