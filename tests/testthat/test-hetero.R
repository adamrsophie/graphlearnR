toy_hetero <- function() {
  hetero_graph_data(
    nodes = list(
      author = list(x = matrix(1:6, 3, 2), y = factor(c("x", "y", "x")),
                    train_mask = c(TRUE, TRUE, FALSE), test_mask = c(FALSE, FALSE, TRUE)),
      paper = matrix(1:12, 4, 3),
      venue = 2
    ),
    edges = list(
      author__writes__paper = rbind(c(1, 1, 2, 3), c(1, 2, 3, 4)),
      paper__cites__paper = rbind(c(2, 3), c(1, 1)),
      list(src = "paper", rel = "published_in", dst = "venue", edge_index = rbind(1:4, c(1, 1, 2, 2)))
    )
  )
}

test_that("heterogeneous graphs store typed nodes and relations", {
  hg <- toy_hetero()
  expect_s3_class(hg, "hetero_graph_data")
  expect_equal(num_nodes(hg), c(author = 3L, paper = 4L, venue = 2L))
  expect_equal(unname(num_edges(hg)), c(4L, 2L, 4L))
  et <- edge_types(hg)
  expect_equal(et$src, c("author", "paper", "paper"))
  expect_equal(et$dst, c("paper", "paper", "venue"))
  expect_output(print(hg), "3 node types, 3 relations")
})

test_that("cross type indices are validated per type", {
  expect_error(hetero_graph_data(nodes = list(a = diag(2), b = diag(3)),
                                 edges = list(a__r__b = rbind(3, 1))), "source indices must be in 1:2")
  expect_error(hetero_graph_data(nodes = list(a = diag(2), b = diag(3)),
                                 edges = list(a__r__b = rbind(1, 4))), "target indices must be in 1:3")
  expect_error(hetero_graph_data(nodes = list(a = diag(2)), edges = list(a__r__c = rbind(1, 1))),
               "unknown target")
  expect_error(hetero_graph_data(nodes = list(a = diag(2)), edges = list(bad = rbind(1, 1))),
               "src__relation__dst")
  expect_error(hetero_graph_data(nodes = list(a__b = diag(2)), edges = list()), "must not contain")
})

test_that("reverse relations flip direction and are paired", {
  hg <- add_reverse_relations(toy_hetero())
  expect_true("paper__rev_writes__author" %in% names(hg$edges))
  r <- hg$edges$paper__rev_writes__author
  expect_equal(r$edge_index, hg$edges$author__writes__paper$edge_index[2:1, ])
  expect_equal(hg$edges$author__writes__paper$reverse, "paper__rev_writes__author")
  expect_equal(r$reverse_of, "author__writes__paper")
  again <- add_reverse_relations(hg)
  expect_equal(names(again$edges), names(hg$edges))
})

test_that("to_homogeneous maps typed indices through offsets", {
  hg <- toy_hetero()
  g <- to_homogeneous(hg, target_type = "author")
  ptr <- g$metadata$type_ptr
  expect_equal(unname(ptr), c(0L, 3L, 7L))
  expect_equal(num_nodes(g), 9L)
  e <- hg$edges$author__writes__paper$edge_index
  cols <- which(g$edge_type == "author__writes__paper")
  expect_equal(g$edge_index[, cols], rbind(e[1, ] + 0L, e[2, ] + 3L))
  expect_equal(g$x[4:7, ], hg$nodes$paper$x)
  expect_equal(g$x[1:3, 3], c(0, 0, 0))
  expect_equal(as.character(g$y[1:3]), c("x", "y", "x"))
  expect_true(all(is.na(g$y[4:9])))
  expect_equal(g$train_mask, c(TRUE, TRUE, FALSE, rep(FALSE, 6)))
  expect_equal(unname(g$metadata$feature_dims), c(2L, 3L, 0L))
})

test_that("hetero_conv aggregates relations into each target type", {
  skip_if_no_torch()
  hg <- toy_hetero()
  ht <- hetero_tensors(hg)
  x <- list(author = torch::torch_ones(3, 2), paper = torch::torch_ones(4, 2) * 2,
            venue = torch::torch_zeros(2, 2))
  sum_layer <- torch::nn_module(
    initialize = function() {},
    forward = function(x, edge_index) message_passing(x, edge_index, aggr = "sum")
  )
  conv <- hetero_conv(list(author__writes__paper = sum_layer(), paper__cites__paper = sum_layer()))
  out <- conv(x, ht$edge_index)
  # paper 1 receives author 1 (1) and papers 2, 3 (2 + 2)
  expect_equal(as_mat(out$paper)[, 1], c(1 + 4, 1, 1, 1))
  expect_equal(as_mat(out$author), as_mat(x$author))
  expect_equal(as_mat(out$venue), as_mat(x$venue))
})

test_that("heterogeneous models learn drug classes from typed relations", {
  skip_if_no_torch()
  hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 1))
  for (cv in c("rgcn", "sage", "gat")) {
    m <- hetero_gnn_model(hg, hidden_dim = 16, output_dim = 3, target_type = "drug", conv = cv,
                          dropout = 0.1, heads = 2)
    fit <- train_graph_model(m, hg, "node_classification", target_type = "drug", epochs = 60,
                             lr = 0.02, seed = 1)
    ev <- evaluate_graph_model(fit, hg)
    expect_gt(ev$value[ev$metric == "accuracy"], 0.6)
  }
  p <- predict(fit, hg)
  expect_equal(nrow(p), hg$nodes$drug$num_nodes)
  expect_error(train_graph_model(m, hg, "node_classification"), "target_type")
})

test_that("featureless node types get learned embeddings", {
  skip_if_no_torch()
  hg <- toy_hetero()
  m <- hetero_gnn_model(hg, hidden_dim = 4, output_dim = 2, target_type = "author")
  out <- m(hetero_tensors(hg))
  expect_equal(dim(out$author), c(3, 2))
  expect_equal(dim(out$venue), c(2, 4))
})

test_that("hetero link splits remove held out edges from the reverse relation", {
  hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 2))
  sp <- split_edges_hetero(hg, "drug__targets__protein", val = 0.1, test = 0.2, seed = 2)
  rev <- sp$train_graph$edges$protein__rev_targets__drug$edge_index
  fwd <- sp$train_graph$edges$drug__targets__protein$edge_index
  held <- cbind(sp$val_pos, sp$test_pos)
  hk <- paste(held[1, ], held[2, ])
  expect_false(any(paste(fwd[1, ], fwd[2, ]) %in% hk))
  expect_false(any(paste(rev[2, ], rev[1, ]) %in% hk))
  expect_equal(ncol(rev), ncol(fwd))
  allk <- paste(sp$all_pos[1, ], sp$all_pos[2, ])
  expect_false(any(paste(sp$test_neg[1, ], sp$test_neg[2, ]) %in% allk))
  expect_true(all(sp$test_neg[1, ] <= 60 & sp$test_neg[2, ] <= 80))
  expect_output(print(sp), "drug__targets__protein")
})

test_that("relation specific link prediction trains", {
  skip_if_no_torch()
  hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 3))
  sp <- split_edges_hetero(hg, "drug__targets__protein", seed = 3)
  torch::torch_manual_seed(1)
  m <- hetero_gnn_model(hg, hidden_dim = 16, conv = "sage", dropout = 0)
  fit <- train_graph_model(m, sp, "link_prediction", epochs = 40, seed = 3)
  ev <- evaluate_graph_model(fit)
  expect_gt(ev$value[ev$metric == "roc_auc"], 0.6)
  s <- predict(fit, pairs = sp$test_pos, type = "prob")
  expect_length(s, ncol(sp$test_pos))
})

test_that("typed neighbor sampling respects types and fanouts", {
  hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 4))
  sg <- sample_hetero_neighbors(hg, seeds = list(drug = c(3, 1)), fanouts = c(2, 2), seed = 1)
  expect_s3_class(sg, "sampled_hetero_graph")
  expect_equal(sg$n_id$drug[1:2], c(3L, 1L))
  expect_equal(sg$num_seeds[["drug"]], 2L)
  for (k in names(sg$edges)) {
    e <- sg$edges[[k]]
    orig <- hg$edges[[k]]$edge_index
    if (!ncol(e$edge_index)) next
    mapped <- paste(sg$n_id[[e$src]][e$edge_index[1, ]], sg$n_id[[e$dst]][e$edge_index[2, ]])
    expect_true(all(mapped %in% paste(orig[1, ], orig[2, ])), info = k)
  }
  first <- sg$edges$protein__rev_targets__drug$edge_index
  expect_true(all(table(first[2, first[2, ] %in% 1:2]) <= 2))
  per_rel <- sample_hetero_neighbors(hg, list(drug = 1), fanouts = list(protein__rev_targets__drug = 1))
  expect_equal(ncol(per_rel$edges$protein__interacts__protein$edge_index), 0)
})

test_that("heterogeneous batching offsets each type separately", {
  a <- toy_hetero()
  hb <- hetero_batch(list(a, a))
  expect_equal(num_nodes(hb), c(author = 6L, paper = 8L, venue = 4L))
  e <- hb$edges$author__writes__paper$edge_index
  expect_equal(e[, 5:8], a$edges$author__writes__paper$edge_index + c(3L, 4L))
  expect_equal(hb$batch$paper, rep(1:2, each = 4))
  expect_error(hetero_batch(list(a, add_reverse_relations(a))), "same node types")
})

test_that("splitting a generated reverse relation removes held out forward links", {
  hg <- add_reverse_relations(dataset_synthetic_hetero(seed = 2))
  sp <- split_edges_hetero(hg, "protein__rev_targets__drug", val = 0.1, test = 0.2, seed = 2)
  held <- paste(c(sp$val_pos[2, ], sp$test_pos[2, ]), c(sp$val_pos[1, ], sp$test_pos[1, ]))
  fwd <- sp$train_graph$edges$drug__targets__protein$edge_index
  expect_length(intersect(held, paste(fwd[1, ], fwd[2, ])), 0L)
})

test_that("same type relations split both orientations together", {
  hg <- dataset_synthetic_hetero(seed = 2)
  sp <- split_edges_hetero(hg, "protein__interacts__protein", val = 0.1, test = 0.2, seed = 2)
  key <- function(m) paste(pmin(m[1, ], m[2, ]), pmax(m[1, ], m[2, ]))
  held <- key(cbind(sp$val_pos, sp$test_pos))
  tr <- sp$train_graph$edges$protein__interacts__protein$edge_index
  expect_length(intersect(held, key(tr)), 0L)
  expect_length(intersect(held, key(sp$train_pos)), 0L)
  neg <- cbind(sp$val_neg, sp$test_neg)
  expect_true(all(neg[1, ] != neg[2, ]))
  expect_length(intersect(key(neg), key(sp$all_pos)), 0L)
})

test_that("featureless node types use the embedding rows of sampled and batched nodes", {
  skip_if_no_torch()
  hg <- hetero_graph_data(
    nodes = list(a = list(x = matrix(rnorm(8), 4, 2)), b = list(num_nodes = 6)),
    edges = list(b__to__a = rbind(c(6, 1, 2, 3), c(4, 1, 2, 3)),
                 a__back__b = rbind(c(4, 1, 2, 3), c(6, 1, 2, 3))))
  torch::torch_manual_seed(1)
  m <- hetero_gnn_model(hg, hidden_dim = 4, num_layers = 1, conv = "sage")
  m$eval()
  full <- as_mat(m(hetero_tensors(hg))$b)
  sg <- sample_hetero_neighbors(hg, list(a = c(4, 1)), c(2, 2), seed = 1)
  out <- m(hetero_tensors(sg))
  expect_equal(out$b$size(1), length(sg$n_id$b))
  hb <- hetero_batch(list(hg, hg))
  outb <- as_mat(m(hetero_tensors(hb))$b)
  expect_equal(outb[1:6, ], full, tolerance = 1e-5)
  expect_equal(outb[7:12, ], full, tolerance = 1e-5)
  expect_error(m(hetero_tensors(hetero_graph_data(
    nodes = list(a = list(x = matrix(1, 4, 2)), b = list(num_nodes = 5)),
    edges = list(b__to__a = rbind(1, 1), a__back__b = rbind(1, 1))))), "original node")
})

test_that("relation names with a double underscore are rejected", {
  expect_error(hetero_graph_data(nodes = list(a = diag(3), b = diag(3)),
                                 edges = list(list(src = "a", rel = "is__part", dst = "b",
                                                   edge_index = rbind(1:3, 1:3)))), "__")
})

test_that("an rgcn hetero model runs when every relation is empty", {
  skip_if_no_torch()
  hg <- hetero_graph_data(nodes = list(a = matrix(1, 3, 2), b = matrix(1, 2, 2)),
                          edges = list(a__to__b = matrix(integer(), 2, 0)))
  m <- hetero_gnn_model(hg, hidden_dim = 4, num_layers = 1, conv = "rgcn")
  expect_equal(m(hetero_tensors(hg))$a$size(1), 3)
})

test_that("unbatch rebuilds heterogeneous graphs exactly", {
  a <- dataset_synthetic_hetero(num_drugs = 30, num_proteins = 40, seed = 1)
  b <- dataset_synthetic_hetero(num_drugs = 24, num_proteins = 33, seed = 2)
  b$metadata$name <- "second"
  k <- names(a$edges)[1]
  a$edges[[k]]$edge_y <- seq_len(ncol(a$edges[[k]]$edge_index))
  b$edges[[k]]$edge_y <- -seq_len(ncol(b$edges[[k]]$edge_index))
  parts <- unbatch(hetero_batch(list(a, b)))
  for (pair in list(list(parts[[1]], a), list(parts[[2]], b))) {
    got <- pair[[1]]; ref <- pair[[2]]
    for (t in ref$node_types) {
      for (f in c("x", "y", "num_nodes", "node_ids", "train_mask", "test_mask")) {
        expect_equal(got$nodes[[t]][[f]], ref$nodes[[t]][[f]], info = paste(t, f))
      }
    }
    for (r in names(ref$edges)) {
      for (f in c("edge_index", "edge_attr", "edge_y", "edge_weight")) {
        expect_equal(got$edges[[r]][[f]], ref$edges[[r]][[f]], info = paste(r, f))
      }
    }
    expect_equal(got$metadata, ref$metadata)
  }
})
