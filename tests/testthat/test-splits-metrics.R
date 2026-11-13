test_that("node splits are disjoint, sized and stratified", {
  g <- split_nodes(dataset_sbm(c(30, 30, 30), seed = 1), 0.5, 0.2, 0.3, seed = 2)
  expect_equal(sum(g$train_mask), 45)
  expect_equal(sum(g$val_mask), 18)
  expect_equal(sum(g$test_mask), 27)
  expect_false(any(g$train_mask & g$val_mask | g$train_mask & g$test_mask | g$val_mask & g$test_mask))
  expect_equal(as.vector(table(g$y[g$train_mask])), c(15, 15, 15))
  h <- split_nodes(dataset_karate(), train = 10, val = 5, test = 5, stratify = FALSE, seed = 1)
  expect_equal(sum(h$train_mask), 10)
  k <- dataset_karate()
  k$y[1:4] <- NA
  k <- split_nodes(k, seed = 1)
  expect_false(any(k$train_mask[1:4] | k$val_mask[1:4] | k$test_mask[1:4]))
})

test_that("edge splits never leak held out pairs into the training graph", {
  g <- dataset_sbm(c(30, 30), p_in = 0.2, seed = 3)
  sp <- split_edges(g, val = 0.1, test = 0.2, seed = 3)
  tg <- sp$train_graph
  train_keys <- c(paste(tg$edge_index[1, ], tg$edge_index[2, ]))
  for (s in c("val_pos", "test_pos")) {
    p <- sp[[s]]
    expect_false(any(paste(p[1, ], p[2, ]) %in% train_keys), info = s)
    expect_false(any(paste(p[2, ], p[1, ]) %in% train_keys), info = s)
  }
  all_keys <- c(paste(sp$all_pos[1, ], sp$all_pos[2, ]), paste(sp$all_pos[2, ], sp$all_pos[1, ]))
  for (s in c("val_neg", "test_neg")) {
    expect_false(any(paste(sp[[s]][1, ], sp[[s]][2, ]) %in% all_keys), info = s)
  }
  expect_equal(ncol(sp$train_pos) + ncol(sp$val_pos) + ncol(sp$test_pos), num_edges(g, TRUE))
  expect_equal(num_edges(tg), 2 * ncol(sp$train_pos))
  expect_equal(tg$x, g$x)
  expect_output(print(sp), "link_split")
})

test_that("directed edge splits keep direction", {
  g <- random_graph(n = 30, m = 80, seed = 4)
  g <- coalesce_edges(remove_self_loops(g))
  sp <- split_edges(g, seed = 4)
  expect_true(sp$directed)
  expect_equal(num_edges(sp$train_graph), ncol(sp$train_pos))
})

test_that("edge label splits keep both directions together", {
  g <- dataset_karate()
  g$edge_y <- withr::with_seed(1, factor(sample(c("a", "b"), num_edges(g), TRUE)))
  g$edge_y <- g$edge_y[match(g$edge_pair, g$edge_pair)]
  g <- split_edge_labels(g, seed = 1)
  for (m in c("edge_train_mask", "edge_val_mask", "edge_test_mask")) {
    by_pair <- tapply(g[[m]], g$edge_pair, function(v) length(unique(v)))
    expect_true(all(by_pair == 1), info = m)
  }
  expect_false(any(g$edge_train_mask & g$edge_test_mask))
})

test_that("graph splits support stratified, group and scaffold modes", {
  sp <- split_graphs(50, seed = 1)
  expect_equal(lengths(sp), c(train = 40, val = 5, test = 5))
  expect_setequal(unlist(sp), 1:50)
  groups <- rep(letters[1:10], each = 5)
  gs <- split_graphs(50, method = "group", groups = groups, seed = 2)
  for (a in names(gs)) for (b in setdiff(names(gs), a)) {
    expect_length(intersect(groups[gs[[a]]], groups[gs[[b]]]), 0)
  }
  sc <- split_graphs(10, 0.6, 0.2, 0.2, method = "scaffold",
                     groups = c(rep("A", 5), rep("B", 2), "C", "D", "E"))
  # Largest groups first; a group that does not fit the train quota goes to
  # validation, then test (the DeepChem rule), so the single "C" fills train.
  expect_equal(sort(sc$train), c(1:5, 8L))
  expect_equal(sort(sc$val), 6:7)
  expect_equal(sort(sc$test), 9:10)
  expect_identical(sc, split_graphs(10, 0.6, 0.2, 0.2, method = "scaffold",
                                    groups = c(rep("A", 5), rep("B", 2), "C", "D", "E")))
  expect_error(split_graphs(5, method = "group"), "groups")
})

test_that("classification metrics match hand calculations", {
  t <- factor(c("a", "a", "b", "b", "c"))
  e <- factor(c("a", "b", "b", "b", "a"), levels = c("a", "b", "c"))
  expect_equal(metric_accuracy(t, e), 3 / 5)
  expect_equal(metric_balanced_accuracy(t, e), mean(c(0.5, 1, 0)))
  pa <- 1 / 2; pb <- 2 / 3; pc <- 0
  ra <- 1 / 2; rb <- 1; rc <- 0
  expect_equal(metric_precision(t, e), mean(c(pa, pb, pc)))
  expect_equal(metric_recall(t, e), mean(c(ra, rb, rc)))
  f <- function(p, r) if (p + r == 0) 0 else 2 * p * r / (p + r)
  expect_equal(metric_f1(t, e), mean(c(f(pa, ra), f(pb, rb), 0)))
  expect_equal(metric_f1(t, e, average = "micro"), 3 / 5)
  expect_equal(metric_precision(factor(c("n", "y", "y")), factor(c("y", "y", "n")),
                                average = "binary", positive = "y"), 1 / 2)
})

test_that("ranking metrics match pairwise definitions", {
  withr::local_seed(1)
  y <- c(rep(TRUE, 20), rep(FALSE, 30))
  s <- c(rnorm(20, 1), rnorm(30))
  s[5] <- s[25]
  pairs <- outer(s[y], s[!y], ">") + 0.5 * outer(s[y], s[!y], "==")
  expect_equal(metric_roc_auc(y, s), mean(pairs))
  ap_ref <- {
    o <- order(-s)
    yy <- y[o]
    mean(cumsum(yy)[yy] / which(yy))
  }
  s2 <- s + seq_along(s) * 1e-9
  expect_equal(metric_average_precision(y, s2), {
    o <- order(-s2); yy <- y[o]; mean(cumsum(yy)[yy] / which(yy))
  })
  expect_equal(metric_hits_at_k(c(0.9, 0.3, 0.6), c(0.5, 0.4, 0.1), k = 1), 2 / 3)
  neg <- matrix(c(0.1, 0.95, 0.2, 0.5, 0.4, 0.4), 3, 2, byrow = TRUE)
  pos <- c(0.9, 0.3, 0.4)
  # one negative above 0.9; one above 0.3; two ties at 0.4 count one half each
  ranks <- c(2, 2, 2)
  expect_equal(metric_mrr(pos, neg), mean(1 / ranks))
  expect_equal(metric_hits_at_k(pos, neg, 1), 0)
  expect_equal(metric_hits_at_k(pos, neg, 2), 1)
  prob <- cbind(a = c(0.9, 0.2, 0.6), b = c(0.1, 0.8, 0.4))
  tr <- factor(c("a", "b", "b"))
  expect_equal(metric_roc_auc(tr, prob), metric_roc_auc(tr == "b", prob[, 2]))
  expect_equal(metric_brier(tr, prob), mean(c(0.01 + 0.01, 0.04 + 0.04, 0.36 + 0.36)))
  expect_error(metric_roc_auc(c(TRUE, TRUE), c(1, 2)), "both")
})

test_that("regression and clustering metrics", {
  t <- c(1, 2, 3, 4)
  e <- c(1.5, 2, 2.5, 4)
  expect_equal(metric_mse(t, e), 0.125)
  expect_equal(metric_rmse(t, e), sqrt(0.125))
  expect_equal(metric_mae(t, e), 0.25)
  expect_equal(metric_r2(t, e), 1 - 0.5 / 5)
  cl <- c(1, 1, 2, 2, 3, 3)
  expect_equal(metric_nmi(cl, c(5, 5, 6, 6, 7, 7)), 1)
  expect_equal(metric_ari(cl, c(5, 5, 6, 6, 7, 7)), 1)
  expect_lt(metric_ari(cl, c(1, 2, 3, 1, 2, 3)), 0.1)
  ece <- metric_ece(factor(c("a", "b")), cbind(c(1, 0), c(0, 1)))
  expect_equal(ece, 0)
})

test_that("evaluate_predictions returns task metric tables", {
  p <- cbind(a = c(0.8, 0.3, 0.6), b = c(0.2, 0.7, 0.4))
  ev <- evaluate_predictions(factor(c("a", "b", "b")), p, "classification")
  expect_equal(ev$value[ev$metric == "accuracy"], 2 / 3)
  rv <- evaluate_predictions(c(1, 2), c(1, 3), "regression")
  expect_equal(rv$value[rv$metric == "mae"], 0.5)
  lv <- evaluate_predictions(NULL, c(0.9, 0.8), "link", neg = c(0.1, 0.85), ks = 1)
  expect_equal(lv$value[lv$metric == "hits@1"], 0.5)
  expect_error(evaluate_predictions(1, 1, "link"), "negative")
})
