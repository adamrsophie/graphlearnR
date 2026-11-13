# graphlearnR side of the comparison with PyTorch Geometric.
# Writes the random graphs and the SAGE and GCN weights, checks nothing,
# and times one forward and backward pass per layer with the same settings
# as compare-pyg.py. Usage: Rscript benchmarks/pyg/compare-r.R <dir> <threads>
suppressPackageStartupMessages({
  library(graphlearnR)
  library(torch)
})
args <- commandArgs(TRUE)
out_dir <- args[1]
threads <- as.integer(args[2])
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
torch_set_num_threads(threads)

time_it <- function(f, reps = 5) {
  f()
  stats::median(vapply(seq_len(reps), function(i) system.time(f())[["elapsed"]], numeric(1)))
}
wcsv <- function(m, f) utils::write.table(m, file.path(out_dir, f), sep = ",", row.names = FALSE,
                                          col.names = FALSE)

rows <- list()
sizes <- if (length(args) > 2L) as.integer(strsplit(args[3], ",")[[1]]) else c(10000L, 50000L)
for (n in sizes) {
  m <- 10L * n
  set.seed(1)
  ei <- rbind(sample.int(n, m, TRUE), sample.int(n, m, TRUE))
  x <- matrix(round(rnorm(n * 32), 6), n, 32)
  wcsv(t(ei) - 1L, sprintf("edges_%d.csv", n))
  wcsv(x, sprintf("x_%d.csv", n))
  g <- graph_data(x = x, edge_index = ei, num_nodes = n)
  tt <- graph_tensors(g)
  torch_manual_seed(1)
  layers <- list(sage = sage_conv(32, 64), gcn = gcn_conv(32, 64), gat = gat_conv(32, 16, heads = 4),
                 gin = gin_conv(graph_mlp(c(32, 64, 64))))
  if (n == sizes[1]) {
    wcsv(as.matrix(layers$sage$lin_l$weight$detach()), "sage_lin_l_weight.csv")
    wcsv(as.numeric(as.array(layers$sage$lin_l$bias$detach())), "sage_lin_l_bias.csv")
    wcsv(as.matrix(layers$sage$lin_r$weight$detach()), "sage_lin_r_weight.csv")
    wcsv(as.matrix(layers$gcn$lin$weight$detach()), "gcn_lin_weight.csv")
    wcsv(as.matrix(layers$sage(tt$x, tt$edge_index)$detach()), "sage_out_r.csv")
    wcsv(as.matrix(layers$gcn(tt$x, tt$edge_index)$detach()), "gcn_out_r.csv")
  }
  for (a in names(layers)) {
    f <- function() layers[[a]](tt$x, tt$edge_index)$sum()$backward()
    rows[[length(rows) + 1L]] <- data.frame(library = "graphlearnR", layer = a, nodes = n, edges = m,
                                            median_seconds = time_it(f))
  }
}
res <- do.call(rbind, rows)
print(res, row.names = FALSE)
utils::write.csv(res, file.path(out_dir, "timing-r.csv"), row.names = FALSE)
writeLines(c(paste("R torch", as.character(packageVersion("torch"))),
             paste("threads:", torch_get_num_threads())), file.path(out_dir, "versions-r.txt"))
