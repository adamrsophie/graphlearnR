# Times one forward and backward pass of common layers on random graphs.
# Usage: Rscript benchmarks/bench-message-passing.R <output_dir>
suppressPackageStartupMessages({
  library(graphlearnR)
  library(torch)
})
out_dir <- commandArgs(TRUE)[1]
if (is.na(out_dir)) out_dir <- "."
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

time_it <- function(f, reps = 5) {
  f()
  t <- vapply(seq_len(reps), function(i) system.time(f())[["elapsed"]], numeric(1))
  stats::median(t)
}

rows <- list()
for (n in c(1000, 10000, 50000)) {
  m <- 10 * n
  set.seed(1)
  g <- graph_data(x = matrix(rnorm(n * 32), n, 32),
                  edge_index = rbind(sample.int(n, m, TRUE), sample.int(n, m, TRUE)),
                  num_nodes = n)
  tt <- graph_tensors(g)
  for (arch in c("sage", "gcn", "gat", "gin")) {
    torch_manual_seed(1)
    layer <- switch(arch,
      sage = sage_conv(32, 64),
      gcn = gcn_conv(32, 64),
      gat = gat_conv(32, 16, heads = 4),
      gin = gin_conv(graph_mlp(c(32, 64, 64))))
    f <- function() {
      out <- layer(tt$x, tt$edge_index)
      out$sum()$backward()
    }
    rows[[length(rows) + 1L]] <- data.frame(
      benchmark = "forward_backward", layer = arch, nodes = n, edges = m,
      median_seconds = time_it(f))
  }
}
res <- do.call(rbind, rows)
print(res, row.names = FALSE)
utils::write.csv(res, file.path(out_dir, "message-passing.csv"), row.names = FALSE)
writeLines(c(capture.output(sessionInfo()),
             paste("threads:", torch_get_num_threads()),
             paste("cores:", parallel::detectCores())),
           file.path(out_dir, "message-passing-session.txt"))
