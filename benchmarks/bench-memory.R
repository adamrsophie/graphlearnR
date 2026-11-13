# Times graph construction, conversion and batching, and measures peak
# process memory (resident set size) of a forward and backward pass.
# R's gc() does not see LibTorch allocations, so memory is read from the
# operating system (VmHWM on Linux). The high water mark never resets, so
# each memory case runs in a fresh R process. GPU memory is reported only
# when CUDA is available; the CI runner has no GPU.
# Usage: Rscript benchmarks/bench-memory.R <output_dir>
out_dir <- commandArgs(TRUE)[1]
if (is.na(out_dir)) out_dir <- "."
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

case <- Sys.getenv("GLR_MEM_CASE")
if (nzchar(case)) {
  # Child process: run one case and print its peak memory.
  suppressPackageStartupMessages({
    library(graphlearnR)
    library(torch)
  })
  peak_mb <- function() {
    if (!file.exists("/proc/self/status")) return(NA_real_)
    l <- grep("^VmHWM", readLines("/proc/self/status"), value = TRUE)
    as.numeric(gsub("[^0-9]", "", l)) / 1024
  }
  base <- peak_mb()
  p <- strsplit(case, ":", fixed = TRUE)[[1]]
  arch <- p[1]
  n <- as.integer(p[2])
  m <- 10L * n
  set.seed(1)
  dev <- if (cuda_is_available()) "cuda" else "cpu"
  g <- graph_data(x = matrix(rnorm(n * 32), n, 32),
                  edge_index = rbind(sample.int(n, m, TRUE), sample.int(n, m, TRUE)),
                  num_nodes = n)
  tt <- graph_tensors(g, dev)
  torch_manual_seed(1)
  layer <- switch(arch,
    sage = sage_conv(32, 64),
    gcn = gcn_conv(32, 64),
    gat = gat_conv(32, 16, heads = 4),
    gin = gin_conv(graph_mlp(c(32, 64, 64))))
  layer$to(device = dev)
  out <- layer(tt$x, tt$edge_index)
  out$sum()$backward()
  # Untested: no GPU runner. Falls back to NA if the stats layout differs.
  gpu <- if (dev == "cuda") {
    tryCatch(cuda_memory_stats()$allocated_bytes$all$peak / 2^20, error = function(e) NA_real_)
  } else NA_real_
  cat(sprintf("RESULT %f %f %f\n", base, peak_mb(), gpu))
  quit(save = "no")
}

suppressPackageStartupMessages(library(graphlearnR))
time_it <- function(f, reps = 5) {
  f()
  stats::median(vapply(seq_len(reps), function(i) system.time(f())[["elapsed"]], numeric(1)))
}

rows <- list()
for (n in c(1000, 10000, 100000)) {
  m <- 10 * n
  set.seed(1)
  ei <- rbind(sample.int(n, m, TRUE), sample.int(n, m, TRUE))
  x <- matrix(rnorm(n * 16), n, 16)
  g <- graph_data(x = x, edge_index = ei, num_nodes = n)
  el <- data.frame(from = paste0("v", ei[1, ]), to = paste0("v", ei[2, ]))
  ig <- to_igraph(g)
  add <- function(what, s) rows[[length(rows) + 1L]] <<- data.frame(
    benchmark = what, nodes = n, edges = m, median_seconds = s)
  add("graph_data", time_it(function() graph_data(x = x, edge_index = ei, num_nodes = n)))
  add("undirected_graph_data", time_it(function() graph_data(edge_index = ei, num_nodes = n, directed = FALSE)))
  add("from_edge_list", time_it(function() graph_from_edgelist(el)))
  add("from_igraph", time_it(function() from_igraph(ig)))
  add("graph_tensors", time_it(function() graph_tensors(g)))
}
small <- dataset_synthetic_graphs(2000, seed = 1)
rows[[length(rows) + 1L]] <- data.frame(
  benchmark = "graph_batch_2000_graphs", nodes = sum(vapply(small, num_nodes, 1L)),
  edges = sum(vapply(small, num_edges, 1L)),
  median_seconds = time_it(function() graph_batch(small)))
timing <- do.call(rbind, rows)

self_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
mem <- list()
for (arch in c("sage", "gcn", "gat", "gin")) for (n in c(10000, 50000)) {
  res <- system2(file.path(R.home("bin"), "Rscript"), shQuote(self_path),
                 stdout = TRUE, env = paste0("GLR_MEM_CASE=", arch, ":", n))
  v <- as.numeric(strsplit(sub("^RESULT ", "", grep("^RESULT", res, value = TRUE)), " ")[[1]])
  mem[[length(mem) + 1L]] <- data.frame(layer = arch, nodes = n, edges = 10 * n,
                                        rss_after_load_mb = v[1], peak_rss_mb = v[2],
                                        peak_gpu_mb = v[3])
}
memory <- do.call(rbind, mem)

print(timing)
print(memory)
utils::write.csv(timing, file.path(out_dir, "construction.csv"), row.names = FALSE)
utils::write.csv(memory, file.path(out_dir, "memory.csv"), row.names = FALSE)
