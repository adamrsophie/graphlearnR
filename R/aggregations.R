#' Scatter aggregation over groups
#'
#' Reduce rows of `src` into `dim_size` groups given by `index`: row `k` of
#' the result combines all rows `j` with `index[j] == k`. These are the
#' aggregation kernels used by [message_passing()]; they are vectorized
#' `torch` operations (`index_add`, `scatter_reduce`) and differentiable.
#'
#' * `scatter_sum()`: sum of rows. Empty groups give 0.
#' * `scatter_mean()`: sum divided by the group size. Empty groups give 0.
#' * `scatter_max()`, `scatter_min()`: elementwise maximum or minimum.
#'   Empty groups give 0. Gradients flow to the selected entries; ties
#'   split the gradient as `torch` defines for `scatter_reduce`.
#' * `scatter_std()`: population standard deviation, computed as
#'   \eqn{\sqrt{\max(\mathrm{var}, \epsilon)}} and set to 0 where the
#'   variance is at most \eqn{\epsilon} (empty or constant groups), as in
#'   PyTorch Geometric's `StdAggregation`; the gradient stays finite.
#' * `scatter_softmax()`: softmax of `src` within each group (used for
#'   attention over incoming edges); columns are treated separately.
#' * `scatter_aggregate()`: dispatch by name; several names give their
#'   results concatenated column wise.
#'
#' @param src Tensor with one row per element (for example per edge).
#' @param index One based long tensor (or integer vector) of group ids, one
#'   per row of `src`.
#' @param dim_size Number of groups (for example the number of nodes).
#' @return A tensor with `dim_size` rows.
#' @examples
#' if (torch::torch_is_installed()) {
#'   src <- torch::torch_tensor(matrix(c(1, 2, 3, 4), 4, 1))
#'   idx <- torch::torch_tensor(c(1L, 1L, 2L, 2L), dtype = torch::torch_long())
#'   scatter_sum(src, idx, 3)
#'   scatter_mean(src, idx, 3)
#'   scatter_max(src, idx, 3)
#' }
#' @name scatter
NULL

as_index_tensor <- function(index, device = NULL) {
  if (inherits(index, "torch_tensor")) {
    return(index)
  }
  torch_tensor(as.integer(index), dtype = torch_long(), device = device)
}

with_2d <- function(src, fun) {
  vec <- src$dim() == 1L
  if (vec) src <- src$unsqueeze(2L)
  out <- fun(src)
  if (vec) out$squeeze(2L) else out
}

#' @rdname scatter
#' @export
scatter_sum <- function(src, index, dim_size) {
  index <- as_index_tensor(index, src$device)
  with_2d(src, function(s) {
    out <- torch_zeros(dim_size, s$size(2L), dtype = s$dtype, device = s$device)
    if (index$numel() == 0L) out else out$index_add(1L, index, s)
  })
}

#' @rdname scatter
#' @export
scatter_count <- function(index, dim_size, dtype = torch_float(), device = NULL) {
  index <- as_index_tensor(index, device)
  ones <- torch_ones(index$numel(), dtype = dtype, device = index$device)
  out <- torch_zeros(dim_size, dtype = dtype, device = index$device)
  if (index$numel() == 0L) out else out$index_add(1L, index, ones)
}

#' @rdname scatter
#' @export
scatter_mean <- function(src, index, dim_size) {
  index <- as_index_tensor(index, src$device)
  total <- scatter_sum(src, index, dim_size)
  cnt <- scatter_count(index, dim_size, dtype = src$dtype)$clamp(min = 1)
  if (total$dim() == 1L) total / cnt else total / cnt$unsqueeze(2L)
}

scatter_extreme <- function(src, index, dim_size, reduce) {
  index <- as_index_tensor(index, src$device)
  with_2d(src, function(s) {
    out <- torch_zeros(dim_size, s$size(2L), dtype = s$dtype, device = s$device)
    if (index$numel() == 0L) {
      return(out)
    }
    idx <- index$unsqueeze(2L)$expand(c(-1L, s$size(2L)))
    out$scatter_reduce(1L, idx, s, reduce = reduce, include_self = FALSE)
  })
}

#' @rdname scatter
#' @export
scatter_max <- function(src, index, dim_size) scatter_extreme(src, index, dim_size, "amax")

#' @rdname scatter
#' @export
scatter_min <- function(src, index, dim_size) scatter_extreme(src, index, dim_size, "amin")

#' @rdname scatter
#' @export
scatter_std <- function(src, index, dim_size, eps = 1e-5) {
  index <- as_index_tensor(index, src$device)
  mean <- scatter_mean(src, index, dim_size)
  sq <- scatter_mean(src * src, index, dim_size)
  # As PyTorch Geometric's StdAggregation: clamp the variance at eps, then
  # report 0 where the clamp was active (empty groups, constant groups).
  out <- (sq - mean * mean)$clamp(min = eps)$sqrt()
  out$masked_fill(out <= sqrt(eps), 0)
}

#' @rdname scatter
#' @export
scatter_softmax <- function(src, index, dim_size) {
  index <- as_index_tensor(index, src$device)
  with_2d(src, function(s) {
    mx <- scatter_max(s$detach(), index, dim_size)
    e <- torch_exp(s - mx$index_select(1L, index))
    tot <- scatter_sum(e, index, dim_size)
    e / (tot$index_select(1L, index) + 1e-16)
  })
}

#' @rdname scatter
#' @param aggr One or more of `"sum"` (alias `"add"`), `"mean"`, `"max"`,
#'   `"min"`, `"std"`.
#' @param dtype,device Dtype and device of the count tensor.
#' @param eps Variance floor in `scatter_std()`; smaller variances give 0.
#' @export
scatter_aggregate <- function(src, index, dim_size, aggr = "sum") {
  if (length(aggr) > 1L) {
    return(torch_cat(lapply(aggr, function(a) {
      scatter_aggregate(src, index, dim_size, a)
    }), dim = -1L))
  }
  switch(match_aggr(aggr),
    sum = scatter_sum(src, index, dim_size),
    mean = scatter_mean(src, index, dim_size),
    max = scatter_max(src, index, dim_size),
    min = scatter_min(src, index, dim_size),
    std = scatter_std(src, index, dim_size)
  )
}

match_aggr <- function(aggr) {
  if (identical(aggr, "add")) aggr <- "sum"
  if (!is.character(aggr) || length(aggr) != 1L ||
      !aggr %in% c("sum", "mean", "max", "min", "std")) {
    stopf("Unknown aggregation `%s`; use sum, mean, max, min or std.",
          paste(aggr, collapse = ","))
  }
  aggr
}
