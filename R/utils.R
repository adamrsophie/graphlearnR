#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @import torch
#' @importFrom stats setNames predict runif rnorm sd quantile
#' @importFrom utils head tail packageVersion
## usethis namespace: end
NULL

`%||%` <- function(x, y) if (is.null(x)) y else x

stopf <- function(fmt, ...) {
  stop(sprintf(fmt, ...), call. = FALSE)
}

warnf <- function(fmt, ...) {
  warning(sprintf(fmt, ...), call. = FALSE)
}

is_scalar_integerish <- function(x) {
  is.numeric(x) && length(x) == 1L && !is.na(x) && x == round(x)
}

is_integerish <- function(x) {
  is.numeric(x) && all(is.na(x) | x == round(x))
}

# Whole number node indices: numeric, finite, not missing and without a
# fractional part, so integer conversion never truncates silently.
check_index <- function(x, name) {
  if (!is.numeric(x) || anyNA(x) || any(!is.finite(x)) || any(x != round(x))) {
    stopf("`%s` must hold whole numbers without missing values.", name)
  }
  invisible(x)
}

check_count <- function(x, name, min = 0L) {
  if (!is_scalar_integerish(x) || x < min) {
    stopf("`%s` must be a single whole number >= %d.", name, min)
  }
  as.integer(x)
}

check_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stopf("`%s` must be TRUE or FALSE.", name)
  }
  x
}

# Turn a numeric vector, matrix or data frame of node or edge features into a
# double matrix with one row per item. NULL stays NULL.
as_feature_matrix <- function(x, name) {
  if (is.null(x)) {
    return(NULL)
  }
  if (inherits(x, "torch_tensor")) {
    x <- as.array(x$cpu())
  }
  if (is.data.frame(x)) {
    ok <- vapply(x, is.numeric, logical(1))
    if (!all(ok)) {
      stopf("`%s` has non-numeric columns: %s.", name,
            paste(names(x)[!ok], collapse = ", "))
    }
    x <- as.matrix(x)
  }
  if (is.vector(x) && is.numeric(x)) {
    x <- matrix(x, ncol = 1L)
  }
  if (!is.matrix(x) || !is.numeric(x)) {
    stopf("`%s` must be a numeric matrix, vector or data frame.", name)
  }
  storage.mode(x) <- "double"
  x
}

n_rows <- function(x) {
  if (is.null(x)) {
    return(NA_integer_)
  }
  if (is.matrix(x) || is.data.frame(x)) nrow(x) else length(x)
}

# Subset rows of a feature matrix or elements of a vector, keeping NULL.
take_rows <- function(x, i) {
  if (is.null(x)) {
    return(NULL)
  }
  if (is.matrix(x)) x[i, , drop = FALSE] else x[i]
}

# Stack per-graph values (vectors, factors or matrices) in order.
bind_rows_values <- function(values) {
  values <- Filter(Negate(is.null), values)
  if (length(values) == 0L) {
    return(NULL)
  }
  if (is.matrix(values[[1L]])) {
    return(do.call(rbind, values))
  }
  if (is.factor(values[[1L]])) {
    # Union of levels, so a class present only in a later graph is kept.
    lv <- unique(unlist(lapply(values, function(v) if (is.factor(v)) levels(v) else unique(as.character(v)))))
    return(factor(unlist(lapply(values, as.character)), levels = lv))
  }
  unlist(values, use.names = FALSE)
}

# Run code with a fixed R RNG seed without changing the caller's random
# state. NULL seed means use the caller's stream as is.
with_optional_seed <- function(seed, code) {
  if (is.null(seed)) {
    return(force(code))
  }
  withr::with_seed(seed, code)
}

# Independent integer seeds for `n` sub tasks, drawn from `seed`. Valid for
# any integer seed, unlike arithmetic such as seed * 1000 + i, which can
# overflow or give two tasks the same seed.
sub_seeds <- function(seed, n) {
  if (is.null(seed)) return(NULL)
  withr::with_seed(seed, sample.int(.Machine$integer.max, n))
}

resolve_device <- function(device = NULL) {
  if (is.null(device)) {
    return(torch_device("cpu"))
  }
  if (inherits(device, "torch_device")) {
    return(device)
  }
  device <- match.arg(device, c("cpu", "cuda", "mps", "auto"))
  if (device == "auto") {
    if (cuda_is_available()) {
      return(torch_device("cuda"))
    }
    if (backends_mps_is_available()) {
      return(torch_device("mps"))
    }
    return(torch_device("cpu"))
  }
  if (device == "cuda" && !cuda_is_available()) {
    stopf("CUDA was requested but is not available.")
  }
  if (device == "mps" && !backends_mps_is_available()) {
    stopf("MPS was requested but is not available.")
  }
  torch_device(device)
}

as_r_array <- function(t) {
  if (inherits(t, "torch_tensor")) as.array(t$detach()$cpu()) else t
}
