#' Train a graph neural network
#'
#' A training loop for the common graph learning tasks. It uses the model,
#' optimizer and loss as ordinary `torch` objects, so all of them stay
#' accessible in the returned fit.
#'
#' | `task` | `data` | targets | loss on |
#' |---|---|---|---|
#' | `"node_classification"`, `"node_regression"` | `graph_data` (or `hetero_graph_data` with `target_type`) with `train_mask` | node `y` | training nodes only |
#' | `"graph_classification"`, `"graph_regression"` | list of `graph_data` or a `graph_dataset`; `val_data` likewise | `graph_y` | every training graph |
#' | `"link_prediction"` | a `link_split` from [split_edges()] (or [split_edges_hetero()]) | positive and sampled negative pairs | training pairs |
#' | `"edge_classification"`, `"edge_regression"` | `graph_data` with `edge_y` and masks from [split_edge_labels()] | edge `y` | training edges |
#'
#' **Node mini-batches.** With `fanouts`, node tasks are trained on sampled
#' neighborhoods ([node_loader()]); the loss uses only the seed nodes of each
#' batch, never the sampled context nodes. Validation uses the full graph.
#'
#' **Link prediction.** Message passing uses `split$train_graph`, which holds
#' training edges only. Fresh negative pairs are drawn every epoch, excluding
#' training positives; validation and test pairs are never used for
#' training.
#'
#' **Reproducibility.** With `seed`, `torch`'s global generator is seeded
#' with `torch_manual_seed(seed)` (needed for weight initialization done
#' before the call, and for dropout), and every R side random draw (batching,
#' neighbor and negative sampling) uses a seed derived from `seed` inside
#' [withr::with_seed()], so the caller's R random state is not changed. The
#' fit records seeds, settings, package versions and the device in
#' `fit$reproducibility`. Training on GPUs may still be nondeterministic,
#' because scatter additions on CUDA use atomic operations whose order is not
#' fixed.
#'
#' @param model An `nn_module` whose forward takes `(x, edge_index, ...)`
#'   with the named arguments documented in [gnn_model()].
#' @param data Training data; see the table.
#' @param task Prediction task.
#' @param epochs Number of epochs.
#' @param optimizer `"adam"`, `"adamw"`, `"sgd"`, `"rmsprop"`, `"adagrad"`, or
#'   a function taking a parameter list and returning a `torch` optimizer.
#' @param lr Learning rate.
#' @param weight_decay L2 penalty passed to the optimizer.
#' @param loss `NULL` for the task default (cross entropy, mean squared error,
#'   binary cross entropy with logits), a name (`"cross_entropy"`, `"mse"`,
#'   `"l1"`, `"huber"`, `"bce"`, `"focal"`), or a function `(prediction,
#'   target)`.
#' @param val_data Validation graphs for graph level tasks.
#' @param batch_size Graphs per batch (graph tasks) or seeds per batch (node
#'   tasks with `fanouts`).
#' @param fanouts Neighbor sampling fanouts for node mini-batches.
#' @param decoder Decoder module for link and edge tasks; defaults to
#'   [link_decoder()] `"dot"` and an [edge_decoder()].
#' @param scheduler `NULL`, a list such as `list(type = "step", step_size =
#'   50, gamma = 0.5)`, `list(type = "plateau", patience = 10, factor = 0.5)`
#'   or `list(type = "cosine")`, or a function taking the optimizer and
#'   returning a `torch` scheduler.
#' @param early_stopping Patience in epochs, or `NULL` to train all epochs.
#'   Monitors validation loss; the best weights are restored at the end.
#' @param checkpoint_path Optional file; the model state is saved there with
#'   [torch::torch_save()] whenever validation loss improves.
#' @param device Device, see [graph_tensors()].
#' @param seed Optional integer seed.
#' @param grad_clip Optional maximum gradient norm.
#' @param class_weights Optional class weights for cross entropy.
#' @param target_type Node type to predict for heterogeneous graphs.
#' @param verbose Print progress every `verbose` epochs (`FALSE` for none).
#' @return An object of class `graph_fit`.
#' @examples
#' if (torch::torch_is_installed()) {
#'   g <- split_nodes(dataset_karate(), seed = 1)
#'   model <- gnn_model("sage", input_dim = num_node_features(g),
#'                      hidden_dim = 16, output_dim = 2)
#'   fit <- train_graph_model(model, g, "node_classification", epochs = 30,
#'                            seed = 1)
#'   fit
#'   head(predict(fit, g))
#' }
#' @export
train_graph_model <- function(model, data,
                              task = c("node_classification", "node_regression",
                                       "graph_classification", "graph_regression",
                                       "link_prediction", "edge_classification",
                                       "edge_regression"),
                              epochs = 100L, optimizer = "adam", lr = 0.01,
                              weight_decay = 0, loss = NULL, val_data = NULL,
                              batch_size = NULL, fanouts = NULL, decoder = NULL,
                              scheduler = NULL, early_stopping = NULL,
                              checkpoint_path = NULL, device = NULL, seed = NULL,
                              grad_clip = NULL, class_weights = NULL,
                              target_type = NULL, verbose = FALSE) {
  task <- match.arg(task)
  if (!inherits(model, "nn_module")) stopf("`model` must be a torch nn_module.")
  epochs <- check_count(epochs, "epochs", 1L)
  dev <- resolve_device(device)
  if (!is.null(seed)) torch_manual_seed(seed)
  model$to(device = dev)

  runner <- switch(task,
    node_classification = , node_regression =
      node_task_runner(model, data, task, dev, fanouts, batch_size, seed, target_type),
    graph_classification = , graph_regression =
      graph_task_runner(model, data, val_data, task, dev, batch_size %||% 32L, seed),
    link_prediction = link_task_runner(model, data, dev, decoder, seed),
    edge_classification = , edge_regression =
      edge_task_runner(model, data, task, dev, decoder)
  )
  if (!is.null(runner$decoder)) runner$decoder$to(device = dev)
  loss_fn <- resolve_loss(loss, runner$default_loss, class_weights, dev)
  params <- c(model$parameters, if (!is.null(runner$decoder)) runner$decoder$parameters)
  opt <- make_optimizer(optimizer, params, lr, weight_decay)
  sched <- make_scheduler(scheduler, opt, epochs)

  history <- data.frame(epoch = integer(), train_loss = numeric(), val_loss = numeric(),
                        val_metric = numeric(), lr = numeric(), seconds = numeric())
  best <- list(loss = Inf, epoch = NA_integer_, state = NULL, dec_state = NULL)
  wait <- 0L
  for (epoch in seq_len(epochs)) {
    t0 <- proc.time()[["elapsed"]]
    model$train()
    if (!is.null(runner$decoder)) runner$decoder$train()
    train_loss <- runner$train_epoch(epoch, opt, loss_fn, grad_clip)
    model$eval()
    if (!is.null(runner$decoder)) runner$decoder$eval()
    val <- with_no_grad(runner$validate(loss_fn))
    lr_now <- opt$param_groups[[1L]]$lr
    if (!is.null(sched)) {
      if (inherits(sched, "lr_reduce_on_plateau")) sched$step(val$loss %||% train_loss)
      else sched$step()
    }
    history[nrow(history) + 1L, ] <- list(epoch, train_loss, val$loss %||% NA_real_,
                                          val$metric %||% NA_real_, lr_now,
                                          proc.time()[["elapsed"]] - t0)
    monitor <- val$loss %||% train_loss
    if (is.finite(monitor) && monitor < best$loss - 1e-8) {
      best$loss <- monitor
      best$epoch <- epoch
      if (!is.null(early_stopping) || !is.null(checkpoint_path)) {
        best$state <- clone_state(model)
        if (!is.null(runner$decoder)) best$dec_state <- clone_state(runner$decoder)
        if (!is.null(checkpoint_path)) torch_save(model$state_dict(), checkpoint_path)
      }
      wait <- 0L
    } else {
      wait <- wait + 1L
    }
    if (!isFALSE(verbose) && (epoch %% as.integer(verbose) == 0L || epoch == 1L)) {
      message(sprintf("epoch %d  train_loss %.4f  val_loss %s  val_%s %s", epoch,
                      train_loss, format(val$loss %||% NA, digits = 4),
                      runner$metric_name, format(val$metric %||% NA, digits = 4)))
    }
    if (!is.null(early_stopping) && wait >= early_stopping) break
  }
  if (!is.null(early_stopping) && !is.null(best$state)) {
    model$load_state_dict(best$state)
    if (!is.null(best$dec_state)) runner$decoder$load_state_dict(best$dec_state)
  }
  model$eval()
  if (!is.null(runner$decoder)) runner$decoder$eval()

  structure(list(
    model = model, decoder = runner$decoder, optimizer = opt, scheduler = sched,
    task = task, history = history, best_epoch = best$epoch,
    levels = runner$levels, data = data, target_type = target_type,
    metric_name = runner$metric_name, device = dev,
    reproducibility = reproducibility_info(
      seed = seed, task = task, epochs = epochs, optimizer = optimizer, lr = lr,
      weight_decay = weight_decay, loss = loss, batch_size = batch_size,
      fanouts = fanouts, early_stopping = early_stopping, scheduler = scheduler,
      device = dev, model = model, data = data
    )
  ), class = "graph_fit")
}

clone_state <- function(module) {
  lapply(module$state_dict(), function(t) t$detach()$clone())
}

make_optimizer <- function(optimizer, params, lr, weight_decay) {
  if (is.function(optimizer)) {
    return(optimizer(params))
  }
  switch(match.arg(optimizer, c("adam", "adamw", "sgd", "rmsprop", "adagrad")),
    adam = optim_adam(params, lr = lr, weight_decay = weight_decay),
    adamw = optim_adamw(params, lr = lr, weight_decay = weight_decay),
    sgd = optim_sgd(params, lr = lr, momentum = 0.9, weight_decay = weight_decay),
    rmsprop = optim_rmsprop(params, lr = lr, weight_decay = weight_decay),
    adagrad = optim_adagrad(params, lr = lr, weight_decay = weight_decay)
  )
}

make_scheduler <- function(scheduler, opt, epochs) {
  if (is.null(scheduler)) {
    return(NULL)
  }
  if (is.function(scheduler)) {
    return(scheduler(opt))
  }
  type <- match.arg(scheduler$type, c("step", "plateau", "cosine"))
  switch(type,
    step = lr_step(opt, step_size = scheduler$step_size %||% 50L,
                   gamma = scheduler$gamma %||% 0.5),
    plateau = lr_reduce_on_plateau(opt, factor = scheduler$factor %||% 0.5,
                                   patience = scheduler$patience %||% 10L),
    cosine = lr_cosine_annealing(opt, T_max = scheduler$T_max %||% epochs)
  )
}

resolve_loss <- function(loss, default, class_weights, dev) {
  if (is.function(loss)) {
    return(loss)
  }
  name <- loss %||% default
  w <- if (!is.null(class_weights)) torch_tensor(as.double(class_weights), device = dev)
  as_target <- function(pred, target) {
    if (pred$dim() == 2L && pred$size(2L) == 1L && target$dim() == 1L) pred$squeeze(2L) else pred
  }
  switch(match.arg(name, c("cross_entropy", "mse", "l1", "huber", "bce", "focal")),
    cross_entropy = function(pred, target) nnf_cross_entropy(pred, target, weight = w),
    mse = function(pred, target) nnf_mse_loss(as_target(pred, target), target),
    l1 = function(pred, target) nnf_l1_loss(as_target(pred, target), target),
    huber = function(pred, target) nnf_smooth_l1_loss(as_target(pred, target), target),
    bce = function(pred, target) {
      nnf_binary_cross_entropy_with_logits(as_target(pred, target), target$to(dtype = pred$dtype))
    },
    focal = function(pred, target, gamma = 2) {
      logp <- nnf_log_softmax(pred, dim = 2L)$gather(2L, target$unsqueeze(2L))$squeeze(2L)
      p <- torch_exp(logp)
      (-(1 - p)$pow(gamma) * logp)$mean()
    }
  )
}

step_optimizer <- function(opt, loss, grad_clip, params) {
  opt$zero_grad()
  loss$backward()
  if (!is.null(grad_clip)) nn_utils_clip_grad_norm_(params, grad_clip)
  opt$step()
  as.numeric(loss$item())
}

all_params <- function(opt) {
  do.call(c, lapply(opt$param_groups, function(g) g$params))
}

class_levels <- function(y) {
  if (is.factor(y)) levels(y) else if (is.integer(y)) as.character(sort(unique(y))) else NULL
}

# ---- node tasks -------------------------------------------------------------

node_task_runner <- function(model, data, task, dev, fanouts, batch_size, seed, target_type) {
  classif <- task == "node_classification"
  if (inherits(data, "hetero_graph_data")) {
    return(hetero_node_runner(model, data, classif, dev, target_type))
  }
  if (is.null(data$y)) stopf("Node tasks need node labels `y`.")
  if (is.null(data$train_mask)) {
    stopf("Node tasks need `train_mask`; use split_nodes() first.")
  }
  if (classif && !is.factor(data$y) && !is.integer(data$y)) {
    stopf("Node classification needs factor or integer labels.")
  }
  if (!classif && !is.numeric(data$y)) stopf("Node regression needs numeric labels.")
  tt <- graph_tensors(data, dev)
  train_idx <- tensor_which(tt$train_mask)
  val_idx <- if (!is.null(tt$val_mask) && any(data$val_mask)) tensor_which(tt$val_mask)
  metric_fn <- node_metric(classif)
  loader <- if (!is.null(fanouts)) {
    node_loader(data, which(data$train_mask), batch_size %||% 64L, fanouts, TRUE, seed)
  }
  list(
    default_loss = if (classif) "cross_entropy" else "mse",
    metric_name = if (classif) "accuracy" else "rmse",
    levels = if (classif) class_levels(data$y),
    train_epoch = function(epoch, opt, loss_fn, grad_clip) {
      if (is.null(loader)) {
        out <- model_forward(model, tt)
        loss <- loss_fn(out$index_select(1L, train_idx), tt$y$index_select(1L, train_idx))
        return(step_optimizer(opt, loss, grad_clip, all_params(opt)))
      }
      losses <- vapply(loader_batches(loader, epoch), function(sg) {
        bt <- graph_tensors(sg, dev)
        seeds <- torch_arange(1L, sg$num_seeds, dtype = torch_long(), device = dev)
        out <- model_forward(model, bt)
        loss <- loss_fn(out$index_select(1L, seeds), bt$y$index_select(1L, seeds))
        step_optimizer(opt, loss, grad_clip, all_params(opt)) * sg$num_seeds
      }, numeric(1))
      sum(losses) / length(loader$nodes)
    },
    validate = function(loss_fn) {
      if (is.null(val_idx)) return(list())
      out <- model_forward(model, tt)$index_select(1L, val_idx)
      target <- tt$y$index_select(1L, val_idx)
      list(loss = as.numeric(loss_fn(out, target)$item()), metric = metric_fn(out, target))
    }
  )
}

tensor_which <- function(mask) {
  torch_nonzero(mask)$squeeze(2L)$to(dtype = torch_long())
}

node_metric <- function(classif) {
  if (classif) {
    function(out, target) mean(as_r_array(out$argmax(dim = 2L)) == as_r_array(target))
  } else {
    function(out, target) {
      o <- as_r_array(out)
      sqrt(mean((as.vector(o) - as.vector(as_r_array(target)))^2))
    }
  }
}

# ---- graph tasks ------------------------------------------------------------

as_graph_list <- function(data, what = "data") {
  if (inherits(data, "graph_data")) stopf("`%s` must be a list of graphs or a graph_dataset.", what)
  data
}

graph_task_runner <- function(model, data, val_data, task, dev, batch_size, seed) {
  classif <- task == "graph_classification"
  data <- as_graph_list(data)
  if (is.null(data[[1L]]$graph_y)) stopf("Graph tasks need `graph_y` on every graph.")
  loader <- graph_loader(data, batch_size, TRUE, seed)
  val_batches <- if (!is.null(val_data)) {
    loader_batches(graph_loader(as_graph_list(val_data, "val_data"), batch_size, FALSE))
  }
  val_tt <- lapply(val_batches, graph_tensors, device = dev)
  metric_fn <- node_metric(classif)
  levels <- if (classif) class_levels(stack_graph_labels(lapply(seq_len(length(data)), function(i) data[[i]]$graph_y)))
  list(
    default_loss = if (classif) "cross_entropy" else "mse",
    metric_name = if (classif) "accuracy" else "rmse",
    levels = levels,
    train_epoch = function(epoch, opt, loss_fn, grad_clip) {
      tot <- 0
      n <- 0
      for (b in loader_batches(loader, epoch)) {
        bt <- graph_tensors(b, dev)
        out <- model_forward(model, bt)
        tot <- tot + step_optimizer(opt, loss_fn(out, bt$graph_y), grad_clip, all_params(opt)) * b$num_graphs
        n <- n + b$num_graphs
      }
      tot / n
    },
    validate = function(loss_fn) {
      if (!length(val_tt)) return(list())
      outs <- lapply(val_tt, function(bt) model_forward(model, bt))
      out <- torch_cat(outs, dim = 1L)
      target <- torch_cat(lapply(val_tt, `[[`, "graph_y"), dim = 1L)
      list(loss = as.numeric(loss_fn(out, target)$item()), metric = metric_fn(out, target))
    }
  )
}

# ---- link prediction --------------------------------------------------------

link_task_runner <- function(model, split, dev, decoder, seed) {
  if (inherits(split, "hetero_link_split")) {
    return(hetero_link_runner(model, split, dev, decoder, seed))
  }
  if (!inherits(split, "link_split")) stopf("Link prediction needs a link_split from split_edges().")
  decoder <- decoder %||% link_decoder("dot")
  tt <- graph_tensors(split$train_graph, dev)
  pairs_t <- function(m) torch_tensor(m, dtype = torch_long(), device = dev)
  train_pos <- pairs_t(split$train_pos)
  val_pos <- if (ncol(split$val_pos)) pairs_t(split$val_pos)
  val_neg <- if (ncol(split$val_neg)) pairs_t(split$val_neg)
  list(
    default_loss = "bce", metric_name = "roc_auc", decoder = decoder,
    train_epoch = function(epoch, opt, loss_fn, grad_clip) {
      neg <- negative_sampling(split$train_pos, num_nodes = split$num_nodes,
                               num_samples = ncol(split$train_pos),
                               directed = split$directed,
                               seed = if (is.null(seed)) NULL else seed + epoch)
      z <- model_forward(model, tt)
      s_pos <- decoder(z, train_pos)
      s_neg <- decoder(z, pairs_t(neg))
      scores <- torch_cat(list(s_pos, s_neg))
      target <- torch_cat(list(torch_ones_like(s_pos), torch_zeros_like(s_neg)))
      step_optimizer(opt, loss_fn(scores, target), grad_clip, all_params(opt))
    },
    validate = function(loss_fn) {
      if (is.null(val_pos) || is.null(val_neg)) return(list())
      z <- model_forward(model, tt)
      s_pos <- decoder(z, val_pos)
      s_neg <- decoder(z, val_neg)
      scores <- torch_cat(list(s_pos, s_neg))
      target <- torch_cat(list(torch_ones_like(s_pos), torch_zeros_like(s_neg)))
      list(loss = as.numeric(loss_fn(scores, target)$item()),
           metric = metric_roc_auc(as_r_array(target) > 0.5, as_r_array(scores)))
    }
  )
}

# ---- edge tasks -------------------------------------------------------------

edge_task_runner <- function(model, data, task, dev, decoder) {
  classif <- task == "edge_classification"
  if (is.null(data$edge_y)) stopf("Edge tasks need `edge_y`.")
  if (is.null(data$edge_train_mask)) stopf("Edge tasks need edge masks; use split_edge_labels().")
  tt <- graph_tensors(data, dev)
  levels <- if (classif) class_levels(data$edge_y)
  if (is.null(decoder)) {
    dim <- with_no_grad(model_forward(model, tt))$size(2L)
    out_dim <- if (classif) length(levels) else if (is.matrix(data$edge_y)) ncol(data$edge_y) else 1L
    decoder <- edge_decoder(dim, out_dim, edge_dim = num_edge_features(data))
  }
  sel <- function(mask) {
    idx <- which(mask)
    list(pairs = torch_tensor(data$edge_index[, idx, drop = FALSE], dtype = torch_long(), device = dev),
         idx = torch_tensor(idx, dtype = torch_long(), device = dev))
  }
  tr <- sel(data$edge_train_mask)
  va <- if (any(data$edge_val_mask %||% FALSE)) sel(data$edge_val_mask)
  score <- function(z, s) {
    ea <- if (!is.null(tt$edge_attr)) tt$edge_attr$index_select(1L, s$idx)
    decoder(z, s$pairs, ea)
  }
  metric_fn <- node_metric(classif)
  list(
    default_loss = if (classif) "cross_entropy" else "mse",
    metric_name = if (classif) "accuracy" else "rmse",
    levels = levels, decoder = decoder,
    train_epoch = function(epoch, opt, loss_fn, grad_clip) {
      z <- model_forward(model, tt)
      loss <- loss_fn(score(z, tr), tt$edge_y$index_select(1L, tr$idx))
      step_optimizer(opt, loss, grad_clip, all_params(opt))
    },
    validate = function(loss_fn) {
      if (is.null(va)) return(list())
      out <- score(model_forward(model, tt), va)
      target <- tt$edge_y$index_select(1L, va$idx)
      list(loss = as.numeric(loss_fn(out, target)$item()), metric = metric_fn(out, target))
    }
  )
}

reproducibility_info <- function(seed, task, epochs, optimizer, lr, weight_decay,
                                 loss, batch_size, fanouts, early_stopping,
                                 scheduler, device, model, data) {
  data_meta <- if (inherits(data, "graph_data")) data$metadata
               else if (inherits(data, "link_split")) data$train_graph$metadata
               else if (inherits(data, "graph_dataset")) data$metadata
  splits <- if (inherits(data, "graph_data")) {
    c(train = sum(data$train_mask %||% FALSE), val = sum(data$val_mask %||% FALSE),
      test = sum(data$test_mask %||% FALSE))
  } else if (inherits(data, "link_split")) {
    c(train = ncol(data$train_pos), val = ncol(data$val_pos), test = ncol(data$test_pos))
  }
  list(
    seed = seed, task = task,
    hyperparameters = list(epochs = epochs, lr = lr, weight_decay = weight_decay,
                           batch_size = batch_size, fanouts = fanouts,
                           early_stopping = early_stopping),
    optimizer = if (is.character(optimizer)) optimizer else "custom",
    scheduler = if (is.list(scheduler)) scheduler else if (!is.null(scheduler)) "custom",
    loss = if (is.character(loss)) loss else if (is.null(loss)) "default" else "custom",
    model_config = tryCatch(model$config, error = function(e) NULL),
    model_class = class(model)[1L],
    num_parameters = sum(vapply(model$parameters, function(p) p$numel(), numeric(1))),
    dataset = data_meta, splits = splits,
    device = device$type, precision = "float32",
    versions = c(R = paste(R.version$major, R.version$minor, sep = "."),
                 graphlearnR = as.character(packageVersion("graphlearnR")),
                 torch = as.character(packageVersion("torch"))),
    platform = R.version$platform,
    time = format(Sys.time(), tz = "UTC", usetz = TRUE)
  )
}

#' @export
print.graph_fit <- function(x, ...) {
  h <- x$history
  cat(sprintf("<graph_fit> %s, %d epochs on %s\n", x$task, nrow(h), x$device$type))
  if (nrow(h)) {
    last <- h[nrow(h), ]
    cat(sprintf("  final train loss %.4f", last$train_loss))
    if (!is.na(last$val_loss)) cat(sprintf(", val loss %.4f, val %s %.4f", last$val_loss,
                                           x$metric_name, last$val_metric))
    cat("\n")
    if (!is.na(x$best_epoch)) cat(sprintf("  best epoch %d\n", x$best_epoch))
  }
  invisible(x)
}

#' Save and load a trained model
#'
#' `save_graph_fit()` writes the model and decoder with `torch`'s own
#' serialization ([torch::torch_serialize()]) together with the task,
#' class levels, training history and reproducibility record into one
#' `.rds` file. Training data and optimizer state are not saved.
#' `load_graph_fit()` restores a fit that [predict()] can use.
#'
#' @param fit A `graph_fit`.
#' @param path File path.
#' @param device Device to load onto.
#' @return `save_graph_fit()` returns `path` invisibly; `load_graph_fit()`
#'   returns a `graph_fit` (without `data` and `optimizer`).
#' @export
save_graph_fit <- function(fit, path) {
  obj <- list(
    model = torch_serialize(fit$model),
    decoder = if (!is.null(fit$decoder)) torch_serialize(fit$decoder),
    task = fit$task, history = fit$history, best_epoch = fit$best_epoch,
    levels = fit$levels, target_type = fit$target_type,
    metric_name = fit$metric_name, reproducibility = fit$reproducibility,
    format = "graphlearnR_fit_v1"
  )
  saveRDS(obj, path)
  invisible(path)
}

#' @rdname save_graph_fit
#' @export
load_graph_fit <- function(path, device = NULL) {
  obj <- readRDS(path)
  if (!identical(obj$format, "graphlearnR_fit_v1")) stopf("`%s` is not a saved graph_fit.", path)
  dev <- resolve_device(device)
  model <- torch_load(obj$model, device = dev)
  model$eval()
  decoder <- if (!is.null(obj$decoder)) torch_load(obj$decoder, device = dev)
  if (!is.null(decoder)) decoder$eval()
  structure(list(model = model, decoder = decoder, optimizer = NULL, task = obj$task,
                 history = obj$history, best_epoch = obj$best_epoch, levels = obj$levels,
                 data = NULL, target_type = obj$target_type, metric_name = obj$metric_name,
                 device = dev, reproducibility = obj$reproducibility),
            class = "graph_fit")
}
