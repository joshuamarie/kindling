#' @param n_classes Positive integer. Number of output classes. Required when
#'   `x` is a `dataset` with scalar (classification) labels; ignored otherwise.
#'
#' @section Dataset method (`train_nn.dataset()`):
#' Trains a neural network directly on a `torch` dataset object. Batching and
#' lazy loading are handled by `torch::dataloader()`, making this method
#' well-suited for large datasets that do not fit entirely in memory.
#'
#' Architecture configuration follows the same contract as other `train_nn()`
#' methods via `architecture = nn_arch(...)` (or legacy `arch = ...`).
#' For non-tabular inputs (time series, images), set `flatten_input = FALSE` to
#' preserve tensor dimensions expected by recurrent or convolutional layers.
#'
#' Labels are taken from the second element of each dataset item (i.e.
#' `dataset[[i]][[2]]`), so `y` is ignored. When the label is a scalar tensor,
#' a classification task is assumed and `n_classes` must be supplied. The loss
#' is automatically switched to `"cross_entropy"` in that case.
#'
#' Fitted values are **not** cached in the returned object. Use
#' [predict.nn_fit_ds()] with `newdata` to obtain predictions after training.
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'     # torch dataset method — labels come from the dataset itself
#'     iris_cls_dataset = torch::dataset(
#'         name = "iris_cls_dataset",
#'         
#'         initialize = function(data = iris) {
#'             self$x = torch::torch_tensor(
#'                 as.matrix(data[, 1:4]),
#'                 dtype = torch::torch_float32()
#'             )
#'             # Species is a factor; convert to integer (1-indexed -> keep as-is for cross_entropy)
#'             self$y = torch::torch_tensor(
#'                 as.integer(data$Species),
#'                 dtype = torch::torch_long()
#'             )
#'         },
#'         
#'         .getitem = function(i) {
#'             list(self$x[i, ], self$y[i])
#'         },
#'         
#'         .length = function() {
#'             self$x$size(1)
#'         }
#'     )()
#'     
#'     model_nn_ds = train_nn(
#'         x = iris_cls_dataset,
#'         hidden_neurons = c(32, 10),
#'         activations = "relu",
#'         epochs = 80,
#'         batch_size = 16,
#'         learn_rate = 0.01,
#'         n_classes = 3, # Iris dataset has only 3 species
#'         validation_split = 0.2,
#'         verbose = TRUE
#'     )
#'     
#'     pred_nn = predict(model_nn_ds, iris_cls_dataset)
#'     class_preds = c("Setosa", "Versicolor", "Virginica")[predict(model_nn_ds, iris_cls_dataset)]
#'     
#'     # Confusion Matrix
#'     table(actual = iris$Species, pred = class_preds)
#' }
#' }
#'
#' @rdname gen-nn-train
#' @export
train_nn.dataset = 
    function(
        x,
        y = NULL,
        hidden_neurons = NULL,
        activations = NULL,
        output_activation = NULL,
        bias = TRUE,
        arch = NULL,
        architecture = NULL,
        flatten_input = NULL,
        epochs = 100,
        batch_size = 32,
        penalty = 0,
        mixture = 0,
        learn_rate = 0.001,
        optimizer = "adam",
        optimizer_args = list(),
        loss = "mse",
        validation_split = 0,
        device = NULL,
        verbose = FALSE,
        cache_weights = FALSE,
        n_classes = NULL,
        ...
    ) 
{
    if (!requireNamespace("torch", quietly = TRUE)) {
        cli::cli_abort("Package {.pkg torch} is required but not installed.")
    }

    if (length(x) < 1L) {
        cli::cli_abort("{.arg x} dataset is empty. Provide a dataset with at least one observation.")
    }

    arch = .resolve_train_architecture(architecture = architecture, arch = arch)
    flatten_input = .resolve_flatten_input(flatten_input = flatten_input, arch = arch)

    if (!flatten_input && is.null(arch)) {
        cli::cli_abort(c(
            "{.arg flatten_input = FALSE} requires a custom {.cls nn_arch}.",
            i = "Use {.arg architecture} to provide a layer stack that supports non-flattened tensors."
        ))
    }

    if (!is.null(y)) {
        cli::cli_warn("{.arg y} is ignored when {.arg x} is a dataset. Labels come from the dataset itself.")
    }

    act_specs = eval_act_funs({{ activations }}, {{ output_activation }})
    activations = act_specs$activations
    output_activation = act_specs$output_activation

    # ---- Infer dimensions from first batch ----
    first_item = .dataset_get_item(x, 1L)
    x_sample = first_item[[1]]
    y_sample = first_item[[2]]

    if (!inherits(x_sample, "torch_tensor") || !inherits(y_sample, "torch_tensor")) {
        cli::cli_abort(c(
            "Dataset items must be lists of two torch tensors: {.code list(x, y)}.",
            i = "Found an incompatible structure at {.code dataset[[1]]}."
        ))
    }

    no_x = as.integer(prod(x_sample$size()))

    is_classification = FALSE
    if (inherits(y_sample, "torch_tensor")) {
        y_dims = as.integer(y_sample$size())
        is_scalar_label = length(y_dims) == 0L || prod(y_dims) == 1L

        if (y_sample$dtype == torch::torch_long() && is_scalar_label) {
            is_classification = TRUE
            if (
                is.null(n_classes) ||
                    !is.numeric(n_classes) ||
                    length(n_classes) != 1L ||
                    is.na(n_classes) ||
                    n_classes < 2L ||
                    (n_classes %% 1L) != 0
            ) {
                cli::cli_abort("{.arg n_classes} must be a single integer >= 2 for classification datasets.")
            }

            no_y = as.integer(n_classes)
        } else {
            no_y = as.integer(if (length(y_dims) == 0L) 1L else prod(y_dims))

            if (!is.null(n_classes)) {
                cli::cli_warn("{.arg n_classes} is ignored when dataset labels are not scalar integer class IDs.")
            }
        }
    } else {
        cli::cli_abort("Dataset labels must be torch tensors.")
    }

    if (is_classification && loss == "mse") {
        loss = "cross_entropy"
        if (verbose) cli::cli_alert("Auto-detected classification task. Using cross_entropy loss.")
    }

    train_nn_impl_dataset(
        dataset = x,
        no_x = no_x,
        no_y = no_y,
        is_classification = is_classification,
        hidden_neurons = hidden_neurons,
        activations = activations,
        output_activation = output_activation,
        bias = bias,
        epochs = epochs,
        batch_size = batch_size,
        penalty = penalty,
        mixture = mixture,
        learn_rate = learn_rate,
        optimizer = optimizer,
        optimizer_args = optimizer_args,
        loss = loss,
        validation_split = validation_split,
        device = device,
        verbose = verbose,
        cache_weights = cache_weights,
        flatten_input = flatten_input,
        arch = arch,
        fit_class = "nn_fit_ds"
    )
}


#' Retrieve and validate one item from a torch dataset
#' @noRd
.dataset_get_item = function(dataset, i) {
    item = tryCatch(
        dataset[i],
        error = function(e) {
            cli::cli_abort(c(
                "Failed to read item {.val {i}} from dataset.",
                i = "Ensure {.arg x} implements {.code .getitem(i)} and {.code .length()}.",
                x = conditionMessage(e)
            ))
        }
    )

    if (!is.list(item) || length(item) < 2L) {
        cli::cli_abort(c(
            "Each dataset item must be a list with at least two elements: {.code list(x, y)}.",
            i = "Found {.cls {class(item)[1]}} at {.code dataset[[{i}]]}."
        ))
    }

    item
}


#' Normalize flatten_input argument for dataset training
#' @noRd
.resolve_flatten_input = function(flatten_input = NULL, arch = NULL) {
    if (is.null(flatten_input)) {
        return(is.null(arch))
    }

    if (!is.logical(flatten_input) || length(flatten_input) != 1L || is.na(flatten_input)) {
        cli::cli_abort("{.arg flatten_input} must be a single TRUE or FALSE value.")
    }

    flatten_input
}


#' train_nn implementation for torch datasets
#' @keywords internal
train_nn_impl_dataset = 
    function(
        dataset,
        no_x,
        no_y,
        is_classification,
        hidden_neurons,
        activations = NULL,
        output_activation = NULL,
        bias = TRUE,
        epochs = 100,
        batch_size = 32,
        penalty = 0,
        mixture = 0,
        learn_rate = 0.001,
        optimizer = "adam",
        optimizer_args = list(),
        loss = "mse",
        validation_split = 0,
        device = NULL,
        verbose = FALSE,
        cache_weights = FALSE,
        flatten_input = TRUE,
        arch = NULL,
        fit_class = "nn_fit_ds"
    ) 
{
    if (missing(hidden_neurons) || is.null(hidden_neurons) || length(hidden_neurons) == 0L) {
        hidden_neurons = integer(0L)
    }

    # ---- Device ----
    if (is.null(device)) {
        device = get_default_device()
    } else {
        device = validate_device(device)
    }

    if (verbose) cli::cli_alert_info("Using device: {device}")

    validate_regularization(penalty, mixture)

    if (!is.numeric(validation_split) || length(validation_split) != 1L || is.na(validation_split) ||
        validation_split < 0 || validation_split >= 1) {
        cli::cli_abort("{.arg validation_split} must be a single number in [0, 1).")
    }

    # ---- Input transform ----
    input_fn = if (!is.null(arch) && !is.null(arch$input_transform)) {
        rlang::as_function(arch$input_transform)
    } else {
        identity
    }

    # ---- Validation split ----
    n_obs = length(dataset)
    if (n_obs < 1L) {
        cli::cli_abort("{.arg dataset} is empty. Provide at least one observation.")
    }

    if (validation_split > 0 && validation_split < 1) {
        n_val = floor(n_obs * validation_split)

        if (n_val < 1L || n_val >= n_obs) {
            cli::cli_abort(c(
                "{.arg validation_split} yields an empty train/validation partition.",
                i = "Use more observations or a different split ratio."
            ))
        }

        val_idx = sample(n_obs, n_val)
        train_idx = setdiff(seq_len(n_obs), val_idx)

        train_ds = torch::dataset_subset(dataset, train_idx)
        val_ds = torch::dataset_subset(dataset, val_idx)
    } else {
        train_ds = dataset
        val_ds = NULL
    }

    # ---- Build model via nn_module_generator() ----
    arch_args = if (!is.null(arch)) {
        args = unclass(arch)
        args$input_transform = NULL
        args
    } else {
        list()
    }

    arch_env = if (!is.null(arch)) attr(arch, "env") else parent.frame()
    
    model_expr = do.call(
        nn_module_generator,
        c(
            list(
                hd_neurons = hidden_neurons,
                no_x = no_x,
                no_y = no_y,
                activations = activations,
                output_activation = output_activation,
                bias = bias
            ),
            arch_args,
            .env = arch_env
        )
    )
    # model = eval(model_expr)()
    model = rlang::eval_tidy(model_expr)()
    model$to(device = device)

    # ---- Dataloaders ----
    train_dl = torch::dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
    val_dl = if (!is.null(val_ds)) torch::dataloader(val_ds, batch_size = batch_size, shuffle = FALSE) else NULL

    # ---- Optimizer ----
    validate_optimizer(tolower(optimizer))
    optimizer_fn = get(paste0("optim_", tolower(optimizer)), envir = asNamespace("torch"))
    opt = do.call(
        optimizer_fn,
        c(list(params = model$parameters, lr = learn_rate), optimizer_args)
    )

    # ---- Loss function ----
    loss_name = tolower(loss)
    loss_fn = switch(
        loss_name,
        mse = function(input, target) torch::nnf_mse_loss(input, target),
        mae = function(input, target) torch::nnf_l1_loss(input, target),
        cross_entropy = function(input, target) torch::nnf_cross_entropy(input, target),
        bce = function(input, target) torch::nnf_binary_cross_entropy_with_logits(input, target),
        cli::cli_abort("Unknown loss function: {loss}")
    )
    is_cross_entropy = identical(loss_name, "cross_entropy")

    # ---- Training loop ----
    loss_history = numeric(epochs)
    val_loss_history = if (!is.null(val_dl)) numeric(epochs) else NULL

    for (epoch in seq_len(epochs)) {
        model$train()
        epoch_loss = 0
        n_batches = 0

        coro::loop(for (batch in train_dl) {
            x_batch = batch[[1]]$to(device = device)
            y_batch = batch[[2]]$to(device = device)

            if (is_classification && is_cross_entropy) {
                y_batch = y_batch$to(dtype = torch::torch_long())$view(c(-1))
            }

            if (flatten_input && length(x_batch$size()) > 2) {
                x_batch = x_batch$view(c(x_batch$size(1), -1))
            }

            x_batch = input_fn(x_batch)

            opt$zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            reg_loss = regularizer(model, penalty, mixture)
            total_loss = loss + reg_loss
            total_loss$backward()
            opt$step()

            epoch_loss = epoch_loss + total_loss$item()
            n_batches = n_batches + 1
        })

        loss_history[epoch] = epoch_loss / n_batches

        if (!is.null(val_dl)) {
            model$eval()
            val_epoch_loss = 0
            val_n_batches = 0

            torch::with_no_grad({
                coro::loop(for (batch in val_dl) {
                    x_batch = batch[[1]]$to(device = device)
                    y_batch = batch[[2]]$to(device = device)

                    if (is_classification && is_cross_entropy) {
                        y_batch = y_batch$to(dtype = torch::torch_long())$view(c(-1))
                    }

                    if (flatten_input && length(x_batch$size()) > 2) {
                        x_batch = x_batch$view(c(x_batch$size(1), -1))
                    }

                    x_batch = input_fn(x_batch)

                    y_pred = model(x_batch)
                    val_loss = loss_fn(y_pred, y_batch)
                    val_epoch_loss = val_epoch_loss + val_loss$item()
                    val_n_batches = val_n_batches + 1
                })
            })

            val_loss_history[epoch] = val_epoch_loss / val_n_batches
        }

        if (verbose && (epoch %% max(1L, epochs %/% 10L) == 0L || epoch == epochs)) {
            msg = sprintf("Epoch %d/%d - Loss: %.4f", epoch, epochs, loss_history[epoch])
            if (!is.null(val_loss_history)) {
                msg = paste0(msg, sprintf(" - Val Loss: %.4f", val_loss_history[epoch]))
            }
            message(msg)
        }
    }

    # ---- No fitted values for datasets (too expensive to materialize) ----
    fitted = NULL

    # ---- Weight caching ----
    cached_weights = NULL
    if (cache_weights) {
        cached_weights = tryCatch(
            lapply(model$parameters, function(p) as.matrix(p$cpu())),
            error = function(e) {
                cli::cli_warn("Weight caching failed: {conditionMessage(e)}")
                NULL
            }
        )
    }

    structure(
        list(
            model = model,
            fitted = fitted,
            loss_history = loss_history,
            val_loss_history = val_loss_history,
            n_epochs = epochs,
            hidden_neurons = hidden_neurons,
            activations = activations,
            output_activation = output_activation,
            penalty = penalty,
            mixture = mixture,
            feature_names = NULL,
            response_name = NULL,
            no_x = no_x,
            no_y = no_y,
            is_classification = is_classification,
            y_levels = if (is_classification) as.character(1:no_y) else NULL,
            n_classes = if (is_classification) no_y else NULL,
            device = device,
            cached_weights = cached_weights,
            flatten_input = flatten_input,
            arch = arch
        ),
        class = unique(c(fit_class, "nn_fit"))
    )
}


#' Predict method for dataset-trained neural networks
#'
#' @rdname gen-nn-predict
#' @keywords internal
#' @export
predict.nn_fit_ds = 
    function(
        object,
        newdata = NULL,
        new_data = NULL,
        type = "response",
        ...
    ) 
{
    newdata = .resolve_predict_newdata(newdata = newdata, new_data = new_data)

    if (!type %in% c("response", "prob")) {
        cli::cli_abort("{.arg type} must be one of {.val response} or {.val prob}.")
    }

    if (!object$is_classification && type == "prob") {
        cli::cli_abort("{.arg type = 'prob'} is only available for classification models.")
    }

    if (is.null(newdata)) {
        cli::cli_abort("Cannot compute fitted values for dataset fits. Provide {.arg newdata}.")
    }

    flatten_input = object$flatten_input %||% TRUE
    device = object$device
    input_fn = if (!is.null(object$arch) && !is.null(object$arch$input_transform)) {
        rlang::as_function(object$arch$input_transform)
    } else {
        identity
    }

    if (inherits(newdata, "dataset")) {
        dl = torch::dataloader(newdata, batch_size = 32, shuffle = FALSE)
        all_preds = list()

        object$model$eval()
        torch::with_no_grad({
            coro::loop(for (batch in dl) {
                x_batch = batch[[1]]$to(device = device)

                if (flatten_input && length(x_batch$size()) > 2) {
                    x_batch = x_batch$view(c(x_batch$size(1), -1))
                }

                x_batch = input_fn(x_batch)
                pred_tensor = object$model(x_batch)

                if (object$is_classification) {
                    probs = torch::nnf_softmax(pred_tensor, dim = 2L)
                    if (type == "prob") {
                        all_preds[[length(all_preds) + 1]] = as.matrix(probs$cpu())
                    } else {
                        pred_classes = torch::torch_argmax(probs, dim = 2L)
                        all_preds[[length(all_preds) + 1]] = as.integer(pred_classes$cpu())
                    }
                } else {
                    all_preds[[length(all_preds) + 1]] = as.matrix(pred_tensor$cpu())
                }
            })
        })

        # Combine batches
        if (object$is_classification) {
            if (type == "prob") {
                prob_matrix = do.call(rbind, all_preds)
                colnames(prob_matrix) = object$y_levels
                return(prob_matrix)
            } else {
                predictions = unlist(all_preds)
                predictions = factor(predictions,
                    levels = seq_along(object$y_levels),
                    labels = object$y_levels)
                return(predictions)
            }
        } else {
            predictions = do.call(rbind, all_preds)
            if (object$no_y == 1L) predictions = as.vector(predictions)
            return(predictions)
        }
    }

    if (is.data.frame(newdata)) {
        x_in = as.matrix(newdata)
    } else if (is.matrix(newdata) || is.array(newdata)) {
        x_in = newdata
    } else {
        cli::cli_abort(c(
            "Unsupported {.arg newdata} type for {.fn predict.nn_fit_ds}.",
            i = "Use a {.cls dataset}, matrix, array, or data.frame."
        ))
    }

    x_new_t = torch::torch_tensor(x_in, dtype = torch::torch_float32(), device = device)
    if (flatten_input && length(x_new_t$size()) > 2) {
        x_new_t = x_new_t$view(c(x_new_t$size(1), -1))
    }
    x_new_t = input_fn(x_new_t)

    object$model$eval()
    pred_tensor = torch::with_no_grad(object$model(x_new_t))

    if (object$is_classification) {
        probs = torch::nnf_softmax(pred_tensor, dim = 2L)

        if (type == "prob") {
            prob_matrix = as.matrix(probs$cpu())
            colnames(prob_matrix) = object$y_levels
            return(prob_matrix)
        }

        pred_classes = torch::torch_argmax(probs, dim = 2L)
        predictions = as.integer(pred_classes$cpu())
        predictions = factor(predictions,
            levels = seq_along(object$y_levels),
            labels = object$y_levels)
        return(predictions)
    }

    predictions = as.matrix(pred_tensor$cpu())
    if (object$no_y == 1L) predictions = as.vector(predictions)
    predictions
}
