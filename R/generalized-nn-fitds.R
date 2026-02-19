#' @param x A torch dataset object.
#' @param y Ignored. Labels come from the dataset itself.
#' @param n_classes Positive integer. Number of output classes. Required when
#'   `x` is a `dataset` with scalar (classification) labels; ignored otherwise.
#'
#' @section Dataset method (`train_nn.dataset()`):
#' Trains a neural network directly on a `torch` dataset object. Batching and
#' lazy loading are handled by `torch::dataloader()`, making this method
#' well-suited for large datasets that do not fit entirely in memory.
#'
#' Labels are taken from the second element of each dataset item (i.e.
#' `dataset[[i]][[2]]`), so `y` is ignored. When the label is a scalar tensor,
#' a classification task is assumed and `n_classes` must be supplied. The loss
#' is automatically switched to `"cross_entropy"` in that case.
#'
#' Fitted values are **not** cached in the returned object. Use
#' [predict.nn_fit_ds()] with `newdata` to obtain predictions after training.
#'
#' @return An object of class `c("nn_fit_ds", "nn_fit")`.
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
#'     model = train_nn(
#'         x = iris_cls_dataset,
#'         hidden_neurons = c(32, 10),
#'         activations = "relu",
#'         epochs = 80,
#'         n_classes = 3 # Iris dataset has only 3 species
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
        hidden_neurons,
        activations = NULL,
        output_activation = NULL,
        bias = TRUE,
        arch = NULL,
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

    if (!is.null(arch) && !inherits(arch, "nn_arch")) {
        cli::cli_abort("{.arg arch} must be an {.cls nn_arch} object created with {.fn nn_arch}.")
    }

    if (!is.null(y)) {
        cli::cli_warn("{.arg y} is ignored when {.arg x} is a dataset. Labels come from the dataset itself.")
    }

    act_specs = eval_act_funs({{ activations }}, {{ output_activation }})
    activations = act_specs$activations
    output_activation = act_specs$output_activation

    # ---- Infer dimensions from first batch ----
    first_item = x[1]
    x_sample = first_item[[1]]
    y_sample = first_item[[2]]
    no_x = as.integer(prod(x_sample$size()))

    is_classification = FALSE
    if (inherits(y_sample, "torch_tensor")) {
        if (y_sample$dtype == torch::torch_long()) {
            is_classification = TRUE
            if (is.null(n_classes)) {
                cli::cli_abort("{.arg n_classes} must be provided for classification datasets.")
            }
            no_y = n_classes
        } else {
            no_y = as.integer(prod(y_sample$size()))
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
        arch = arch,
        fit_class = "nn_fit_ds"
    )
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
        arch = NULL,
        fit_class = "nn_fit_ds"
    ) 
{
    # ---- Device ----
    if (is.null(device)) {
        device = get_default_device()
    } else {
        device = validate_device(device)
    }

    if (verbose) cli::cli_alert_info("Using device: {device}")

    validate_regularization(penalty, mixture)

    # ---- Input transform ----
    input_fn = if (!is.null(arch) && !is.null(arch$input_transform)) {
        rlang::as_function(arch$input_transform)
    } else {
        identity
    }

    # ---- Validation split ----
    n_obs = length(dataset)
    if (validation_split > 0 && validation_split < 1) {
        n_val = floor(n_obs * validation_split)
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
    loss_fn = switch(
        tolower(loss),
        mse = function(input, target) torch::nnf_mse_loss(input, target),
        mae = function(input, target) torch::nnf_l1_loss(input, target),
        cross_entropy = function(input, target) torch::nnf_cross_entropy(input, target),
        bce = function(input, target) torch::nnf_binary_cross_entropy_with_logits(input, target),
        cli::cli_abort("Unknown loss function: {loss}")
    )

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

            if (length(x_batch$size()) > 2) {
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

                    if (length(x_batch$size()) > 2) {
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
            arch = arch
        ),
        class = unique(c(fit_class, "nn_fit"))
    )
}


#' Predict method for nn_fit_ds objects
#'
#' @param object An object of class `"nn_fit_ds"`.
#' @param newdata Dataset or matrix. New data for predictions.
#' @param new_data Alternative to `newdata` (hardhat-style).
#' @param type Character. `"response"` (default) or `"prob"` (classification only).
#' @param ... Currently unused.
#'
#' @return Numeric vector/matrix (regression) or factor / probability matrix (classification).
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
    if (!is.null(new_data) && is.null(newdata)) newdata = new_data

    if (is.null(newdata)) {
        cli::cli_abort("Cannot compute fitted values for dataset fits. Provide {.arg newdata}.")
    }

    if (inherits(newdata, "dataset")) {
        device = object$device
        input_fn = if (!is.null(object$arch) && !is.null(object$arch$input_transform)) {
            rlang::as_function(object$arch$input_transform)
        } else {
            identity
        }

        dl = torch::dataloader(newdata, batch_size = 32, shuffle = FALSE)
        all_preds = list()

        object$model$eval()
        torch::with_no_grad({
            coro::loop(for (batch in dl) {
                x_batch = batch[[1]]$to(device = device)

                if (length(x_batch$size()) > 2) {
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
    } else {
        NextMethod()
    }
}
