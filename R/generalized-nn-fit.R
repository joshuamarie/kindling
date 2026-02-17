#' Generalized Neural Network Trainer
#'
#' @description
#' `r lifecycle::badge("experimental")`
#'
#' `train_nn()` is a generic function for training neural networks with a
#' user-defined architecture via [nn_arch()]. Dispatch is based on the class
#' of `x`, allowing different preprocessing pipelines per data type:
#'
#' - `train_nn.matrix()` — raw interface, no preprocessing
#' - `train_nn.data.frame()` — tabular interface via `hardhat::mold()`
#' - `train_nn.formula()` — formula interface via `hardhat::mold()`
#'
#' All methods delegate to the shared [train_nn_impl()] core after preprocessing.
#' When `arch = NULL`, the model falls back to a plain FFNN (`nn_linear`) architecture.
#'
#' @param x Predictor data. Dispatch is based on its class:
#'   - `matrix`: used directly, no preprocessing
#'   - `data.frame`: preprocessed via `hardhat::mold()`
#'   - `formula`: combined with `data` and preprocessed via `hardhat::mold()`
#' @param y Outcome data (vector, factor, or matrix). Not used when `x` is a formula.
#' @param data Data frame. Required when `x` is a formula.
#' @param hidden_neurons Integer vector. Number of neurons in each hidden layer.
#' @param activations Activation function specifications. See `act_funs()`.
#' @param output_activation Optional. Activation for the output layer.
#' @param bias Logical. Use bias weights. Default `TRUE`.
#' @param arch An `nn_arch()` object specifying the architecture. Default `NULL` (FFNN fallback).
#' @param epochs Integer. Number of training epochs. Default `100`.
#' @param batch_size Integer. Batch size for training. Default `32`.
#' @param penalty Numeric. Regularization penalty (lambda). Default `0`.
#' @param mixture Numeric. Elastic net mixing parameter (0-1). Default `0`.
#' @param learn_rate Numeric. Learning rate for optimizer. Default `0.001`.
#' @param optimizer Character. Optimizer type (`"adam"`, `"sgd"`, `"rmsprop"`). Default `"adam"`.
#' @param optimizer_args Named list. Additional arguments passed to the optimizer. Default `list()`.
#' @param loss Character. Loss function (`"mse"`, `"mae"`, `"cross_entropy"`, `"bce"`).
#'   Default `"mse"`.
#' @param validation_split Numeric. Proportion of data for validation (0-1). Default `0`.
#' @param device Character. Device to use (`"cpu"`, `"cuda"`, `"mps"`). Default `NULL` (auto-detect).
#' @param verbose Logical. Print training progress. Default `FALSE`.
#' @param cache_weights Logical. Cache weight matrices. Default `FALSE`.
#' @param ... Additional arguments passed to methods.
#'
#' @return An object of class `"nn_fit"`, or a subclass thereof:
#'   - `c("nn_fit_tab", "nn_fit")` when called via `data.frame` or `formula` method
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'     # matrix method
#'     model = train_nn(
#'         x = as.matrix(iris[, 2:4]),
#'         y = iris$Sepal.Length,
#'         hidden_neurons = c(64, 32),
#'         activations = "relu",
#'         epochs = 50
#'     )
#'
#'     # data.frame method
#'     model = train_nn(
#'         x = iris[, 2:4],
#'         y = iris$Sepal.Length,
#'         hidden_neurons = c(64, 32),
#'         activations = "relu",
#'         epochs = 50
#'     )
#'
#'     # formula method
#'     model = train_nn(
#'         x = Sepal.Length ~ .,
#'         data = iris[, 1:4],
#'         hidden_neurons = c(64, 32),
#'         activations = "relu",
#'         epochs = 50
#'     )
#' }
#' }
#'
#' @name gen-nn-train
#' @export
train_nn = function(x, ...) UseMethod("train_nn")

#' @rdname gen-nn-train
#' @export
train_nn.matrix = 
    function(
        x,
        y,
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
        ...
    ) {
    if (!is.null(arch) && !inherits(arch, "nn_arch")) {
        cli::cli_abort("{.arg arch} must be an {.cls nn_arch} object created with {.fn nn_arch}.")
    }
    
    act_specs = eval_act_funs({{ activations }}, {{ output_activation }})
    activations = act_specs$activations
    output_activation = act_specs$output_activation
    
    train_nn_impl(
        x = x,
        y = y,
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
        fit_class = "nn_fit"
    )
}


#' @rdname gen-nn-train
#' @export
train_nn.data.frame = 
    function(
        x,
        y,
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
        ...
    ) {
    if (!is.null(arch) && !inherits(arch, "nn_arch")) {
        cli::cli_abort("{.arg arch} must be an {.cls nn_arch} object created with {.fn nn_arch}.")
    }
    
    act_specs = eval_act_funs({{ activations }}, {{ output_activation }})
    activations = act_specs$activations
    output_activation = act_specs$output_activation
    
    processed = hardhat::mold(x, y)
    
    .train_nn_tab_impl(
        processed = processed,
        formula = NULL,
        hidden_neurons = hidden_neurons,
        activations = activations,
        output_activation = output_activation,
        bias = bias,
        arch = arch,
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
        cache_weights = cache_weights
    )
}


#' @rdname gen-nn-train
#' @export
train_nn.formula = 
    function(
        x,
        data,
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
        ...
    ) {
    if (!is.null(arch) && !inherits(arch, "nn_arch")) {
        cli::cli_abort("{.arg arch} must be an {.cls nn_arch} object created with {.fn nn_arch}.")
    }
    
    if (missing(data) || is.null(data)) {
        cli::cli_abort("{.arg data} must be provided when using the formula interface.")
    }
    
    act_specs = eval_act_funs({{ activations }}, {{ output_activation }})
    activations = act_specs$activations
    output_activation = act_specs$output_activation
    
    processed = hardhat::mold(x, data)
    
    .train_nn_tab_impl(
        processed = processed,
        formula = x,
        hidden_neurons = hidden_neurons,
        activations = activations,
        output_activation = output_activation,
        bias = bias,
        arch = arch,
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
        cache_weights = cache_weights
    )
}


#' @rdname gen-nn-train
#' @export
train_nn.default = function(x, ...) {
    cli::cli_abort(c(
        "No {.fn train_nn} method for class {.cls {class(x)}}.",
        i = "Supported classes: {.cls matrix}, {.cls data.frame}, {.cls formula}."
    ))
}


# Shared post-mold logic for data.frame and formula methods
# @keywords internal
.train_nn_tab_impl = 
    function(
        processed,
        formula,
        hidden_neurons,
        activations,
        output_activation,
        bias,
        arch,
        epochs,
        batch_size,
        penalty,
        mixture,
        learn_rate,
        optimizer,
        optimizer_args,
        loss,
        validation_split,
        device,
        verbose,
        cache_weights
    ) {
    predictors = processed$predictors
    outcomes = processed$outcomes
    
    if (!is.matrix(predictors)) predictors = as.matrix(predictors)
    
    if (is.data.frame(outcomes)) {
        outcomes = if (ncol(outcomes) == 1L) outcomes[[1L]] else as.matrix(outcomes)
    } else if (!is.null(ncol(outcomes)) && ncol(outcomes) == 1L) {
        outcomes = outcomes[[1L]]
    }
    
    fit = train_nn_impl(
        x = predictors,
        y = outcomes,
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
        fit_class = "nn_fit_tab"
    )
    
    fit$blueprint = processed$blueprint
    if (!is.null(formula)) fit$formula = formula
    
    fit
}


#' Shared core implementation
#' @keywords internal
train_nn_impl = 
    function(
        x,
        y,
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
        fit_class = "nn_fit"
    ) {
    if (!requireNamespace("torch", quietly = TRUE)) {
        cli::cli_abort("Package {.pkg torch} is required but not installed.")
    }
    
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
    
    # ---- Metadata ----
    feature_names = colnames(x)
    if (is.null(feature_names)) feature_names = paste0("V", seq_len(ncol(x)))
    
    response_name = if (is.null(names(y))) "y" else names(y)[1L]
    is_classification = is.factor(y) || is.character(y)
    
    # ---- y encoding ----
    if (is_classification) {
        if (is.character(y)) y = as.factor(y)
        y_levels = levels(y)
        n_classes = length(y_levels)
        y_numeric = as.integer(y)
        no_y = n_classes
        
        if (loss == "mse") {
            loss = "cross_entropy"
            if (verbose) cli::cli_alert("Auto-detected classification task. Using cross_entropy loss.")
        }
    } else {
        y_levels = NULL
        n_classes = NULL
        y_numeric = if (is.matrix(y)) y else as.numeric(y)
        no_y = if (is.matrix(y)) ncol(y) else 1L
    }
    
    no_x = ncol(x)
    n_obs = nrow(x)
    
    # ---- Validation split ----
    if (validation_split > 0 && validation_split < 1) {
        n_val = floor(n_obs * validation_split)
        val_idx = sample(n_obs, n_val)
        train_idx = setdiff(seq_len(n_obs), val_idx)
        
        x_train = x[train_idx, , drop = FALSE]
        y_train = y_numeric[train_idx]
        x_val = x[val_idx, , drop = FALSE]
        y_val = y_numeric[val_idx]
    } else {
        x_train = x
        y_train = y_numeric
        x_val = NULL
        y_val = NULL
    }
    
    # ---- Build model via nn_module_generator() ----
    arch_args = if (!is.null(arch)) {
        args = unclass(arch)
        args$input_transform = NULL
        args
    } else {
        list()
    }
    
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
            arch_args
        )
    )
    model = eval(model_expr)()
    model$to(device = device)
    
    # ---- Tensors: training set ----
    x_train_t = input_fn(
        torch::torch_tensor(x_train, dtype = torch::torch_float32(), device = device)
    )
    
    if (is_classification) {
        y_train_t = torch::torch_tensor(y_train, dtype = torch::torch_long(), device = device)
    } else {
        y_train_t = torch::torch_tensor(
            if (is.matrix(y_train)) y_train else matrix(y_train, ncol = 1L),
            dtype = torch::torch_float32(),
            device = device
        )
    }
    
    # ---- Tensors: validation set ----
    if (!is.null(x_val)) {
        x_val_t = input_fn(
            torch::torch_tensor(x_val, dtype = torch::torch_float32(), device = device)
        )
        if (is_classification) {
            y_val_t = torch::torch_tensor(y_val, dtype = torch::torch_long(), device = device)
        } else {
            y_val_t = torch::torch_tensor(
                if (is.matrix(y_val)) y_val else matrix(y_val, ncol = 1L),
                dtype = torch::torch_float32(),
                device = device
            )
        }
    }
    
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
    val_loss_history = if (!is.null(x_val)) numeric(epochs) else NULL
    n_batches = ceiling(nrow(x_train) / batch_size)
    
    for (epoch in seq_len(epochs)) {
        model$train()
        epoch_loss = 0
        idx = sample(nrow(x_train))
        
        for (batch in seq_len(n_batches)) {
            start_idx = (batch - 1L) * batch_size + 1L
            end_idx = min(batch * batch_size, nrow(x_train))
            batch_idx = idx[start_idx:end_idx]
            
            x_batch = x_train_t[batch_idx, ]
            y_batch = y_train_t[batch_idx]
            
            opt$zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            reg_loss = regularizer(model, penalty, mixture)
            total_loss = loss + reg_loss
            total_loss$backward()
            opt$step()
            
            epoch_loss = epoch_loss + total_loss$item()
        }
        
        loss_history[epoch] = epoch_loss / n_batches
        
        if (!is.null(x_val)) {
            model$eval()
            torch::with_no_grad({
                y_val_pred = model(x_val_t)
                val_loss = loss_fn(y_val_pred, y_val_t)
                val_loss_history[epoch] = val_loss$item()
            })
        }
        
        if (verbose && (epoch %% max(1L, epochs %/% 10L) == 0L || epoch == epochs)) {
            msg = sprintf("Epoch %d/%d - Loss: %.4f", epoch, epochs, loss_history[epoch])
            if (!is.null(val_loss_history)) {
                msg = paste0(msg, sprintf(" - Val Loss: %.4f", val_loss_history[epoch]))
            }
            message(msg)
        }
    }
    
    # ---- Fitted values ----
    model$eval()
    x_full_t = input_fn(
        torch::torch_tensor(x, dtype = torch::torch_float32(), device = device)
    )
    fitted_tensor = torch::with_no_grad({ model(x_full_t) })
    
    if (is_classification) {
        fitted_probs = torch::nnf_softmax(fitted_tensor, dim = 2L)
        fitted_classes = torch::torch_argmax(fitted_probs, dim = 2L)
        fitted_values = as.integer(fitted_classes$cpu())
        fitted_values = factor(fitted_values, levels = seq_along(y_levels), labels = y_levels)
    } else {
        fitted_values = as.matrix(fitted_tensor$cpu())
        if (no_y == 1L) fitted_values = as.vector(fitted_values)
    }
    
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
            fitted = fitted_values,
            loss_history = loss_history,
            val_loss_history = val_loss_history,
            n_epochs = epochs,
            hidden_neurons = hidden_neurons,
            activations = activations,
            output_activation = output_activation,
            penalty = penalty,
            mixture = mixture,
            feature_names = feature_names,
            response_name = response_name,
            no_x = no_x,
            no_y = no_y,
            is_classification = is_classification,
            y_levels = y_levels,
            n_classes = n_classes,
            device = device,
            cached_weights = cached_weights,
            arch = arch
        ),
        class = if (tab) c("nn_fit_tab", "nn_fit") else "nn_fit"
    )
}


#' Predict method for nn_fit objects
#'
#' @param object An object of class `"nn_fit"` or `"nn_fit_tab"`.
#' @param newdata Data frame or matrix. New data for predictions.
#' @param new_data Alternative to `newdata` (hardhat-style).
#' @param type Character. `"response"` (default) or `"prob"` (classification only).
#' @param ... Currently unused.
#'
#' @return Numeric vector/matrix (regression) or factor / probability matrix (classification).
#'
#' @name gen-nn-predict
#' @keywords internal
#' @export
predict.nn_fit = function(object, newdata = NULL, new_data = NULL, type = "response", ...) {
    if (!requireNamespace("torch", quietly = TRUE)) {
        cli::cli_abort("Package {.pkg torch} is required but not installed.")
    }
    
    if (!is.null(new_data) && is.null(newdata)) newdata = new_data
    
    device = object$device
    
    # ---- Input transform ----
    input_fn = if (!is.null(object$arch) && !is.null(object$arch$input_transform)) {
        rlang::as_function(object$arch$input_transform)
    } else {
        identity
    }
    
    if (is.null(newdata)) {
        if (type == "prob" && object$is_classification) {
            cli::cli_abort("Cannot compute probabilities without {.arg newdata}. Use fitted values instead.")
        }
        return(object$fitted)
    }
    
    if (!is.matrix(newdata)) newdata = as.matrix(newdata)
    
    x_new_t = input_fn(
        torch::torch_tensor(newdata, dtype = torch::torch_float32(), device = device)
    )
    
    object$model$eval()
    pred_tensor = torch::with_no_grad({ object$model(x_new_t) })
    
    if (object$is_classification) {
        probs = torch::nnf_softmax(pred_tensor, dim = 2L)
        
        if (type == "prob") {
            prob_matrix = as.matrix(probs$cpu())
            colnames(prob_matrix) = object$y_levels
            return(prob_matrix)
        } else {
            pred_classes = torch::torch_argmax(probs, dim = 2L)
            predictions = as.integer(pred_classes$cpu())
            predictions = factor(predictions,
                                 levels = seq_along(object$y_levels),
                                 labels = object$y_levels)
            return(predictions)
        }
    } else {
        predictions = as.matrix(pred_tensor$cpu())
        if (object$no_y == 1L) predictions = as.vector(predictions)
        return(predictions)
    }
}


#' @rdname gen-nn-predict
#' @keywords internal
#' @export
predict.nn_fit_tab = function(object, newdata = NULL, new_data = NULL, type = "response", ...) {
    if (!is.null(new_data) && is.null(newdata)) newdata = new_data
    
    if (!is.null(newdata) && !is.null(object$blueprint)) {
        processed = hardhat::forge(newdata, object$blueprint)
        newdata = as.matrix(processed$predictors)
    }
    
    NextMethod()
}


#' @keywords internal
#' @export
`$.nn_fit` = function(x, name) {
    if (name %in% names(x)) return(x[[name]])
    attr(x, name, exact = TRUE)
}
