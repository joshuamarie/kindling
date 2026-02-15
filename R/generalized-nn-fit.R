#' Architecture specification for train_nn()
#'
#' @description
#' `r lifecycle::badge("experimental")`
#' 
#' `nn_arch()` is a helper that bundles `nn_module_generator()` arguments into a
#' single object passed to `train_nn()` via the `arch` parameter. All arguments
#' mirror those of `nn_module_generator()` exactly, including their defaults.
#'
#' @param nn_name Character. Name of the generated module class. Default `"nnModule"`.
#' @param nn_layer Layer type. See `nn_module_generator()`. Default `NULL` (`nn_linear`).
#' @param out_nn_layer Optional. Layer type forced on the last layer. Default `NULL`.
#' @param nn_layer_args Named list. Additional arguments passed to every layer constructor.
#'   Default `list()`.
#' @param layer_arg_fn Formula or function. Generates per-layer constructor arguments.
#'   Default `NULL` (FFNN-style: `list(in_dim, out_dim, bias = bias)`).
#' @param forward_extract Formula or function. Processes layer output in the forward pass.
#'   Default `NULL`.
#' @param before_output_transform Formula or function. Transforms input before the output
#'   layer. Default `NULL`.
#' @param after_output_transform Formula or function. Transforms output after the output
#'   layer. Default `NULL`.
#' @param last_layer_args Named list or formula. Extra arguments for the output layer only.
#'   Default `list()`.
#' @param use_namespace Logical or character. Controls torch namespace prefixing.
#'   Default `TRUE`.
#'
#' @return An object of class `"nn_arch"`, a named list of `nn_module_generator()` arguments.
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'     # LSTM architecture spec
#'     lstm_arch = nn_arch(
#'         nn_name = "LSTM",
#'         nn_layer = "nn_lstm",
#'         layer_arg_fn = ~ if (.is_output) {
#'             list(.in, .out)
#'         } else {
#'             list(input_size = .in, hidden_size = .out, batch_first = TRUE)
#'         },
#'         forward_extract = ~ .[[1]],
#'         before_output_transform = ~ .[, .$size(2), ]
#'     )
#'
#'     model = train_nn(
#'         Sepal.Length ~ .,
#'         data = iris[, 1:4],
#'         hidden_neurons = c(64, 32),
#'         activations = "relu",
#'         epochs = 50,
#'         arch = lstm_arch
#'     )
#' }
#' }
#'
#' @export
nn_arch = 
    function(
        nn_name = "nnModule",
        nn_layer = NULL,
        out_nn_layer = NULL,
        nn_layer_args = list(),
        layer_arg_fn = NULL,
        forward_extract = NULL,
        before_output_transform = NULL,
        after_output_transform = NULL,
        last_layer_args = list(),
        use_namespace = TRUE
    ) {
    structure(
        list(
            nn_name = nn_name,
            nn_layer = nn_layer,
            out_nn_layer = out_nn_layer,
            nn_layer_args = nn_layer_args,
            layer_arg_fn = layer_arg_fn,
            forward_extract = forward_extract,
            before_output_transform = before_output_transform,
            after_output_transform = after_output_transform,
            last_layer_args = last_layer_args,
            use_namespace = use_namespace
        ),
        class = "nn_arch"
    )
}

#' @export
print.nn_arch = function(x, ...) {
    cli::cli_h3("Neural Network Architecture Spec")
    cli::cli_bullets(c(
        "*" = "Name:      {x$nn_name}",
        "*" = "Layer:     {x$nn_layer %||% 'nn_linear (default)'}",
        "*" = "Out layer: {x$out_nn_layer %||% 'same as nn_layer'}",
        "*" = "Namespace: {x$use_namespace}"
    ))
    invisible(x)
}


#' Generalized Neural Network Trainer
#'
#' @description
#' `r lifecycle::badge("experimental")`
#' 
#' Train a neural network with a user-defined architecture supplied via `nn_arch()`.
#' `train_nn()` handles data preprocessing, the training loop, and prediction,
#' while the model architecture is fully controlled through the `arch` argument.
#' When `arch = NULL`, it falls back to a plain FFNN (`nn_linear`) architecture.
#'
#' @param formula Formula. Model formula (e.g., `y ~ x1 + x2`).
#' @param data Data frame. Training data.
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
#' @param ... Additional arguments. It is currently unused. 
#' @param x When not using formula: predictor data (data.frame or matrix).
#' @param y When not using formula: outcome data (vector, factor, or matrix).
#'
#' @return An object of class `"nn_fit"`.
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'     # Default FFNN fallback (arch = NULL)
#'     model_ffnn = train_nn(
#'         Sepal.Length ~ .,
#'         data = iris[, 1:4],
#'         hidden_neurons = c(64, 32),
#'         activations = "relu",
#'         epochs = 50
#'     )
#'
#'     # GRU via nn_arch()
#'     gru_arch = nn_arch(
#'         nn_name = "GRU",
#'         nn_layer = "nn_gru",
#'         layer_arg_fn = ~ if (.is_output) {
#'             list(.in, .out)
#'         } else {
#'             list(input_size = .in, hidden_size = .out, batch_first = TRUE)
#'         },
#'         forward_extract = ~ .[[1]],
#'         before_output_transform = ~ .[, .$size(2), ]
#'     )
#'
#'     model_gru = train_nn(
#'         hidden_neurons = c(64, 32),
#'         activations = "relu",
#'         epochs = 50,
#'         arch = gru_arch,
#'         x = iris[, 2:4],
#'         y = iris$Sepal.Length
#'     )
#' }
#' }
#'
#' @export
train_nn = 
    function(
        formula = NULL,
        data = NULL,
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
        ...,
        x = NULL,
        y = NULL
    ) {
    
    if (!is.null(arch) && !inherits(arch, "nn_arch")) {
        cli::cli_abort("{.arg arch} must be an {.cls nn_arch} object created with {.fn nn_arch}.")
    }
    
    act_specs = eval_act_funs({{ activations }}, {{ output_activation }})
    activations = act_specs$activations
    output_activation = act_specs$output_activation
    
    # ---- Data ingestion ----
    if (!is.null(x) || !is.null(y)) {
        if (is.null(x) || is.null(y)) {
            cli::cli_abort("Both {.arg x} and {.arg y} must be provided when not using formula interface.")
        }
        if (!is.null(formula) || !is.null(data)) {
            cli::cli_warn("Both formula/data and x/y provided. Using x/y interface.")
        }
        processed = hardhat::mold(x, y)
        
    } else if (!is.null(formula)) {
        if (is.null(data)) {
            cli::cli_abort("{.arg data} must be provided when using formula interface.")
        }
        processed = hardhat::mold(formula, data)
        
    } else {
        cli::cli_abort("Must provide either {.arg formula} and {.arg data}, or {.arg x} and {.arg y}.")
    }
    
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
        arch = arch
    )
    
    fit$blueprint = processed$blueprint
    if (!is.null(formula)) fit$formula = formula
    
    fit
}


#' train_nn Implementation
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
        arch = NULL
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
    # arch = NULL falls back to nn_linear (FFNN-style), matching nn_module_generator() defaults
    arch_args = if (!is.null(arch)) unclass(arch) else list()
    
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
    x_train_t = torch::torch_tensor(x_train, dtype = torch::torch_float32(), device = device)
    
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
        x_val_t = torch::torch_tensor(x_val, dtype = torch::torch_float32(), device = device)
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
    x_full_t = torch::torch_tensor(x, dtype = torch::torch_float32(), device = device)
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
        class = "nn_fit"
    )
}


#' Predict method for nn_fit objects
#'
#' @param object An object of class `"nn_fit"`.
#' @param newdata Data frame. New data for predictions.
#' @param new_data Alternative to `newdata` (hardhat-style).
#' @param type Character. `"response"` (default) or `"prob"` (classification only).
#' @param ... Currently unused.
#'
#' @return Numeric vector/matrix (regression) or factor / probability matrix (classification).
#'
#' @export
predict.nn_fit = function(object, newdata = NULL, new_data = NULL, type = "response", ...) {
    if (!requireNamespace("torch", quietly = TRUE)) {
        cli::cli_abort("Package {.pkg torch} is required but not installed.")
    }
    
    if (!is.null(new_data) && is.null(newdata)) newdata = new_data
    
    device = object$device
    
    if (is.null(newdata)) {
        if (type == "prob" && object$is_classification) {
            cli::cli_abort("Cannot compute probabilities without {.arg newdata}. Use fitted values instead.")
        }
        return(object$fitted)
    }
    
    if (!is.null(object$blueprint)) {
        processed = hardhat::forge(newdata, object$blueprint)
        x_new = processed$predictors
    } else {
        x_new = newdata
    }
    
    if (!is.matrix(x_new)) x_new = as.matrix(x_new)
    
    x_new_t = torch::torch_tensor(x_new, dtype = torch::torch_float32(), device = device)
    
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


#' @keywords internal
#' @export
`$.nn_fit` = function(x, name) {
    if (name %in% names(x)) return(x[[name]])
    attr(x, name, exact = TRUE)
}
