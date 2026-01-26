#' @section RNN:
#' Train a recurrent neural network using the torch package.
#'
#' @param rnn_type Character. Type of RNN ("rnn", "lstm", "gru"). Default `"lstm"`.
#' @param bidirectional Logical. Use bidirectional RNN. Default `TRUE`.
#' @param dropout Numeric. Dropout rate between layers. Default `0`.
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'     # Formula interface (original)
#'     model_rnn = rnn(
#'         Sepal.Length ~ .,
#'         data = iris[, 1:4],
#'         hidden_neurons = c(64, 32),
#'         rnn_type = "lstm",
#'         activations = "relu",
#'         epochs = 50
#'     )
#'
#'     # XY interface (new)
#'     model_xy = rnn(
#'         hidden_neurons = c(64, 32),
#'         rnn_type = "gru",
#'         epochs = 50,
#'         x = iris[, 2:4],
#'         y = iris$Sepal.Length
#'     )
#' }
#' }
#'
#' @rdname kindling-basemodels
#' @export
rnn = 
    function(
        formula = NULL,
        data = NULL,
        hidden_neurons,
        rnn_type = "lstm",
        activations = NULL,
        output_activation = NULL,
        bias = TRUE,
        bidirectional = TRUE,
        dropout = 0,
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
    dots = list(...)
    
    # Starting from 0.2.0
    # Use 'hardhat' package instead
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
    
    if (!is.matrix(predictors)) {
        predictors = as.matrix(predictors)
    }
    
    if (is.data.frame(outcomes)) {
        if (ncol(outcomes) == 1) {
            outcomes = outcomes[[1]]
        } else {
            outcomes = as.matrix(outcomes)
        }
    } else if (ncol(outcomes) == 1) {
        outcomes = outcomes[[1]]
    }
    
    fit = rnn_impl(
        x = predictors,
        y = outcomes,
        hidden_neurons = hidden_neurons,
        rnn_type = rnn_type,
        activations = activations,
        output_activation = output_activation,
        bias = bias,
        bidirectional = bidirectional,
        dropout = dropout,
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
    
    fit$blueprint = processed$blueprint
    if (!is.null(formula)) {
        fit$formula = formula
    }
    
    fit
}

#' RNN Implementation
#' @keywords internal
rnn_impl = 
    function(
        x,
        y,
        hidden_neurons,
        rnn_type = "lstm",
        activations = NULL,
        output_activation = NULL,
        bias = TRUE,
        bidirectional = TRUE,
        dropout = 0,
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
        cache_weights = FALSE
    ) {
    if (!requireNamespace("torch", quietly = TRUE)) {
        cli::cli_abort("Package {.pkg torch} is required but not installed.")
    }
    
    if (is.null(device)) {
        device = get_default_device()
    } else {
        device = validate_device(device)
    }
    
    if (verbose) {
        cli::cli_alert_info("Using device: {device}")
    }
    
    validate_regularization(penalty, mixture)
    
    feature_names = colnames(x)
    if (is.null(feature_names)) {
        feature_names = paste0("V", seq_len(ncol(x)))
    }
    
    response_name = if (is.null(names(y))) "y" else names(y)[1]
    
    is_classification = is.factor(y) || is.character(y)
    
    if (is_classification) {
        if (is.character(y)) y = as.factor(y)
        y_levels = levels(y)
        n_classes = length(y_levels)
        y_numeric = as.integer(y)
        no_y = n_classes
        
        if (loss == "mse") {
            loss = "cross_entropy"
            if (verbose) {
                cli::cli_alert("Auto-detected classification task. Using cross_entropy loss.")
            }
        }
    } else {
        y_levels = NULL
        n_classes = NULL
        y_numeric = if (is.matrix(y)) y else as.numeric(y)
        no_y = if (is.matrix(y)) ncol(y) else 1L
    }
    
    no_x = ncol(x)
    n_obs = nrow(x)
    
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
    
    x_train_t = torch::torch_tensor(x_train, dtype = torch::torch_float32(), device = device)$unsqueeze(2)
    
    if (is_classification) {
        y_train_t = torch::torch_tensor(y_train, dtype = torch::torch_long(), device = device)
    } else {
        y_train_t = torch::torch_tensor(
            if (is.matrix(y_train)) y_train else matrix(y_train, ncol = 1),
            dtype = torch::torch_float32(),
            device = device
        )
    }
    
    if (!is.null(x_val)) {
        x_val_t = torch::torch_tensor(x_val, dtype = torch::torch_float32(), device = device)$unsqueeze(2)
        if (is_classification) {
            y_val_t = torch::torch_tensor(y_val, dtype = torch::torch_long(), device = device)
        } else {
            y_val_t = torch::torch_tensor(
                if (is.matrix(y_val)) y_val else matrix(y_val, ncol = 1),
                dtype = torch::torch_float32(),
                device = device
            )
        }
    }
    
    model_expr = rnn_generator(
        nn_name = "RNN",
        hd_neurons = hidden_neurons,
        no_x = no_x,
        no_y = no_y,
        rnn_type = rnn_type,
        activations = activations,
        output_activation = output_activation,
        bias = bias,
        bidirectional = bidirectional,
        dropout = dropout
    )
    model = eval(model_expr)()
    model$to(device = device)
    
    validate_optimizer(tolower(optimizer))
    optimizer_fn = get(paste0("optim_", tolower(optimizer)), envir = asNamespace("torch"))
    opt = do.call(
        optimizer_fn,
        c(list(params = model$parameters, lr = learn_rate), optimizer_args)
    )
    
    loss_fn = switch(
        tolower(loss),
        mse = function(input, target) torch::nnf_mse_loss(input, target),
        mae = function(input, target) torch::nnf_l1_loss(input, target),
        cross_entropy = function(input, target) torch::nnf_cross_entropy(input, target),
        bce = function(input, target) torch::nnf_binary_cross_entropy_with_logits(input, target),
        cli::cli_abort("Unknown loss function: {loss}")
    )
    
    loss_history = numeric(epochs)
    val_loss_history = if (!is.null(x_val)) numeric(epochs) else NULL
    n_batches = ceiling(nrow(x_train) / batch_size)
    
    for (epoch in seq_len(epochs)) {
        model$train()
        epoch_loss = 0
        idx = sample(nrow(x_train))
        
        for (batch in seq_len(n_batches)) {
            start_idx = (batch - 1) * batch_size + 1
            end_idx = min(batch * batch_size, nrow(x_train))
            batch_idx = idx[start_idx:end_idx]
            
            x_batch = x_train_t[batch_idx, , ]
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
        
        if (verbose && (epoch %% max(1, epochs %/% 10) == 0 || epoch == epochs)) {
            msg = sprintf("Epoch %d/%d - Loss: %.4f", epoch, epochs, loss_history[epoch])
            if (!is.null(val_loss_history)) {
                msg = paste0(msg, sprintf(" - Val Loss: %.4f", val_loss_history[epoch]))
            }
            message(msg)
        }
    }
    
    model$eval()
    x_full_t = torch::torch_tensor(x, dtype = torch::torch_float32(), device = device)$unsqueeze(2)
    fitted_tensor = torch::with_no_grad({
        model(x_full_t)
    })
    
    if (is_classification) {
        fitted_probs = torch::nnf_softmax(fitted_tensor, dim = 2)
        fitted_classes = torch::torch_argmax(fitted_probs, dim = 2)
        fitted_values = as.integer(fitted_classes$cpu())
        fitted_values = factor(fitted_values, levels = seq_along(y_levels), labels = y_levels)
    } else {
        fitted_values = as.matrix(fitted_tensor$cpu())
        if (no_y == 1L) fitted_values = as.vector(fitted_values)
    }
    
    cached_weights = NULL
    if (cache_weights) {
        cached_weights = list()
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
            bidirectional = bidirectional,
            no_x = no_x,
            no_y = no_y,
            rnn_type = rnn_type,
            is_classification = is_classification,
            y_levels = y_levels,
            n_classes = n_classes,
            device = device,
            cached_weights = cached_weights
        ),
        class = "rnn_fit"
    )
}

#' @rdname predict-basemodel
#' @export
predict.rnn_fit = function(object, newdata = NULL, new_data = NULL, type = "response", ...) {
    if (!requireNamespace("torch", quietly = TRUE)) {
        cli::cli_abort("Package {.pkg torch} is required but not installed.")
    }
    
    if (!is.null(new_data) && is.null(newdata)) {
        newdata = new_data
    }
    
    device = object$device
    
    if (is.null(newdata)) {
        if (type == "prob" && object$is_classification) {
            cli::cli_abort("Cannot compute probabilities without {.arg newdata}. Use the fitted values instead.")
        }
        return(object$fitted)
    }
    
    if (!is.null(object$blueprint)) {
        processed = hardhat::forge(newdata, object$blueprint)
        x_new = processed$predictors
    } else {
        x_new = newdata
    }
    
    if (!is.matrix(x_new)) {
        x_new = as.matrix(x_new)
    }
    
    x_new_t = torch::torch_tensor(x_new, dtype = torch::torch_float32(), device = device)$unsqueeze(2)
    
    object$model$eval()
    pred_tensor = torch::with_no_grad({
        object$model(x_new_t)
    })
    
    if (object$is_classification) {
        probs = torch::nnf_softmax(pred_tensor, dim = 2)
        
        if (type == "prob") {
            prob_matrix = as.matrix(probs$cpu())
            colnames(prob_matrix) = object$y_levels
            return(prob_matrix)
        } else {
            pred_classes = torch::torch_argmax(probs, dim = 2)
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
`$.rnn_fit` = function(x, name) {
    if (name %in% names(x)) {
        return(x[[name]])
    }
    
    attr(x, name, exact = TRUE)
}
