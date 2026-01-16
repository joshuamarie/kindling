#' Base models for Neural Network Training in kindling
#'
#' @section FFNN:
#' Train a feed-forward neural network using the torch package.
#'
#' @param formula Formula. Model formula (e.g., y ~ x1 + x2).
#' @param data Data frame. Training data.
#' @param hidden_neurons Integer vector. Number of neurons in each hidden layer.
#' @param activations Activation function specifications. See `act_funs()`.
#' @param output_activation Optional. Activation for output layer.
#' @param bias Logical. Use bias weights. Default `TRUE`.
#' @param epochs Integer. Number of training epochs. Default `100`.
#' @param batch_size Integer. Batch size for training. Default `32`.
#' @param learn_rate Numeric. Learning rate for optimizer. Default `0.001`.
#' @param optimizer Character. Optimizer type ("adam", "sgd", "rmsprop"). Default `"adam"`.
#' @param optimizer_args Named list. Additional arguments passed to the optimizer. Default `list()`.
#' @param loss Character. Loss function ("mse", "mae", "cross_entropy", "bce"). Default `"mse"`.
#' @param validation_split Numeric. Proportion of data for validation (0-1). Default `0`.
#' @param device Character. Device to use ("cpu", "cuda", "mps"). Default `NULL` (auto-detect).
#' @param verbose Logical. Print training progress. Default `FALSE`.
#' @param cache_weights Logical. Cache weight matrices for faster variable importance
#'   computation. Default `FALSE`. When `TRUE`, weight matrices are extracted and
#'   stored in the returned object, avoiding repeated extraction during importance
#'   calculations. Only enable if you plan to compute variable importance multiple times.
#' @param ... Not used. Reserved for future extensions.
#'
#' @return An object of class "ffnn_fit" containing:
#' \item{model}{Trained torch module}
#' \item{formula}{Model formula}
#' \item{fitted.values}{Fitted values on training data}
#' \item{loss_history}{Training loss per epoch}
#' \item{val_loss_history}{Validation loss per epoch (if validation_split > 0)}
#' \item{n_epochs}{Number of epochs trained}
#' \item{feature_names}{Names of predictor variables}
#' \item{response_name}{Name of response variable}
#' \item{device}{Device used for training}
#' \item{cached_weights}{Weight matrices (only if cache_weights = TRUE)}
#'
#' @examples
#' \dontrun{
#' if (torch::torch_is_installed()) {
#'     # Regression task (auto-detect GPU)
#'     model_reg = ffnn(
#'         Sepal.Length ~ .,
#'         data = iris[, 1:4],
#'         hidden_neurons = c(64, 32),
#'         activations = "relu",
#'         epochs = 50,
#'         verbose = FALSE
#'     )
#'
#'     # With weight caching for multiple importance calculations
#'     model_cached = ffnn(
#'         Species ~ .,
#'         data = iris,
#'         hidden_neurons = c(128, 64, 32),
#'         activations = "relu",
#'         cache_weights = TRUE,
#'         epochs = 100
#'     )
#' } else {
#'     message("Torch not fully installed – skipping example")
#' }
#'
#' }
#'
#' @importFrom stats model.frame model.response model.matrix delete.response terms
#'
#' @rdname kindling-basemodels
#' @export
ffnn =
    function(
        formula,
        data,
        hidden_neurons,
        activations = NULL,
        output_activation = NULL,
        bias = TRUE,
        epochs = 100,
        batch_size = 32,
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
        
        if (!requireNamespace("torch", quietly = TRUE)) {
            cli::cli_abort("Package {.pkg torch} is required but not installed.")
        }
        
        # ---Device selection---
        if (is.null(device)) {
            device = get_default_device()
        } else {
            device = validate_device(device)
        }
        
        if (verbose) {
            cli::cli_alert_info("Using device: {device}")
        }
        
        ffnn_call = match.call()
        mf = model.frame(formula, data)
        mt = attr(mf, "terms")
        y = model.response(mf)
        x = model.matrix(formula, mf)[, -1, drop = FALSE]
        
        feature_names = colnames(x)
        response_name = as.character(formula[[2]])
        
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
        
        # ---Torch data conversion---
        x_train_t = torch::torch_tensor(x_train, dtype = torch::torch_float32(), device = device)
        
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
            x_val_t = torch::torch_tensor(x_val, dtype = torch::torch_float32(), device = device)
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
        
        # ---Generate model---
        model_expr = ffnn_generator(
            nn_name = "FFNN",
            hd_neurons = hidden_neurons,
            no_x = no_x,
            no_y = no_y,
            activations = activations,
            output_activation = output_activation,
            bias = bias
        )
        model = eval(model_expr)()
        model$to(device = device)
        
        # ---Optimizer and Loss Function---
        validate_optimizer(tolower(optimizer))
        
        # Create optimizer with proper argument handling
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
        
        # ---Valid number of batches---
        n_batches = ceiling(nrow(x_train) / batch_size)
        
        # ---Start training loop---
        for (epoch in seq_len(epochs)) {
            model$train()
            epoch_loss = 0
            
            idx = sample(nrow(x_train))
            
            for (batch in seq_len(n_batches)) {
                start_idx = (batch - 1) * batch_size + 1
                end_idx = min(batch * batch_size, nrow(x_train))
                batch_idx = idx[start_idx:end_idx]
                
                x_batch = x_train_t[batch_idx, ]
                y_batch = y_train_t[batch_idx]
                
                opt$zero_grad()
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                loss$backward()
                opt$step()
                
                epoch_loss = epoch_loss + loss$item()
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
        
        # ---Predictions---
        model$eval()
        fitted_tensor = torch::with_no_grad({
            model(torch::torch_tensor(x, dtype = torch::torch_float32(), device = device))
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
        
        # ---Cache weights if requested---
        cached_weights = NULL
        if (cache_weights) {
            n_hidden = length(hidden_neurons)
            input_layer = model$fc1
            W_input = as.matrix(input_layer$weight$cpu())
            output_layer = model$out
            W_output = as.matrix(output_layer$weight$cpu())
            
            intermediate_weights = list()
            if (n_hidden > 1) {
                for (i in seq_len(n_hidden - 1)) {
                    layer = model[[paste0("fc", i + 1)]]
                    intermediate_weights[[i]] = as.matrix(layer$weight$cpu())
                }
            }
            
            cached_weights = list(
                input = W_input,
                output = W_output,
                intermediate = intermediate_weights
            )
        }
        
        structure(
            list(
                model = model,
                formula = formula,
                fitted = fitted_values,
                loss_history = loss_history,
                val_loss_history = val_loss_history,
                n_epochs = epochs,
                hidden_neurons = hidden_neurons,
                activations = activations,
                output_activation = output_activation,
                feature_names = feature_names,
                response_name = response_name,
                no_x = no_x,
                no_y = no_y,
                is_classification = is_classification,
                y_levels = y_levels,
                n_classes = n_classes,
                device = device,
                cached_weights = cached_weights,
                terms = mt,
                call = ffnn_call
            ),
            class = "ffnn_fit"
        )
    }

#' @section RNN:
#' Train a recurrent neural network using the torch package.
#'
#' @param rnn_type Character. Type of RNN ("rnn", "lstm", "gru"). Default `"lstm"`.
#' @param bidirectional Logical. Use bidirectional RNN. Default `TRUE`.
#' @param dropout Numeric. Dropout rate between layers. Default `0`.
#'
#' @examples
#' \dontrun{
#' # Regression with LSTM on GPU
#' if (torch::torch_is_installed()) {
#'     model_rnn = rnn(
#'         Sepal.Length ~ .,
#'         data = iris[, 1:4],
#'         hidden_neurons = c(64, 32),
#'         rnn_type = "lstm",
#'         activations = "relu",
#'         epochs = 50
#'     )
#'
#'     # With weight caching
#'     model_cached = rnn(
#'         Species ~ .,
#'         data = iris,
#'         hidden_neurons = c(128, 64),
#'         rnn_type = "gru",
#'         cache_weights = TRUE,
#'         epochs = 100
#'     )
#' } else {
#'     message("Torch not fully installed – skipping example")
#' }
#' }
#'
#' @rdname kindling-basemodels
#' @export
rnn =
    function(
        formula,
        data,
        hidden_neurons,
        rnn_type = "lstm",
        activations = NULL,
        output_activation = NULL,
        bias = TRUE,
        bidirectional = TRUE,
        dropout = 0,
        epochs = 100,
        batch_size = 32,
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
        
        if (!requireNamespace("torch", quietly = TRUE)) {
            cli::cli_abort("Package {.pkg torch} is required but not installed.")
        }
        
        # ---Device selection---
        if (is.null(device)) {
            device = get_default_device()
        } else {
            device = validate_device(device)
        }
        
        if (verbose) {
            cli::cli_alert_info("Using device: {device}")
        }
        
        rnn_call = match.call()
        mf = model.frame(formula, data)
        mt = attr(mf, "terms")
        y = model.response(mf)
        x = model.matrix(formula, mf)[, -1, drop = FALSE]
        
        feature_names = colnames(x)
        response_name = as.character(formula[[2]])
        
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
                    message("Auto-detected classification task. Using cross_entropy loss.")
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
        
        # ---Torch data conversion---
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
        
        # ---Generate model---
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
        
        # ---Optimizer and Loss Function---
        validate_optimizer(tolower(optimizer))
        
        # Create optimizer with proper argument handling
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
        
        # ---Valid number of batches---
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
                loss$backward()
                opt$step()
                
                epoch_loss = epoch_loss + loss$item()
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
        
        # ---Get predictions---
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
        
        # ---Cache weights if requested---
        cached_weights = NULL
        if (cache_weights) {
            # Note: Weight caching for RNNs is more complex due to their architecture
            # This is a placeholder that can be extended based on your needs
            cached_weights = list()
        }
        
        structure(
            list(
                model = model,
                formula = formula,
                fitted = fitted_values,
                loss_history = loss_history,
                val_loss_history = val_loss_history,
                n_epochs = epochs,
                hidden_neurons = hidden_neurons,
                activations = activations,
                output_activation = output_activation,
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
                cached_weights = cached_weights,
                terms = mt,
                call = rnn_call
            ),
            class = "rnn_fit"
        )
    }

#' Predict Method for FFNN Fits
#'
#' @param object An object of class "ffnn_fit".
#' @param newdata Data frame. New data for predictions.
#' @param type Character. Type of prediction: "response" (default) or "prob" for classification.
#' @param ... Additional arguments (unused).
#'
#' @keywords internal
#' @export
predict.ffnn_fit = function(object, newdata = NULL, type = "response", ...) {
    if (!requireNamespace("torch", quietly = TRUE)) {
        cli::cli_abort("Package {.pkg torch} is required but not installed.")
    }
    
    device = object$device
    
    if (is.null(newdata)) {
        if (type == "prob" && object$is_classification) {
            x_t = torch::torch_tensor(
                model.matrix(object$terms, model.frame(object$terms, object$data))[, -1, drop = FALSE],
                dtype = torch::torch_float32(),
                device = device
            )
            object$model$eval()
            pred_tensor = torch::with_no_grad({
                object$model(x_t)
            })
            probs = torch::nnf_softmax(pred_tensor, dim = 2)
            prob_matrix = as.matrix(probs$cpu())
            colnames(prob_matrix) = object$y_levels
            return(prob_matrix)
        } else {
            return(object$fitted)
        }
    }
    
    mt = delete.response(object$terms)
    mf = model.frame(mt, newdata, xlev = NULL)
    x_new = model.matrix(mt, mf)[, -1, drop = FALSE]
    x_new_t = torch::torch_tensor(x_new, dtype = torch::torch_float32(), device = device)
    
    object$model$eval()
    pred_tensor = torch::with_no_grad({
        object$model(x_new_t)
    })
    
    if (object$is_classification) {
        probs = torch::nnf_softmax(pred_tensor, dim = 2)
        
        if (type == "prob") {
            # ---Probability matrix---
            prob_matrix = as.matrix(probs$cpu())
            colnames(prob_matrix) = object$y_levels
            return(prob_matrix)
        } else {
            # ---Predicted classes---
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

#' Predict Method for RNN Fits
#'
#' @param object An object of class "rnn_fit".
#' @param newdata Data frame. New data for predictions.
#' @param type Character. Type of prediction: "response" (default) or "prob" for classification.
#' @param ... Additional arguments (unused).
#'
#' @keywords internal
#' @export
predict.rnn_fit = function(object, newdata = NULL, type = "response", ...) {
    if (!requireNamespace("torch", quietly = TRUE)) {
        cli::cli_abort("Package {.pkg torch} is required but not installed.")
    }
    
    device = object$device
    
    if (is.null(newdata)) {
        if (type == "prob" && object$is_classification) {
            x_t = torch::torch_tensor(
                model.matrix(object$terms, model.frame(object$terms, object$data))[, -1, drop = FALSE],
                dtype = torch::torch_float32(),
                device = device
            )
            object$model$eval()
            pred_tensor = torch::with_no_grad({
                object$model(x_t)
            })
            probs = torch::nnf_softmax(pred_tensor, dim = 2)
            prob_matrix = as.matrix(probs$cpu())
            colnames(prob_matrix) = object$y_levels
            return(prob_matrix)
        } else {
            return(object$fitted)
        }
    }
    
    mt = delete.response(object$terms)
    mf = model.frame(mt, newdata, xlev = NULL)
    x_new = model.matrix(mt, mf)[, -1, drop = FALSE]
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
`$.ffnn_fit` = function(x, name) {
    if (name %in% names(x)) {
        return(x[[name]])
    }
    
    attr(x, name, exact = TRUE)
}

#' @keywords internal
#' @export
`$.rnn_fit` = function(x, name) {
    if (name %in% names(x)) {
        return(x[[name]])
    }
    
    attr(x, name, exact = TRUE)
}
