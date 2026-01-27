#' Register kindling engines with parsnip
#'
#' @description
#' This function registers the kindling engine for MLP and RNN models with parsnip.
#' It should be called when the package is loaded.
#'
#' @keywords internal
make_kindling = function() {
    model_env = parsnip::get_model_env()
    
    # =========================================
    # MLP (Multi-Layer Perceptron) Registration
    # =========================================
    
    if ("mlp_kindling" %in% model_env$models) {
        if (interactive() || identical(Sys.getenv("DEVTOOLS_LOAD"), "true")) {
            if (exists("mlp_kindling", envir = model_env)) {
                rm(list = "mlp_kindling", envir = model_env)
            }
            model_env$models = setdiff(model_env$models, "mlp_kindling")
        } else {
            if ("rnn_kindling" %in% model_env$models) {
                return(invisible(TRUE))
            }
        }
    }
    
    parsnip::set_new_model("mlp_kindling")
    
    parsnip::set_model_mode(model = "mlp_kindling", mode = "regression")
    parsnip::set_model_mode(model = "mlp_kindling", mode = "classification")
    
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "hidden_neurons",
        original = "hidden_neurons",
        func = list(pkg = "kindling", fun = "hidden_neurons"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "activations",
        original = "activations",
        func = list(pkg = "kindling", fun = "activations"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "output_activation",
        original = "output_activation",
        func = list(pkg = "kindling", fun = "output_activation"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "bias",
        original = "bias",
        func = list(pkg = "kindling", fun = "bias"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "epochs",
        original = "epochs",
        func = list(pkg = "dials", fun = "epochs"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "batch_size",
        original = "batch_size",
        func = list(pkg = "dials", fun = "batch_size"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "penalty",
        original = "penalty",
        func = list(pkg = "dials", fun = "penalty"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "mixture",
        original = "mixture",
        func = list(pkg = "dials", fun = "mixture"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "learn_rate",
        original = "learning_rate",
        func = list(pkg = "dials", fun = "learn_rate"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "optimizer",
        original = "optimizer",
        func = list(pkg = "kindling", fun = "optimizer"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "optimizer_args",
        original = "optimizer_args",
        func = list(pkg = "kindling", fun = "optimizer_args"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "loss",
        original = "loss",
        func = list(pkg = "kindling", fun = "loss"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "validation_split",
        original = "validation_split",
        func = list(pkg = "kindling", fun = "validation_split"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "device",
        original = "device",
        func = list(pkg = "kindling", fun = "device"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "verbose",
        original = "verbose",
        func = list(pkg = "kindling", fun = "verbose"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_engine(
        model = "mlp_kindling",
        mode = "regression",
        eng = "kindling"
    )
    
    parsnip::set_model_engine(
        model = "mlp_kindling",
        mode = "classification",
        eng = "kindling"
    )
    
    parsnip::set_dependency(
        model = "mlp_kindling",
        eng = "kindling",
        pkg = "kindling"
    )
    
    parsnip::set_dependency(
        model = "mlp_kindling",
        eng = "kindling",
        pkg = "torch"
    )
    
    parsnip::set_fit(
        model = "mlp_kindling",
        mode = "regression",
        eng = "kindling",
        value = list(
            interface = "formula",
            protect = c("formula", "data"),
            func = c(pkg = "kindling", fun = "ffnn_wrapper"),
            defaults = list()
        )
    )
    
    parsnip::set_fit(
        model = "mlp_kindling",
        mode = "classification",
        eng = "kindling",
        value = list(
            interface = "formula",
            protect = c("formula", "data"),
            func = c(pkg = "kindling", fun = "ffnn_wrapper"),
            defaults = list(loss = "cross_entropy")
        )
    )
    
    parsnip::set_encoding(
        model = "mlp_kindling",
        mode = "regression",
        eng = "kindling",
        options = list(
            predictor_indicators = "none",
            compute_intercept = FALSE,
            remove_intercept = TRUE,
            allow_sparse_x = FALSE
        )
    )
    
    parsnip::set_encoding(
        model = "mlp_kindling",
        mode = "classification",
        eng = "kindling",
        options = list(
            predictor_indicators = "none",
            compute_intercept = FALSE,
            remove_intercept = TRUE,
            allow_sparse_x = FALSE
        )
    )
    
    parsnip::set_pred(
        model = "mlp_kindling",
        mode = "regression",
        eng = "kindling",
        type = "numeric",
        value = list(
            pre = NULL,
            post = NULL,
            func = c(fun = "predict"),
            args = list(
                object = rlang::expr(object$fit),
                newdata = rlang::expr(new_data),
                type = "response"
            )
        )
    )
    
    parsnip::set_pred(
        model = "mlp_kindling",
        mode = "classification",
        eng = "kindling",
        type = "class",
        value = list(
            pre = NULL,
            post = NULL,
            func = c(fun = "predict"),
            args = list(
                object = rlang::expr(object$fit),
                newdata = rlang::expr(new_data),
                type = "response"
            )
        )
    )
    
    parsnip::set_pred(
        model = "mlp_kindling",
        mode = "classification",
        eng = "kindling",
        type = "prob",
        value = list(
            pre = NULL,
            post = function(result, object) {
                tibble::as_tibble(result)
            },
            func = c(fun = "predict"),
            args = list(
                object = rlang::expr(object$fit),
                newdata = rlang::expr(new_data),
                type = "prob"
            )
        )
    )
    
    # ===========================================
    # RNN (Recurrent Neural Network) Registration
    # ===========================================
    
    if ("rnn_kindling" %in% model_env$models) {
        if (interactive() || identical(Sys.getenv("DEVTOOLS_LOAD"), "true")) {
            if (exists("rnn_kindling", envir = model_env)) {
                rm(list = "rnn_kindling", envir = model_env)
            }
            model_env$models = setdiff(model_env$models, "rnn_kindling")
        } else {
            if ("rnn_kindling" %in% model_env$models) {
                return(invisible(TRUE))
            }
        }
    }
    
    parsnip::set_new_model("rnn_kindling")
    
    parsnip::set_model_mode(model = "rnn_kindling", mode = "regression")
    parsnip::set_model_mode(model = "rnn_kindling", mode = "classification")
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "hidden_neurons",
        original = "hidden_neurons",
        func = list(pkg = "kindling", fun = "hidden_neurons"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "rnn_type",
        original = "rnn_type",
        func = list(pkg = "kindling", fun = "rnn_type"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "activations",
        original = "activations",
        func = list(pkg = "kindling", fun = "activations"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "output_activation",
        original = "output_activation",
        func = list(pkg = "kindling", fun = "output_activation"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "bidirectional",
        original = "bidirectional",
        func = list(pkg = "kindling", fun = "bidirectional"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "dropout",
        original = "dropout",
        func = list(pkg = "dials", fun = "dropout"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "epochs",
        original = "epochs",
        func = list(pkg = "dials", fun = "epochs"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "batch_size",
        original = "batch_size",
        func = list(pkg = "dials", fun = "batch_size"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "penalty",
        original = "penalty",
        func = list(pkg = "dials", fun = "penalty"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "mixture",
        original = "mixture",
        func = list(pkg = "dials", fun = "mixture"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "learn_rate",
        original = "learning_rate",
        func = list(pkg = "dials", fun = "learn_rate"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "optimizer",
        original = "optimizer",
        func = list(pkg = "kindling", fun = "optimizer"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "optimizer_args",
        original = "optimizer_args",
        func = list(pkg = "kindling", fun = "optimizer_args"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "loss",
        original = "loss",
        func = list(pkg = "kindling", fun = "loss"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "validation_split",
        original = "validation_split",
        func = list(pkg = "kindling", fun = "validation_split"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "device",
        original = "device",
        func = list(pkg = "kindling", fun = "device"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "verbose",
        original = "verbose",
        func = list(pkg = "kindling", fun = "verbose"),
        has_submodel = FALSE
    )
    
    parsnip::set_model_engine(
        model = "rnn_kindling",
        mode = "regression",
        eng = "kindling"
    )
    
    parsnip::set_model_engine(
        model = "rnn_kindling",
        mode = "classification",
        eng = "kindling"
    )
    
    parsnip::set_dependency(
        model = "rnn_kindling",
        eng = "kindling",
        pkg = "kindling"
    )
    
    parsnip::set_dependency(
        model = "rnn_kindling",
        eng = "kindling",
        pkg = "torch"
    )
    
    parsnip::set_fit(
        model = "rnn_kindling",
        mode = "regression",
        eng = "kindling",
        value = list(
            interface = "formula",
            protect = c("formula", "data"),
            func = c(pkg = "kindling", fun = "rnn_wrapper"),
            defaults = list()
        )
    )
    
    parsnip::set_fit(
        model = "rnn_kindling",
        mode = "classification",
        eng = "kindling",
        value = list(
            interface = "formula",
            protect = c("formula", "data"),
            func = c(pkg = "kindling", fun = "rnn_wrapper"),
            defaults = list(loss = "cross_entropy")
        )
    )
    
    parsnip::set_encoding(
        model = "rnn_kindling",
        mode = "regression",
        eng = "kindling",
        options = list(
            predictor_indicators = "none",
            compute_intercept = FALSE,
            remove_intercept = TRUE,
            allow_sparse_x = FALSE
        )
    )
    
    parsnip::set_encoding(
        model = "rnn_kindling",
        mode = "classification",
        eng = "kindling",
        options = list(
            predictor_indicators = "none",
            compute_intercept = FALSE,
            remove_intercept = TRUE,
            allow_sparse_x = FALSE
        )
    )
    
    parsnip::set_pred(
        model = "rnn_kindling",
        mode = "regression",
        eng = "kindling",
        type = "numeric",
        value = list(
            pre = NULL,
            post = NULL,
            func = c(fun = "predict"),
            args = list(
                object = rlang::expr(object$fit),
                newdata = rlang::expr(new_data),
                type = "response"
            )
        )
    )
    
    parsnip::set_pred(
        model = "rnn_kindling",
        mode = "classification",
        eng = "kindling",
        type = "class",
        value = list(
            pre = NULL,
            post = NULL,
            func = c(fun = "predict"),
            args = list(
                object = rlang::expr(object$fit),
                newdata = rlang::expr(new_data),
                type = "response"
            )
        )
    )
    
    parsnip::set_pred(
        model = "rnn_kindling",
        mode = "classification",
        eng = "kindling",
        type = "prob",
        value = list(
            pre = NULL,
            post = function(result, object) {
                tibble::as_tibble(result)
            },
            func = c(fun = "predict"),
            args = list(
                object = rlang::expr(object$fit),
                newdata = rlang::expr(new_data),
                type = "prob"
            )
        )
    )
    
    invisible(TRUE)
}

#' Basemodels-tidymodels wrappers
#'
#' @param formula A formula specifying the model (e.g., `y ~ x1 + x2`)
#' @param data A data frame containing the training data
#' @param ... Additional arguments passed to the underlying training function
#'
#' @return
#' * `ffnn_wrapper()` returns an object of class `"ffnn_fit"` containing the trained feedforward neural network model and metadata. See [ffnn()] for details.
#' * `rnn_wrapper()` returns an object of class `"rnn_fit"` containing the trained recurrent neural network model and metadata. See [rnn()] for details.
#'
#' @rdname kindling-nn-wrappers
#' @section FFNN (MLP) Wrapper for `{tidymodels}` interface:
#' This is a function to interface into `{tidymodels}`
#' (do not use this, use `kindling::ffnn()` instead).
#'
#' @export
ffnn_wrapper = function(formula, data, ...) {
    dots = list(...)
    dots = prepare_kindling_args(dots)
    
    do.call(
        ffnn,
        c(list(formula = formula, data = data), dots)
    )
}

#' @rdname kindling-nn-wrappers
#' @section RNN Wrapper for `{tidymodels}` interface:
#' This is a function to interface into `{tidymodels}`
#' (do not use this, use `kindling::rnn()` instead).
#'
#' @export
rnn_wrapper = function(formula, data, ...) {
    dots = list(...)
    dots = prepare_kindling_args(dots)
    
    do.call(
        rnn,
        c(list(formula = formula, data = data), dots)
    )
}
