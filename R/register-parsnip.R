#' Register kindling engines with parsnip
#'
#' @description
#' This function registers the kindling engine for MLP and RNN models with parsnip.
#' It should be called when the package is loaded.
#'
#' @keywords internal
make_kindling = function() {

    # =========================================
    # MLP (Multi-Layer Perceptron) Registration
    # =========================================

    if (!"mlp_kindling" %in% parsnip::get_model_env()$models) {
        parsnip::set_new_model("mlp_kindling")
    }

    parsnip::set_model_mode(model = "mlp_kindling", mode = "regression")
    parsnip::set_model_mode(model = "mlp_kindling", mode = "classification")
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "hidden_neurons",
        original = "hidden_neurons",
        func = list(pkg = "dials", fun = "hidden_units"),
        has_submodel = FALSE
    )
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "activations",
        original = "activations",
        func = list(pkg = "dials", fun = "activation"),
        has_submodel = FALSE
    )
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "output_activation",
        original = "output_activation",
        func = list(pkg = "dials", fun = "activation"),
        has_submodel = FALSE
    )
    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "bias",
        original = "bias",
        func = list(pkg = "foo", fun = "bar"),
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
        func = list(pkg = "foo", fun = "bar"),
        has_submodel = FALSE
    )

    parsnip::set_model_arg(
        model = "mlp_kindling",
        eng = "kindling",
        parsnip = "validation_split",
        original = "validation_split",
        func = list(pkg = "foo", fun = "bar"),
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
            func = c(pkg = "kindling", fun = "ffnn"),
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
            func = c(pkg = "kindling", fun = "ffnn"),
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

    if (!"rnn_kindling" %in% parsnip::get_model_env()$models) {
        parsnip::set_new_model("rnn_kindling")
    }

    parsnip::set_model_mode(model = "rnn_kindling", mode = "regression")
    parsnip::set_model_mode(model = "rnn_kindling", mode = "classification")
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "hidden_neurons",
        original = "hidden_neurons",
        func = list(pkg = "dials", fun = "hidden_units"),
        has_submodel = FALSE
    )
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "rnn_type",
        original = "rnn_type",
        func = list(pkg = "foo", fun = "bar"),
        has_submodel = FALSE
    )
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "activations",
        original = "activations",
        func = list(pkg = "dials", fun = "activation"),
        has_submodel = FALSE
    )
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "output_activation",
        original = "output_activation",
        func = list(pkg = "dials", fun = "activation"),
        has_submodel = FALSE
    )
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "bidirectional",
        original = "bidirectional",
        func = list(pkg = "foo", fun = "bar"),
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
        func = list(pkg = "foo", fun = "bar"),
        has_submodel = FALSE
    )
    parsnip::set_model_arg(
        model = "rnn_kindling",
        eng = "kindling",
        parsnip = "validation_split",
        original = "validation_split",
        func = list(pkg = "foo", fun = "bar"),
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
            func = c(pkg = "kindling", fun = "rnn"),
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
            func = c(pkg = "kindling", fun = "rnn"),
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
