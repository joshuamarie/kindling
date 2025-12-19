#' Recurrent Neural Network via kindling
#'
#' `rnn_kindling()` defines a recurrent neural network model that can be used
#' for classification or regression on sequential data. It integrates with the
#' tidymodels ecosystem and uses the torch backend via kindling.
#'
#' @inheritParams mlp_kindling
#' @param rnn_type A character string for the type of RNN cell ("rnn", "lstm",
#'   "gru"). Can be tuned.
#' @param bidirectional A logical indicating whether to use bidirectional RNN.
#'   Can be tuned.
#' @param dropout A number between 0 and 1 for dropout rate between layers.
#'   Can be tuned.
#'
#' @details
#' This function creates a model specification for a recurrent neural network
#' that can be used within tidymodels workflows. The model supports:
#'
#' - Multiple RNN types: basic RNN, LSTM, and GRU
#' - Bidirectional processing
#' - Dropout regularization
#' - Hyperparameter tuning integration
#' - Both regression and classification tasks
#'
#' @return A model specification object with class `rnn_kindling`.
#'
#' @examples
#' \dontrun{
#' library(parsnip)
#' library(workflows)
#'
#' # Model Specs for LSTM
#' rnn_spec = rnn_kindling(
#'     mode = "classification",
#'     hidden_neurons = c(64, 32),
#'     rnn_type = "lstm",
#'     activation = c("relu", "elu"),
#'     epochs = 100,
#'     bidirectional = TRUE
#' )
#'
#' # Tune RNN
#' rnn_tune_spec = rnn_kindling(
#'     mode = "regression",
#'     hidden_neurons = tune(),
#'     rnn_type = tune(),
#'     activation = tune(),
#'     dropout = tune()
#' )
#' }
#'
#' @export
rnn_kindling = function(mode = "unknown",
                        engine = "kindling",
                        hidden_neurons = NULL,
                        rnn_type = NULL,
                        activations = NULL,
                        output_activation = NULL,
                        bidirectional = NULL,
                        dropout = NULL,
                        epochs = NULL,
                        batch_size = NULL,
                        learn_rate = NULL,
                        optimizer = NULL,
                        validation_split = NULL) {

    if (!requireNamespace("parsnip", quietly = TRUE)) {
        cli::cli_abort("Package {.pkg parsnip} is required but not installed.")
    }

    args = list(
        hidden_neurons = rlang::enquo(hidden_neurons),
        rnn_type = rlang::enquo(rnn_type),
        activations = rlang::enquo(activations),
        output_activation = rlang::enquo(output_activation),
        bidirectional = rlang::enquo(bidirectional),
        dropout = rlang::enquo(dropout),
        epochs = rlang::enquo(epochs),
        batch_size = rlang::enquo(batch_size),
        learn_rate = rlang::enquo(learn_rate),
        optimizer = rlang::enquo(optimizer),
        validation_split = rlang::enquo(validation_split)
    )

    parsnip::new_model_spec(
        "rnn_kindling",
        args = args,
        eng_args = NULL,
        mode = mode,
        user_specified_mode = !missing(mode),
        method = NULL,
        engine = engine,
        user_specified_engine = !missing(engine)
    )
}

#' @export
print.rnn_kindling = function(x, ...) {
    cat("Recurrent Neural Network Model Specification (", x$mode, ")\n\n", sep = "")
    parsnip::model_printer(x, ...)
    invisible(x)
}

#' @exportS3Method recipe::update
update.rnn_kindling = function(object,
                               parameters = NULL,
                               hidden_neurons = NULL,
                               rnn_type = NULL,
                               activations = NULL,
                               output_activation = NULL,
                               bidirectional = NULL,
                               dropout = NULL,
                               epochs = NULL,
                               batch_size = NULL,
                               learn_rate = NULL,
                               optimizer = NULL,
                               validation_split = NULL,
                               fresh = FALSE,
                               ...) {

    args = list(
        hidden_neurons = rlang::enquo(hidden_neurons),
        rnn_type = rlang::enquo(rnn_type),
        activations = rlang::enquo(activations),
        output_activation = rlang::enquo(output_activation),
        bidirectional = rlang::enquo(bidirectional),
        dropout = rlang::enquo(dropout),
        epochs = rlang::enquo(epochs),
        batch_size = rlang::enquo(batch_size),
        learn_rate = rlang::enquo(learn_rate),
        optimizer = rlang::enquo(optimizer),
        validation_split = rlang::enquo(validation_split)
    )

    parsnip::update_spec(
        object = object,
        parameters = parameters,
        args_enquo_list = args,
        fresh = fresh,
        cls = "rnn_kindling",
        ...
    )
}

#' @exportS3Method parsnip::translate
translate.rnn_kindling = function(x, engine = x$engine, ...) {
    if (is.null(engine)) {
        cli::cli_abort("Please set an engine with `set_engine()`.")
    }

    x = parsnip::translate_default(x, engine, ...)
    x
}
