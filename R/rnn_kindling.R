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
#' - GPU acceleration (CUDA, MPS, or CPU)
#' - Hyperparameter tuning integration
#' - Both regression and classification tasks
#'
#' The `device` parameter controls where computation occurs:
#' - `NULL` (default): Auto-detect best available device (CUDA > MPS > CPU)
#' - `"cuda"`: Use NVIDIA GPU
#' - `"mps"`: Use Apple Silicon GPU
#' - `"cpu"`: Use CPU only
#'
#' @return A model specification object with class `rnn_kindling`.
#'
#' @examples
#' \dontrun{
#' box::use(
#'     recipes[recipe],
#'     workflows[workflow, add_recipe, add_model],
#'     parsnip[fit]
#' )
#'
#' # Model specs
#' rnn_spec = rnn_kindling(
#'     mode = "classification",
#'     hidden_neurons = c(64, 32),
#'     rnn_type = "lstm",
#'     activation = c("relu", "elu"),
#'     epochs = 100,
#'     bidirectional = TRUE
#' )
#'
#' wf = workflow() |>
#'     add_recipe(recipe(Species ~ ., data = iris)) |>
#'     add_model(rnn_spec)
#'
#' fit_wf = fit(wf, data = iris)
#' fit_wf
#'
#' }
#'
#' @export
rnn_kindling =
    function(
        mode = "unknown",
        engine = "kindling",
        hidden_neurons = NULL,
        rnn_type = NULL,
        activations = NULL,
        output_activation = NULL,
        bias = NULL,
        bidirectional = NULL,
        dropout = NULL,
        epochs = NULL,
        batch_size = NULL,
        learn_rate = NULL,
        optimizer = NULL,
        loss = NULL,
        validation_split = NULL,
        device = NULL,
        verbose = NULL
    ) {

    if (!requireNamespace("parsnip", quietly = TRUE)) {
        cli::cli_abort("Package {.pkg parsnip} is required but not installed.")
    }

    args = list(
        hidden_neurons = rlang::enquo(hidden_neurons),
        rnn_type = rlang::enquo(rnn_type),
        activations = rlang::enquo(activations),
        output_activation = rlang::enquo(output_activation),
        bias = rlang::enquo(bias),
        bidirectional = rlang::enquo(bidirectional),
        dropout = rlang::enquo(dropout),
        epochs = rlang::enquo(epochs),
        batch_size = rlang::enquo(batch_size),
        learn_rate = rlang::enquo(learn_rate),
        optimizer = rlang::enquo(optimizer),
        loss = rlang::enquo(loss),
        validation_split = rlang::enquo(validation_split),
        device = rlang::enquo(device),
        verbose = rlang::enquo(verbose)
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

#' @export
#' @importFrom stats update
update.rnn_kindling =
    function(
        object,
        parameters = NULL,
        hidden_neurons = NULL,
        rnn_type = NULL,
        activations = NULL,
        output_activation = NULL,
        bias = NULL,
        bidirectional = NULL,
        dropout = NULL,
        epochs = NULL,
        batch_size = NULL,
        learn_rate = NULL,
        optimizer = NULL,
        loss = NULL,
        validation_split = NULL,
        device = NULL,
        verbose = NULL,
        fresh = FALSE,
        ...
    ) {

    args = list(
        hidden_neurons = rlang::enquo(hidden_neurons),
        rnn_type = rlang::enquo(rnn_type),
        activations = rlang::enquo(activations),
        output_activation = rlang::enquo(output_activation),
        bias = rlang::enquo(bias),
        bidirectional = rlang::enquo(bidirectional),
        dropout = rlang::enquo(dropout),
        epochs = rlang::enquo(epochs),
        batch_size = rlang::enquo(batch_size),
        learn_rate = rlang::enquo(learn_rate),
        optimizer = rlang::enquo(optimizer),
        loss = rlang::enquo(loss),
        validation_split = rlang::enquo(validation_split),
        device = rlang::enquo(device),
        verbose = rlang::enquo(verbose)
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

#' @export
#' @importFrom parsnip translate
translate.rnn_kindling = function(x, engine = x$engine, ...) {
    if (is.null(engine)) {
        cli::cli_abort("Please set an engine with `set_engine()`.")
    }

    x = parsnip::translate.default(x, engine, ...)
    x
}
