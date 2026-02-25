#' Recurrent Neural Network via kindling
#'
#' `rnn_kindling()` defines a recurrent neural network model that can be used
#' for classification or regression on sequential data. It integrates with the
#' tidymodels ecosystem and uses the torch backend via kindling.
#'
#' @inheritParams mlp_kindling
#' @param rnn_type A character string for the type of RNN cell ("rnn", "lstm",
#'   "gru"). Cannot be tuned — pass via `set_engine()`.
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
#' \donttest{
#' if (torch::torch_is_installed()) {
#'     box::use(
#'         recipes[recipe],
#'         workflows[workflow, add_recipe, add_model],
#'         parsnip[fit]
#'     )
#'
#'     # Model specs
#'     rnn_spec = rnn_kindling(
#'         mode = "classification",
#'         hidden_neurons = c(64, 32),
#'         rnn_type = "lstm",
#'         activation = c("relu", "elu"),
#'         epochs = 100,
#'         bidirectional = TRUE
#'     )
#'
#'     wf = workflow() |>
#'         add_recipe(recipe(Species ~ ., data = iris)) |>
#'         add_model(rnn_spec)
#'
#'     fit_wf = fit(wf, data = iris)
#'     fit_wf
#' } else {
#'     message("Torch not fully installed — skipping example")
#' }
#' }
#'
#' @export
rnn_kindling = 
    function(
        mode = "unknown",
        engine = "kindling",
        # ---- Tunable parameters ----
        hidden_neurons = NULL,
        activations = NULL,
        output_activation = NULL,
        bias = NULL,
        bidirectional = NULL,
        dropout = NULL,
        epochs = NULL,
        batch_size = NULL,
        penalty = NULL,
        mixture = NULL,
        learn_rate = NULL,
        optimizer = NULL,
        validation_split = NULL,
        # ---- Non-tunable parameters (engine-level, set via set_engine()) ----
        rnn_type = NULL,
        optimizer_args = NULL,
        loss = NULL,
        early_stopping = NULL,
        device = NULL,
        verbose = NULL,
        cache_weights = NULL
    )
{
    
    if (!requireNamespace("parsnip", quietly = TRUE)) {
        cli::cli_abort("Package {.pkg parsnip} is required but not installed.")
    }
    
    # ---- "Tunable" args ----
    args = list(
        hidden_neurons = rlang::enquo(hidden_neurons),
        activations = rlang::enquo(activations),
        output_activation = rlang::enquo(output_activation),
        bias = rlang::enquo(bias),
        bidirectional = rlang::enquo(bidirectional),
        dropout = rlang::enquo(dropout),
        epochs = rlang::enquo(epochs),
        batch_size = rlang::enquo(batch_size),
        penalty = rlang::enquo(penalty),
        mixture = rlang::enquo(mixture),
        learn_rate = rlang::enquo(learn_rate),
        optimizer = rlang::enquo(optimizer),
        validation_split = rlang::enquo(validation_split)
    )
    
    # ---- "Non-tunable" engine args ----
    eng_args = list(
        rnn_type = rlang::enquo(rnn_type),
        optimizer_args = rlang::enquo(optimizer_args),
        loss = rlang::enquo(loss),
        early_stopping = rlang::enquo(early_stopping),
        device = rlang::enquo(device),
        verbose = rlang::enquo(verbose),
        cache_weights = rlang::enquo(cache_weights)
    )
    
    parsnip::new_model_spec(
        "rnn_kindling",
        args = args,
        eng_args = eng_args,
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
        activations = NULL,
        output_activation = NULL,
        bias = NULL,
        bidirectional = NULL,
        dropout = NULL,
        epochs = NULL,
        batch_size = NULL,
        penalty = NULL,
        mixture = NULL,
        learn_rate = NULL,
        optimizer = NULL,
        validation_split = NULL,
        rnn_type = NULL,
        optimizer_args = NULL,
        loss = NULL,
        early_stopping = NULL,
        device = NULL,
        verbose = NULL,
        cache_weights = NULL,
        fresh = FALSE,
        ...
    )
{
    
    args = list(
        hidden_neurons = rlang::enquo(hidden_neurons),
        activations = rlang::enquo(activations),
        output_activation = rlang::enquo(output_activation),
        bias = rlang::enquo(bias),
        bidirectional = rlang::enquo(bidirectional),
        dropout = rlang::enquo(dropout),
        epochs = rlang::enquo(epochs),
        batch_size = rlang::enquo(batch_size),
        penalty = rlang::enquo(penalty),
        mixture = rlang::enquo(mixture),
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

#' @export
#' @importFrom parsnip translate
translate.rnn_kindling = function(x, engine = x$engine, ...) {
    if (is.null(engine)) {
        cli::cli_abort("Please set an engine with `set_engine()`.")
    }
    
    x = parsnip::translate.default(x, engine, ...)
    x
}

#' @export
#' @importFrom tune tunable
tunable.rnn_kindling = function(x, ...) {
    tibble::tibble(
        name = c(
            "hidden_neurons", "activations", "output_activation",
            "bias", "bidirectional", "dropout", "epochs", "batch_size",
            "penalty", "mixture", "learn_rate", "optimizer",
            "validation_split"
        ),
        call_info = list(
            list(pkg = "kindling", fun = "hidden_neurons"),
            list(pkg = "kindling", fun = "activations"),
            list(pkg = "kindling", fun = "output_activation"),
            list(pkg = "kindling", fun = "bias"),
            list(pkg = "kindling", fun = "bidirectional"),
            list(pkg = "dials", fun = "dropout"),
            list(pkg = "dials", fun = "epochs"),
            list(pkg = "dials", fun = "batch_size"),
            list(pkg = "dials", fun = "penalty"),
            list(pkg = "dials", fun = "mixture"),
            list(pkg = "dials", fun = "learn_rate"),
            list(pkg = "kindling", fun = "optimizer"),
            list(pkg = "kindling", fun = "validation_split")
        ),
        source = "model_spec",
        component = "rnn_kindling",
        component_id = "main"
    )
}

#' @rdname kindling-nn-wrappers
#' @section RNN Wrapper for `{tidymodels}` interface:
#' Internal wrapper — use `rnn_kindling()` + `fit()` instead.
#'
#' @keywords internal
#' @export
rnn_wrapper = function(formula, data, ...) {
    dots = list(...)
    dots = prepare_kindling_args(dots)
    
    do.call(
        rnn,
        c(list(formula = formula, data = data), dots)
    )
}
