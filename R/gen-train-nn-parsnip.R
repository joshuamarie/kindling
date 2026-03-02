#' Parsnip Interface of `train_nn()`
#'
#' @description
#' `r lifecycle::badge("experimental")`
#'
#' `train_nnsnip()` defines a neural network model specification that can be used
#' for classification or regression. It integrates with the tidymodels ecosystem
#' and uses [train_nn()] as the fitting backend, supporting any architecture
#' expressible via [nn_arch()] — feedforward, recurrent, convolutional, and beyond.
#'
#' @param mode A single character string for the type of model. Possible values
#'   are "unknown", "regression", or "classification".
#' @param engine A single character string specifying what computational engine
#'   to use for fitting. Currently only "kindling" is supported.
#' @param hidden_neurons An integer vector for the number of units in each hidden
#'   layer. Can be tuned.
#' @param activations A character vector of activation function names for each
#'   hidden layer (e.g., "relu", "tanh", "sigmoid"). Can be tuned.
#' @param output_activation A character string for the output activation function.
#'   Can be tuned.
#' @param bias Logical for whether to include bias terms. Can be tuned.
#' @param epochs An integer for the number of training iterations. Can be tuned.
#' @param penalty A number for the regularization penalty (lambda). Default `0`
#'   (no regularization). Higher values increase regularization strength. Can be tuned.
#' @param mixture A number between 0 and 1 for the elastic net mixing parameter.
#'   Default `0` (pure L2/Ridge regularization).
#'   - `0`: Pure L2 regularization (Ridge)
#'   - `1`: Pure L1 regularization (Lasso)
#'   - `0 < mixture < 1`: Elastic net (combination of L1 and L2)
#'   Only relevant when `penalty > 0`. Can be tuned.
#' @param batch_size An integer for the batch size during training. Can be tuned.
#' @param learn_rate A number for the learning rate. Can be tuned.
#' @param optimizer A character string for the optimizer type ("adam", "sgd",
#'   "rmsprop"). Can be tuned.
#' @param validation_split A number between 0 and 1 for the proportion of data
#'   used for validation. Can be tuned.
#'
#' @param optimizer_args A named list of additional arguments passed to the
#'   optimizer. Cannot be tuned — pass via `set_engine()`.
#' @param loss A character string or a valid `{torch}` function for the loss function ("mse", "mae",
#'   "cross_entropy", "bce"). Cannot be tuned — pass via `set_engine()`.
#' @param architecture An [nn_arch()] object for a custom architecture. Cannot
#'   be tuned — pass via `set_engine()`.
#' @param flatten_input Logical or `NULL`. Controls input flattening. Cannot be
#'   tuned — pass via `set_engine()`.
#' @param early_stopping An [early_stop()] object or `NULL`. Cannot be tuned —
#'   pass via `set_engine()`.
#' @param device A character string for the device to use ("cpu", "cuda", "mps").
#'   If `NULL`, auto-detects the best available device. Cannot be tuned — pass
#'   via `set_engine()`.
#' @param verbose Logical for whether to print training progress. Default `FALSE`.
#'   Cannot be tuned — pass via `set_engine()`.
#' @param cache_weights Logical. If `TRUE`, stores trained weight matrices in
#'   the returned object. Cannot be tuned — pass via `set_engine()`.
#'
#' @details
#' This function creates a model specification for a neural network that can be
#' used within tidymodels workflows. The underlying engine is [train_nn()], which
#' is architecture-agnostic: when `architecture = NULL` it falls back to a
#' standard feed-forward network, but any architecture expressible via [nn_arch()]
#' can be used instead. The model supports:
#'
#' - Configurable hidden layers and activation functions (default MLP path)
#' - Custom architectures via [nn_arch()] (recurrent, convolutional, etc.)
#' - GPU acceleration (CUDA, MPS, or CPU)
#' - Hyperparameter tuning integration
#' - Both regression and classification tasks
#'
#' When using the default MLP path (no custom `architecture`), `hidden_neurons`
#' accepts an integer vector where each element represents the number of neurons
#' in that hidden layer. For example, `hidden_neurons = c(128, 64, 32)` creates
#' a network with three hidden layers. Pass an [nn_arch()] object via
#' `set_engine()` to use a custom architecture instead.
#'
#' The `device` parameter controls where computation occurs:
#' - `NULL` (default): Auto-detect best available device (CUDA > MPS > CPU)
#' - `"cuda"`: Use NVIDIA GPU
#' - `"mps"`: Use Apple Silicon GPU
#' - `"cpu"`: Use CPU only
#'
#' When tuning, you can use special tune tokens:
#' - For `hidden_neurons`: use `tune("hidden_neurons")` with a custom range
#' - For `activation`: use `tune("activation")` with values like "relu", "tanh"
#'
#' @return A model specification object with class `train_nnsnip`.
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'     box::use(
#'         recipes[recipe],
#'         workflows[workflow, add_recipe, add_model],
#'         tune[tune],
#'         parsnip[fit]
#'     )
#'
#'     # Model spec
#'     nn_spec = train_nnsnip(
#'         mode = "classification",
#'         hidden_neurons = c(30, 5),
#'         activations = c("relu", "elu"),
#'         epochs = 100
#'     )
#'
#'     wf = workflow() |>
#'         add_recipe(recipe(Species ~ ., data = iris)) |>
#'         add_model(nn_spec)
#'
#'     fit_wf = fit(wf, data = iris)
#' } else {
#'     message("Torch not fully installed — skipping example")
#' }
#' }
#'
#' @export
train_nnsnip = 
    function(
        mode = "unknown",
        engine = "kindling",
        # ---- Tunable parameters ----
        hidden_neurons = NULL,
        activations = NULL,
        output_activation = NULL,
        bias = NULL,
        epochs = NULL,
        batch_size = NULL,
        penalty = NULL,
        mixture = NULL,
        learn_rate = NULL,
        optimizer = NULL,
        validation_split = NULL,
        # ---- Non-tunable parameters (engine-level, set via set_engine()) ----
        optimizer_args = NULL,
        loss = NULL,
        architecture = NULL,
        flatten_input = NULL,
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
        optimizer_args = rlang::enquo(optimizer_args),
        loss = rlang::enquo(loss),
        architecture = rlang::enquo(architecture),
        flatten_input = rlang::enquo(flatten_input),
        early_stopping = rlang::enquo(early_stopping),
        device = rlang::enquo(device),
        verbose = rlang::enquo(verbose),
        cache_weights = rlang::enquo(cache_weights)
    )
    
    parsnip::new_model_spec(
        "train_nnsnip",
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
print.train_nnsnip = function(x, ...) {
    cat("Kindling Neural Network Model Specification (", x$mode, ")\n\n", sep = "")
    parsnip::model_printer(x, ...)
    invisible(x)
}


#' @export
#' @importFrom stats update
update.train_nnsnip = 
    function(
        object,
        parameters = NULL,
        hidden_neurons = NULL,
        activations = NULL,
        output_activation = NULL,
        bias = NULL,
        epochs = NULL,
        batch_size = NULL,
        penalty = NULL,
        mixture = NULL,
        learn_rate = NULL,
        optimizer = NULL,
        optimizer_args = NULL,
        loss = NULL,
        validation_split = NULL,
        device = NULL,
        verbose = NULL,
        fresh = FALSE,
        ...
    )
{
    
    args = list(
        hidden_neurons = rlang::enquo(hidden_neurons),
        activations = rlang::enquo(activations),
        output_activation = rlang::enquo(output_activation),
        bias = rlang::enquo(bias),
        epochs = rlang::enquo(epochs),
        batch_size = rlang::enquo(batch_size),
        penalty = rlang::enquo(penalty),
        mixture = rlang::enquo(mixture),
        learn_rate = rlang::enquo(learn_rate),
        optimizer = rlang::enquo(optimizer),
        optimizer_args = rlang::enquo(optimizer_args),
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
        cls = "train_nnsnip",
        ...
    )
}


#' @export
#' @importFrom parsnip translate
translate.train_nnsnip = function(x, engine = x$engine, ...) {
    if (is.null(engine)) {
        cli::cli_abort("Please set an engine with `set_engine()`.")
    }
    
    x = parsnip::translate.default(x, engine, ...)
    x
}


#' @export
#' @importFrom tune tunable
tunable.train_nnsnip = function(x, ...) {
    tibble::tibble(
        name = c(
            "hidden_neurons", "activations", "output_activation", "bias",
            "epochs", "batch_size", "penalty", "mixture",
            "learn_rate", "optimizer", "validation_split"
        ),
        call_info = list(
            list(pkg = "kindling", fun = "hidden_neurons"),
            list(pkg = "kindling", fun = "activations"),
            list(pkg = "kindling", fun = "output_activation"),
            list(pkg = "kindling", fun = "bias"),
            list(pkg = "dials", fun = "epochs"),
            list(pkg = "dials", fun = "batch_size"),
            list(pkg = "dials", fun = "penalty"),
            list(pkg = "dials", fun = "mixture"),
            list(pkg = "dials", fun = "learn_rate"),
            list(pkg = "kindling", fun = "optimizer"),
            list(pkg = "kindling", fun = "validation_split")
        ),
        source = "model_spec",
        component = "train_nnsnip",
        component_id = "main"
    )
}

#' kindling-tidymodels wrapper
#'
#' @param formula A formula specifying the model (e.g., `y ~ x1 + x2`)
#' @param data A data frame containing the training data
#' @param ... Additional arguments passed to the underlying training function
#'
#' @return
#' `train_nn_wrapper()` returns an `"nn_fit_tab"` object. See [train_nn()] for details.
#'
#' @details
#' This wrapper function is designed to interface with the `{tidymodels}`
#' ecosystem, particularly for use with [tune::tune_grid()] and workflows.
#' It handles the conversion of tuning parameters (especially list-column
#' parameters from [grid_depth()]) into the format expected by [train_nn()].
#'
#' @rdname kindling-nn-wrappers
#' @section MLP Wrapper for `{tidymodels}` interface:
#' Internal wrapper — use `mlp_kindling()` + `fit()` instead.
#'
#' @keywords internal
#' @export
train_nn_wrapper = function(formula, data, ...) {
    dots = list(...)
    dots = prepare_kindling_args(dots)
    
    do.call(
        train_nn,
        c(list(x = formula, data = data), dots)
    )
}
