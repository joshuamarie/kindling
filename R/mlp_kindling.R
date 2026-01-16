#' Multi-Layer Perceptron (Feedforward Neural Network) via kindling
#'
#' `mlp_kindling()` defines a feedforward neural network model that can be used
#' for classification or regression. It integrates with the tidymodels ecosystem
#' and uses the torch backend via kindling.
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
#' @param batch_size An integer for the batch size during training. Can be tuned.
#' @param learn_rate A number for the learning rate. Can be tuned.
#' @param optimizer A character string for the optimizer type ("adam", "sgd",
#'   "rmsprop"). Can be tuned.
#' @param loss A character string for the loss function ("mse", "mae",
#'   "cross_entropy", "bce"). Can be tuned.
#' @param validation_split A number between 0 and 1 for the proportion of data
#'   used for validation. Can be tuned.
#' @param device A character string for the device to use ("cpu", "cuda", "mps").
#'   If NULL, auto-detects available GPU. Can be tuned.
#' @param verbose Logical for whether to print training progress. Default FALSE.
#'
#' @details
#' This function creates a model specification for a feedforward neural network
#' that can be used within tidymodels workflows. The model supports:
#'
#' - Multiple hidden layers with configurable units
#' - Various activation functions per layer
#' - GPU acceleration (CUDA, MPS, or CPU)
#' - Hyperparameter tuning integration
#' - Both regression and classification tasks
#'
#' The `hidden_neurons` parameter accepts an integer vector where each element
#' represents the number of neurons in that hidden layer. For example,
#' `hidden_neurons = c(128, 64, 32)` creates a network with three hidden layers.
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
#' - For `device`: use `tune("device")` to compare CPU vs GPU performance
#'
#' @return A model specification object with class `mlp_kindling`.
#'
#' @examples
#' \dontrun{
#' if (torch::torch_is_installed()) {
#'     box::use(
#'         recipes[recipe],
#'         workflows[workflow, add_recipe, add_model],
#'         tune[tune],
#'         parsnip[fit]
#'     )
#'
#'     # Model specs
#'     mlp_spec = mlp_kindling(
#'         mode = "classification",
#'         hidden_neurons = c(128, 64, 32),
#'         activation = c("relu", "relu", "relu"),
#'         epochs = 100
#'     )
#'
#'     # If you want to tune
#'     mlp_tune_spec = mlp_kindling(
#'         mode = "classification",
#'         hidden_neurons = tune(),
#'         activation = tune(),
#'         epochs = tune(),
#'         learn_rate = tune()
#'     )
#'      wf = workflow() |>
#'         add_recipe(recipe(Species ~ ., data = iris)) |>
#'         add_model(mlp_spec)
#'
#'      fit_wf = fit(wf, data = iris)
#' } else {
#'     message("Torch not fully installed — skipping example")
#' }
#' }
#'
#' @export
mlp_kindling =
    function(
        mode = "unknown",
        engine = "kindling",
        hidden_neurons = NULL,
        activations = NULL,
        output_activation = NULL,
        bias = NULL,
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
        activations = rlang::enquo(activations),
        output_activation = rlang::enquo(output_activation),
        bias = rlang::enquo(bias),
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
        "mlp_kindling",
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
print.mlp_kindling = function(x, ...) {
    cat("Kindling Multi-Layer Perceptron Model Specification (", x$mode, ")\n\n", sep = "")
    parsnip::model_printer(x, ...)
    invisible(x)
}

#' @export
#' @importFrom stats update
update.mlp_kindling =
    function(
        object,
        parameters = NULL,
        hidden_neurons = NULL,
        activations = NULL,
        output_activation = NULL,
        bias = NULL,
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
        activations = rlang::enquo(activations),
        output_activation = rlang::enquo(output_activation),
        bias = rlang::enquo(bias),
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
        cls = "mlp_kindling",
        ...
    )
}

#' @export
#' @importFrom parsnip translate
translate.mlp_kindling = function(x, engine = x$engine, ...) {
    if (is.null(engine)) {
        cli::cli_abort("Please set an engine with `set_engine()`.")
    }

    x = parsnip::translate.default(x, engine, ...)
    x
}

#' @export
#' @importFrom tune tunable
tunable.mlp_kindling = function(x, ...) {
    tibble::tibble(
        name = c(
            "hidden_neurons", "activations", "output_activation", "bias",
            "epochs", "batch_size", "learn_rate", "optimizer",
            "loss", "validation_split"
        ),
        call_info = list(
            list(pkg = "kindling", fun = "hidden_neurons"),
            list(pkg = "kindling", fun = "activations"),
            list(pkg = "kindling", fun = "output_activation"),
            list(pkg = "kindling", fun = "bias"),
            list(pkg = "dials", fun = "epochs"),
            list(pkg = "dials", fun = "batch_size"),
            list(pkg = "dials", fun = "learn_rate"),
            list(pkg = "kindling", fun = "optimizer"),
            list(pkg = "dials", fun = "loss"),
            list(pkg = "kindling", fun = "validation_split")
        ),
        source = "model_spec",
        component = "mlp_kindling",
        component_id = "main"
    )
}
