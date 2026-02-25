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
#' @param loss A character string for the loss function ("mse", "mae",
#'   "cross_entropy", "bce"). Cannot be tuned — pass via `set_engine()`.
#' @param architecture An [nn_arch()] object for a custom architecture. Cannot
#'   be tuned — pass via `set_engine()`.
#' @param flatten_input Logical or `NULL`. Controls input flattening. Cannot be
#'   tuned — pass via `set_engine()`.
#' @param early_stopping An [early_stop()] object or `NULL`. Cannot be tuned —
#'   pass via `set_engine()`.
#' @param device A character string for the device ("cpu", "cuda", "mps").
#'   Cannot be tuned — pass via `set_engine()`.
#' @param verbose Logical for whether to print training progress. Cannot be
#'   tuned — pass via `set_engine()`.
#' @param cache_weights Logical. If `TRUE`, stores trained weight matrices in
#'   the returned object. Cannot be tuned — pass via `set_engine()`.
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
#' Parameters that cannot be tuned (`architecture`, `flatten_input`,
#' `early_stopping`, `device`, `verbose`, `cache_weights`, `optimizer_args`,
#' `loss`) must be set via `set_engine()`, not as arguments to `mlp_kindling()`.
#'
#' @return A model specification object with class `mlp_kindling`.
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
#'     # library(recipes)
#'     # library(workflows)
#'     # library(parsnip)
#'     # library(tune)
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
        "mlp_kindling",
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
        penalty = NULL,
        mixture = NULL,
        learn_rate = NULL,
        optimizer = NULL,
        validation_split = NULL,
        optimizer_args = NULL,
        loss = NULL,
        architecture = NULL,
        flatten_input = NULL,
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
        component = "mlp_kindling",
        component_id = "main"
    )
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
#' @details
#' These wrapper functions are designed to interface with the `{tidymodels}` 
#' ecosystem, particularly for use with [tune::tune_grid()] and workflows.
#' They handle the conversion of tuning parameters (especially list-column 
#' parameters from [grid_depth()]) into the format expected by [ffnn()] and [rnn()].
#' 
#' @rdname kindling-nn-wrappers
#' @section FFNN (MLP) Wrapper for `{tidymodels}` interface:
#' This is a function to interface into `{tidymodels}`
#' (do not use this, use `kindling::ffnn()` instead).
#' 
#' @keywords internal
#' @export
ffnn_wrapper = function(formula, data, ...) {
    dots = list(...)
    dots = prepare_kindling_args(dots)
    
    do.call(
        ffnn,
        c(list(formula = formula, data = data), dots)
    )
}
