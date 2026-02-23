#' Architecture specification for train_nn()
#'
#' @description
#' `nn_arch()` defines an architecture specification object consumed by
#' [train_nn()] via `architecture` (or legacy `arch`).
#'
#' Conceptual workflow:
#' 1. Define architecture with `nn_arch()`.
#' 2. Train with `train_nn(..., architecture = <nn_arch>)`.
#' 3. Predict with `predict()`.
#'
#' Architecture fields mirror `nn_module_generator()` and are passed through
#' without additional branching logic.
#'
#' @param nn_name Character. Name of the generated module class. Default `"nnModule"`.
#' @param nn_layer Layer type. See `nn_module_generator()`. Default `NULL` (`nn_linear`).
#' @param out_nn_layer Optional. Layer type forced on the last layer. Default `NULL`.
#' @param nn_layer_args Named list. Additional arguments passed to every layer constructor.
#'   Default `list()`.
#' @param layer_arg_fn Formula or function. Generates per-layer constructor arguments.
#'   Default `NULL` (FFNN-style: `list(in_dim, out_dim, bias = bias)`).
#' @param forward_extract Formula or function. Processes layer output in the forward pass.
#'   Default `NULL`.
#' @param before_output_transform Formula or function. Transforms input before the output
#'   layer. Default `NULL`.
#' @param after_output_transform Formula or function. Transforms output after the output
#'   layer. Default `NULL`.
#' @param last_layer_args Named list or formula. Extra arguments for the output layer only.
#'   Default `list()`.
#' @param input_transform Formula or function. Transforms the entire input tensor
#'   before training begins. Applied once to the full dataset tensor, not per-batch.
#'   Transforms must therefore be independent of batch size. Safe examples:
#'   `~ .$unsqueeze(2)` (RNN sequence dim), `~ .$unsqueeze(1)` (CNN channel dim).
#'   Avoid transforms that reshape based on `.$size(1)` as this will reflect the
#'   full dataset size, not the mini-batch size.
#' @param use_namespace Logical. If `TRUE` (default), layer symbols are resolved
#'   with explicit `torch::` namespace when applicable. Set to `FALSE` when using
#'   custom layers defined in the calling environment.
#'
#' @return An object of class `c("nn_arch", "kindling_arch")`, implemented as a
#'   named list of `nn_module_generator()` arguments with an `"env"` attribute
#'   capturing the calling environment for custom layer resolution.
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'     # Standard architecture object for train_nn()
#'     std_arch = nn_arch(nn_name = "mlp_model")
#'
#'     # GRU architecture spec
#'     gru_arch = nn_arch(
#'         nn_name = "GRU",
#'         nn_layer = "torch::nn_gru",
#'         layer_arg_fn = ~ if (.is_output) {
#'             list(.in, .out)
#'         } else {
#'             list(input_size = .in, hidden_size = .out, batch_first = TRUE)
#'         },
#'         out_nn_layer = "torch::nn_linear",
#'         forward_extract = ~ .[[1]],
#'         before_output_transform = ~ .[, .$size(2), ],
#'         input_transform = ~ .$unsqueeze(2)
#'     )
#'
#'     # Custom layer architecture (resolved from calling environment)
#'     custom_linear = torch::nn_module(
#'         "CustomLinear",
#'         initialize = function(in_features, out_features, bias = TRUE) {
#'             self$layer = torch::nn_linear(in_features, out_features, bias = bias)
#'         },
#'         forward = function(x) self$layer(x)
#'     )
#'     custom_arch = nn_arch(
#'         nn_name = "CustomMLP",
#'         nn_layer = "custom_linear",
#'         use_namespace = FALSE
#'     )
#'
#'     model = train_nn(
#'         Sepal.Length ~ .,
#'         data = iris[, 1:4],
#'         hidden_neurons = c(64, 32),
#'         activations = "relu",
#'         epochs = 50,
#'         architecture = gru_arch
#'     )
#' }
#' }
#'
#' @export
nn_arch = function(
        nn_name = "nnModule",
        nn_layer = NULL,
        out_nn_layer = NULL,
        nn_layer_args = list(),
        layer_arg_fn = NULL,
        forward_extract = NULL,
        before_output_transform = NULL,
        after_output_transform = NULL,
        last_layer_args = list(),
        input_transform = NULL,
        use_namespace = TRUE
) {
    struc = structure(
        list(
            nn_name = nn_name,
            nn_layer = nn_layer,
            out_nn_layer = out_nn_layer,
            nn_layer_args = nn_layer_args,
            layer_arg_fn = layer_arg_fn,
            forward_extract = forward_extract,
            before_output_transform = before_output_transform,
            after_output_transform = after_output_transform,
            last_layer_args = last_layer_args,
            input_transform = input_transform,
            use_namespace = use_namespace
        ),
        class = c("nn_arch", "kindling_arch")
    )
    attr(struc, "env") = rlang::caller_env()
    struc
}

#' Display `nn_arch()` configuration
#'
#' @param x An object of class "nn_arch"
#' @param ... Additional arguments (unused)
#'
#' @return No return value, just the information
#'
#' @keywords internal
#' @export
print.nn_arch = function(x, ...) {
    cli::cli_h3("Neural Network Architecture Spec")
    cli::cli_bullets(c(
        "*" = "Name:            {x$nn_name}",
        "*" = "Layer:           {x$nn_layer %||% 'nn_linear (default)'}",
        "*" = "Out layer:       {x$out_nn_layer %||% 'same as nn_layer'}",
        "*" = "Input transform: {if (is.null(x$input_transform)) 'none' else 'yes'}",
        "*" = "Namespace:       {x$use_namespace}"
    ))
    invisible(x)
}
