#' Architecture specification for train_nn()
#'
#' @description
#' `nn_arch()` is a helper that bundles `nn_module_generator()` arguments into a
#' single object passed to `train_nn()` via the `arch` parameter. All arguments
#' mirror those of `nn_module_generator()` exactly, including their defaults.
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
#'
#' @return An object of class `"nn_arch"`, a named list of `nn_module_generator()` arguments.
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
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
#'     model = train_nn(
#'         Sepal.Length ~ .,
#'         data = iris[, 1:4],
#'         hidden_neurons = c(64, 32),
#'         activations = "relu",
#'         epochs = 50,
#'         arch = gru_arch
#'     )
#' }
#' }
#'
#' @export
nn_arch = 
    function(
        nn_name = "nnModule",
        nn_layer = NULL,
        out_nn_layer = NULL,
        nn_layer_args = list(),
        layer_arg_fn = NULL,
        forward_extract = NULL,
        before_output_transform = NULL,
        after_output_transform = NULL,
        last_layer_args = list(),
        input_transform = NULL
    ) {
    
    .check_nn_args(nn_layer, is.character, "nn_layer", "must be a character string")
    .check_nn_args(out_nn_layer, is.character, "out_nn_layer", "must be a character string")
    .check_nn_args(layer_arg_fn, is_fn_or_formula, "layer_arg_fn", "must be a function or formula")
    .check_nn_args(forward_extract, is_fn_or_formula, "forward_extract", "must be a function or formula")
    .check_nn_args(before_output_transform, is_fn_or_formula, "before_output_transform", "must be a function or formula")
    .check_nn_args(after_output_transform, is_fn_or_formula, "after_output_transform", "must be a function or formula")
    .check_nn_args(input_transform, is_fn_or_formula, "input_transform", "must be a function or formula")
    
    if (!rlang::is_list(nn_layer_args)) {
        cli::cli_abort("{.arg `nn_layer_args`} must be a list.")
    }
    if (!rlang::is_list(last_layer_args) && !rlang::is_formula(last_layer_args)) {
        cli::cli_abort("{.arg `last_layer_args`} must be a list or formula.")
    }
    
    struc = vctrs::new_vctr(
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
            input_transform = input_transform
        ),
        class = "nn_arch"
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
        "*" = "Input transform: {if (is.null(x$input_transform)) 'none' else 'yes'}"
    ))
    invisible(x)
}

.check_nn_args = function(x, pred, arg, msg) {
    if (!is.null(x) && !pred(x)) {
        cli::cli_abort("{.arg {arg}} {msg}.")
    }
}

is_fn_or_formula = function(x) {
    rlang::is_function(x) || rlang::is_formula(x)
}
