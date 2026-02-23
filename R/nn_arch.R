#' Architecture specification for train_nn()
#'
#' @description
#' `nn_arch()` is a helper that bundles `nn_module_generator()` arguments into a
#' single object passed to `train_nn()` via the `arch` parameter. All arguments
#' mirror those of `nn_module_generator()` exactly, including their defaults.
#'
#' @param nn_name Character. Name of the generated module class. Default `"nnModule"`.
#' @param nn_layer Layer type for hidden layers. Accepted forms:
#'   - `NULL` (default): uses `nn_linear` from `{torch}`
#'   - Character string: e.g. `"torch::nn_gru"`, `"my_custom_layer"`
#'   - Formula: e.g. `~ rbf_layer`, `~ torch::nn_linear` (the RHS is used as the constructor)
#'   - Function object: a layer constructor function
#' @param out_nn_layer Optional. Layer type forced on the output layer. Accepts the same forms
#'   as `nn_layer`. Default `NULL` (uses the same layer as `nn_layer`).
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
#' @param input_transform Formula or function. Applied to each input tensor before it is
#'   passed to the model — both during training (per-batch) and prediction. Use this for
#'   structural reshaping that must happen every forward pass, such as adding a sequence
#'   dimension for RNNs (`~ .$unsqueeze(2)`) or a channel dimension for CNNs
#'   (`~ .$unsqueeze(1)`).
#'
#' @note The `nn_arch` object captures the caller environment at construction time so that
#'   user-defined layer constructors (e.g. a custom `rbf_layer`) remain accessible when the
#'   model is built. This means saving the `nn_arch` or the resulting `nn_fit` object to disk
#'   with `saveRDS()` will embed that environment, which can produce large files and will
#'   **not** restore the layer constructor when reloaded in a fresh session. If you plan to
#'   serialize fitted models, ensure the custom layer is defined in a package or sourced
#'   before calling `predict()`.
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
        
        .check_nn_args(nn_layer, is_layer_spec, "nn_layer", "must be a character string, formula (e.g. `~ rbf_layer`), or function")
        .check_nn_args(out_nn_layer, is_layer_spec, "out_nn_layer", "must be a character string, formula (e.g. `~ torch::nn_linear`), or function")
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
    fmt_entry = function(val) {
        if (is.null(val)) return("none")
        if (rlang::is_formula(val)) return(deparse(val))
        if (rlang::is_function(val)) return("<function>")
        if (is.list(val) && length(val) == 0L) return("(empty)")
        if (is.list(val)) return(paste0("list(", paste(names(val), collapse = ", "), ")"))
        as.character(val)
    }
    
    cli::cli_h3("Neural Network Architecture Spec")
    cli::cli_bullets(c(
        "*" = "Name:                   {x$nn_name}",
        "*" = "Layer:                  {x$nn_layer %||% 'nn_linear (default)'}",
        "*" = "Out layer:              {x$out_nn_layer %||% 'same as nn_layer'}",
        "*" = "Layer args:             {fmt_entry(x$nn_layer_args)}",
        "*" = "Layer arg fn:           {fmt_entry(x$layer_arg_fn)}",
        "*" = "Last layer args:        {fmt_entry(x$last_layer_args)}",
        "*" = "Forward extract:        {fmt_entry(x$forward_extract)}",
        "*" = "Before output transform:{fmt_entry(x$before_output_transform)}",
        "*" = "After output transform: {fmt_entry(x$after_output_transform)}",
        "*" = "Input transform:        {fmt_entry(x$input_transform)}"
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

is_layer_spec = function(x) {
    is.character(x) || rlang::is_formula(x) || rlang::is_function(x)
}
