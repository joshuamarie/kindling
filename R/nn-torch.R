#' Torch Neural Network Namespaces for Kindling
#'
#' `{kindling}` dynamically constructs two environments, `nn` and `nnf`,
#' by extracting all exported functions from the `{torch}` namespace that
#' begin with `nn_` and `nnf_`, respectively.
#'
#' These functions are then renamed by removing their `nn_` or `nnf_`
#' prefixes and optional trailing underscores, and stored as members of
#' the corresponding environment. This provides an interface similar in
#' spirit to PyTorch's `torch.nn` and `torch.nn.functional`, but programmatically
#' generated from `{torch}`'s own exports.
#'
#' @format
#' `nn`: An environment of class `torch_nn` containing torch neural network modules
#'
#' `nnf`: An environment of class `torch_nnf` containing torch functional neural network operations
#'
#' @section Implementation:
#' The environments `nn` and `nnf` are created at package build time.
#' Each environment contains callable modules and functional operators
#' directly referencing the corresponding `{torch}` objects.
#'
#' @examples
#' # Compose a simple feedforward model using nn
#' model = nn$sequential(
#'     nn$linear(10, 5),
#'     nn$relu(),
#'     nn$linear(5, 1)
#' )
#' model
#'
#' # Use functional operations from nnf
#' \dontrun{
#' x = torch::torch_randn(5, 10)
#' y = nnf$relu(x)
#' }
#'
#' @seealso [torch::nn_linear], [torch::nn_relu], [torch::nnf_relu]
#' @name nn
#' @aliases nnf
#' @export
nn = local ({
    funcs = stringr::str_subset(getNamespaceExports("torch"), "^nn_")
    objs = mget(funcs, envir = asNamespace("torch"))

    names = funcs |>
        stringr::str_remove("^nn_") |>
        stringr::str_remove("_$")

    nn = purrr::set_names(objs, names)
    list2env(nn, envir = environment())
    structure(as.environment(nn), class = "torch_nn")
})

#' @rdname nn
#' @export
nnf = local({
    funcs = stringr::str_subset(getNamespaceExports("torch"), "^nnf_")
    objs = mget(funcs, envir = asNamespace("torch"))

    names = funcs |>
        stringr::str_remove("^nnf_") |>
        stringr::str_remove("_$")

    nnf = purrr::set_names(objs, names)
    list2env(nnf, envir = environment())
    structure(as.environment(nnf), class = "torch_nnf")
})

#' @export
print.torch_nn = function(x, ...) {
    cli::cli_h1("Neural Network Namespace")
    n_members = length(ls(x, all.names = TRUE))
    cli::cli_text("{.val {n_members}} registered modules and layers available.")
    cli::cli_text("Use {.code nn$<name>} to access a specific module.")
    invisible(x)
}

#' @export
print.torch_nnf = function(x, ...) {
    cli::cli_h1("Functional Neural Network Namespace")
    n_members = length(ls(x, all.names = TRUE))
    cli::cli_text("{.val {n_members}} functional operations available.")
    cli::cli_text("Use {.code nnf$<name>} to call a function directly.")
    invisible(x)
}
