#' Activation Function Arguments Helper
#'
#' Type-safe helper to specify parameters for activation functions.
#' All parameters must be named and match the formal arguments of the
#' corresponding `torch` activation function.
#'
#' @param ... Named arguments for the activation function.
#' @return A list with class "activation_args" containing the parameters.
#'
#' @importFrom cli cli_abort
#'
#' @export
args = function(...) {
    params = list(...)

    if (length(params) == 0) {
        structure(list(), class = "activation_args")
    } else {
        param_names = names(params)

        if (is.null(param_names) || any(param_names == "")) {
            cli_abort(c(
                "{.fn args} requires all arguments to be named.",
                i = "Use named arguments like {.code args(dim = 2L)}."
            ), class = "activation_args_error")
        }

        structure(params, class = "activation_args")
    }
}

#' Validate Activation Function Exists
#'
#' @param act_name Character. Activation function name (without prefix).
#' @param prefix Character. Prefix for the function name (default: "nnf_").
#'
#' @importFrom cli cli_abort
#' @importFrom rlang is_installed
#'
#' @noRd
validate_activation = function(act_name, prefix = "nnf_") {
    if (!is_installed("torch")) {
        cli_abort(c(
            "{.pkg torch} package is required but not installed.",
            i = "Install it with: {.code install.packages('torch')}"
        ), class = "torch_missing_error")
    }

    fn_name = paste0(prefix, act_name)

    if (!exists(fn_name, where = asNamespace("torch"), mode = "function")) {
        cli_abort(c(
            "Activation function {.fn {fn_name}} does not exist in {.pkg torch}.",
            i = "Available functions start with {.code {prefix}}.",
            i = "Check {.code ?torch::nnf_relu} for examples."
        ), class = "activation_not_found_error")
    }

    fn_name
}

#' Validate Arguments Match Function Formals
#'
#' @param act_name Character. Activation function name (without prefix).
#' @param params List. Parameters to validate.
#' @param prefix Character. Prefix for the function name (default: "nnf_").
#'
#' @importFrom cli cli_abort
#'
#' @noRd
validate_args_formals = function(act_name, params, prefix = "nnf_") {
    fn_name = paste0(prefix, act_name)

    if (!requireNamespace("torch", quietly = TRUE)) {
        invisible(NULL)
    } else {
        fn = get(fn_name, envir = asNamespace("torch"))
        fn_formals = names(formals(fn))
        fn_formals = fn_formals[!fn_formals %in% c("input", "x", "...")]

        param_names = names(params)
        invalid_params = setdiff(param_names, fn_formals)

        if (length(invalid_params) > 0) {
            valid_params_str = paste(fn_formals, collapse = ", ")
            cli_abort(c(
                "Invalid parameter{?s} for {.fn {fn_name}}: {.arg {invalid_params}}.",
                i = "Valid parameters are: {.code {valid_params_str}}."
            ), class = "activation_invalid_params_error")
        }

        invisible(NULL)
    }
}

#' Activation Functions Specification Helper
#'
#' This function is a DSL function, kind of like `ggplot2::aes()`, that helps to
#' specify activation functions for neural network layers. It validates that
#' activation functions exist in `torch` and that any parameters match the
#' function's formal arguments.
#'
#' @param ... Activation function specifications. Can be:
#' - Bare symbols: `relu`, `tanh`
#' - Character strings: `"relu"`, `"tanh"`
#' - Named with parameters: `softmax = args(dim = 2L)`
#'
#' @return A `vctrs` vector with class "activation_spec" containing validated
#' activation specifications.
#'
#' @importFrom rlang enquos quo_get_expr is_call call_name as_string
#' @importFrom cli cli_abort
#' @importFrom vctrs new_vctr
#'
#' @export
act_funs = function(...) {
    dots = enquos(...)

    out = lapply(seq_along(dots), function(i) {
        quo = dots[[i]]
        expr = quo_get_expr(quo)
        name = names(dots)[i]

        if (!is.null(name) && name != "") {
            validate_activation(name, prefix = "nnf_")

            if (is_call(expr) && call_name(expr) == "args") {
                params = as.list(expr)[-1]
                validate_args_formals(name, params, prefix = "nnf_")
                structure(params, act_name = name, class = "parameterized_activation")
            } else if (is.character(expr) && expr == "") {
                structure(list(), act_name = name, class = "parameterized_activation")
            } else {
                cli_abort(c(
                    "Invalid syntax for parameterized activation at position {i}.",
                    i = "Use: {.code {name} = args(param = value)}."
                ), class = "activation_syntax_error")
            }
        } else if (is.symbol(expr)) {
            act_name = as_string(expr)
            validate_activation(act_name, prefix = "nnf_")
            act_name
        } else if (is.character(expr)) {
            validate_activation(expr, prefix = "nnf_")
            expr
        } else {
            cli_abort(c(
                "Invalid activation specification at position {i}.",
                i = "Use bare names like {.code relu}, strings like {.code 'relu'},",
                i = "or parameterized like {.code softmax = args(dim = 2L)}."
            ), class = "activation_syntax_error")
        }
    })

    new_vctr(out, class = "activation_spec")
}

#' Activation Function Specifications Parser
#'
#' This function parses activation function specifications provided in various formats
#'
#' @param activations Activation function specifications. Can be:
#' - `NULL`: No activation functions.
#' - Character vector: e.g., `c("relu", "tanh")`.
#' - List: e.g., `list(relu, tanh, softmax = args(dim = 2L))`.
#' - `activation_spec` object from `act_funs()`.
#' @param n_layers Number of layers to apply activations to.
#'
#' @return A list with two elements:
#' - `names`: Character vector of activation function names.
#' - `params`: List of parameter lists for each activation function.
#'
#' @importFrom cli cli_abort
#' @importFrom rlang is_call call_name as_string
#' @importFrom purrr imap set_names transpose
#'
#' @noRd
parse_activation_spec = function(activations, n_layers) {
    if (is.null(activations)) {
        list(
            names = rep(NA_character_, n_layers),
            params = vector("list", n_layers)
        )
    } else if (inherits(activations, "activation_spec")) {
        if (length(activations) == 1L) {
            activations = rep(activations, n_layers)
        }

        if (length(activations) != n_layers) {
            cli_abort(c(
                "{.arg activations} must specify 1 or {n_layers} activation(s).",
                x = "You provided {length(activations)}."
            ), class = "activation_spec_length_error")
        }

        parsed_activation =
            imap(activations, function(elem, i) {

            if (inherits(elem, "parameterized_activation")) {
                params = unclass(elem)
                attr(params, "act_name") = NULL

                list(
                    name = attr(elem, "act_name"),
                    param = params
                )
            } else if (is.character(elem)) {
                list(
                    name = elem,
                    param = list()
                )
            } else {
                cli_abort(paste("Unsupported activation type at index", i))
            }
        })

        set_names(transpose(parsed_activation), c("names", "params"))
    } else if (is.character(activations)) {
        if (length(activations) == 1L) {
            activations = rep(activations, n_layers)
        }

        if (length(activations) != n_layers) {
            cli_abort(c(
                "{.arg activations} must be length 1 or {n_layers}.",
                x = "You provided length {length(activations)}."
            ), class = "activation_spec_length_error")
        }

        list(
            names = activations,
            params = vector("list", n_layers)
        )
    } else if (is.list(activations)) {
        if (length(activations) == 1L) {
            activations = rep(activations, n_layers)
        }

        if (length(activations) != n_layers) {
            cli_abort(c(
                "{.arg activations} must be length 1 or {n_layers}.",
                x = "You provided length {length(activations)}."
            ), class = "activation_spec_length_error")
        }

        parsed_activation =
            imap(activations, function(elem, i) {
                elem_name = if (!is.null(act_names)) act_names[i] else ""

                if (elem_name != "" && is.list(elem)) {
                    list(name = elem_name, param = elem)

                } else if (elem_name == "" && is.character(elem) && length(elem) == 1L) {
                    list(name = elem, param = list())

                } else if (elem_name == "" && is.name(elem)) {
                    list(name = as.character(elem), param = list())

                } else if (elem_name != "" && is.character(elem) && elem == "") {
                    list(name = elem_name, param = list())

                } else {
                    cli_abort(c(
                        "Invalid activation specification at position {i}.",
                        i = "Use: {.code relu} or {.code 'relu'} for simple activations",
                        i = "Use: {.code softmax = args(dim = 2L)} for parameterized"
                    ), class = "activation_syntax_error")
                }
            })

        set_names(transpose(parsed_activation), c("names", "params"))
    } else {
        cli_abort(c(
            "{.arg activations} must be character vector, list, or {.code act_funs()}.",
            i = "Examples:",
            "*" = "Character: {.code c('relu', 'tanh')}",
            "*" = "List: {.code list(relu, tanh, softmax = args(dim = 2L))}",
            "*" = "DSL: {.code act_funs(relu, tanh, softmax = args(dim = 2L))}"
        ), class = "activation_type_error")
    }
}

#' Activation Functions Processor
#'
#' This function processes activation function specifications into callable
#' expressions.
#'
#' @param activation_spec A list with two elements:
#' - `names`: Character vector of activation function names.
#' - `params`: List of parameter lists for each activation function.
#' @param prefix Prefix for activation function names (default: "nnf_").
#'
#' @return A list of functions that generate activation function calls.
#'
#' @importFrom rlang call2 sym exec
#' @importFrom purrr map2
#'
#' @noRd
process_activations = function(activation_spec, prefix = "nnf_") {
    act_names = activation_spec$names
    act_params = activation_spec$params

    map2(act_names, act_params, function(name, params) {
        if (is.na(name)) {
            NULL
        } else {
            fn_sym = sym(paste0(prefix, name))

            if (is.null(params) || length(params) == 0) {
                function(x_expr) call2(fn_sym, x_expr)
            } else {
                function(x_expr) exec(call2, fn_sym, x_expr, !!!params)
            }
        }
    })
}
