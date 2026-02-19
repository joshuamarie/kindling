#' Activation Functions Specification Helper
#'
#' This function is a DSL function, kind of like `ggplot2::aes()`, that helps to
#' specify activation functions for neural network layers. It validates that
#' activation functions exist in `torch` and that any parameters match the
#' function's formal arguments.
#'
#' @param ... Activation function specifications. Can be:
#' - Bare symbols: `relu`, `tanh`
#' - Character strings (simple): `"relu"`, `"tanh"`
#' - Character strings (with params): `"softshrink(lambda = 0.1)"`, `"rrelu(lower = 1/5, upper = 1/4)"`
#' - Named with parameters: `softmax = args(dim = 2L)`
#' - Indexed syntax (named): `softshrink[lambd = 0.2]`, `rrelu[lower = 1/5, upper = 1/4]`
#' - Indexed syntax (unnamed): `softshrink[0.5]`, `elu[0.5]`
#'
#' @return A `vctrs` vector with class "activation_spec" containing validated
#' activation specifications.
#'
#' @importFrom rlang enquos quo_get_expr is_call call_name as_string eval_tidy
#' @importFrom cli cli_abort
#' @importFrom vctrs new_vctr
#'
#' @export
act_funs = function(...) {
    dots = enquos(..., .ignore_empty = "all")
    mask = list(args = args)
    
    out = purrr::imap(dots, function(quo, name) {
        expr = quo_get_expr(quo)
        
        # ---- Indexed syntax ----
        # Ex. softshrink[lambd = 0.2] or softshrink[0.5]
        if (is_call(expr) && identical(call_name(expr), "[")) {
            calls = as.list(expr)
            act_name = as_string(calls[[2]])
            params = calls[-(1:2)]
            
            # Keep params as-is (named or unnamed)
            # Names will be NULL for unnamed params, which is fine
            validate_activation(act_name, prefix = "nnf_")
            if (!is.null(names(params)) && any(names(params) != "")) {
                validate_args_formals(act_name, params, prefix = "nnf_")
            }
            structure(params, act_name = act_name, class = "parameterized_activation")
            
        } else if (!is.null(name) && name != "") {
            # ---- Named parameter ---- 
            # This is a superseded behavior after v0.3.x, in favor of indexed syntax
            # Ex. softshrink = args(lambd = 0.5)
            validate_activation(name, prefix = "nnf_")
            extract_call = eval_tidy(quo, data = mask)
            
            if (inherits(extract_call, "activation_args")) {
                params = unclass(extract_call)
                validate_args_formals(name, params, prefix = "nnf_")
                structure(params, act_name = name, class = "parameterized_activation")
            } else if (is.character(extract_call) && length(extract_call) == 1 && extract_call == "") {
                structure(list(), act_name = name, class = "parameterized_activation")
            } else {
                cli_abort(c(
                    "Invalid syntax for parameterized activation '{name}'.",
                    i = "Use: {.code {name} = args(param = value)}."
                ), class = "activation_syntax_error")
            }
        } else if (is.symbol(expr)) {
            # ---- Bare symbol ----
            # Ex. relu, tanh
            act_name = as_string(expr)
            validate_activation(act_name, prefix = "nnf_")
            act_name
        } else if (is.character(expr)) {
            # ---- Character string ----
            # Note: Stringly typed expression is a bad practice, but tolerable
            # Ex. "relu" or "softshrink(lambd = 0.1)"
            parsed = parse_activation_string(expr)
            validate_activation(parsed$name, prefix = "nnf_")
            
            if (length(parsed$params) > 0) {
                validate_args_formals(parsed$name, parsed$params, prefix = "nnf_")
                structure(parsed$params, act_name = parsed$name, class = "parameterized_activation")
            } else {
                parsed$name
            }
        } else {
            cli_abort(c(
                "Invalid activation specification.",
                i = "Use bare names like {.code relu}, strings like {.code 'relu'},",
                i = "bracket syntax like {.code softshrink[lambd = 0.1]},",
                i = "parameterized strings like {.code 'softshrink(lambd = 0.1)'},",
                i = "or DSL like {.code softmax = args(dim = 2L)}."
            ), class = "activation_syntax_error")
        }
    })
    
    new_vctr(out, class = "activation_spec")
}

new_act_fn = function(x_expr) {
    
    structure(x_expr)
}

#' Activation Function Arguments Helper
#' 
#' @description
#' `r lifecycle::badge("superseded")`
#' 
#' This is superseded in v0.3.0 in favour of indexed syntax, e.g. `<act_fn[param = 0]>` type. 
#' Type-safe helper to specify parameters for activation functions.
#' All parameters must be named and match the formal arguments of the
#' corresponding `{torch}` activation function.
#'
#' @param ... Named arguments for the activation function.
#' @return A list with class "activation_args" containing the parameters.
#'
#' @importFrom cli cli_abort
#'
#' @export
args = function(...) {
    lifecycle::deprecate_soft(
        when = "0.3.0",
        what = "args()",
        details = "Use indexed syntax for parametric activation functions, e.g. `<softplus[beta = 0.5]>`."
    )
    
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

#' Parse activation function string with parameters
#'
#' Parse strings like "softshrink(lambda = 0.1)" or "rrelu(lower = 1/5, upper = 1/4)"
#' into activation name and parameter list.
#'
#' @param act_str Character string with activation and optional parameters.
#' @return List with `name` and `params` elements.
#'
#' @importFrom cli cli_abort
#'
#' @noRd
parse_activation_string = function(act_str) {
    if (!is.character(act_str) || length(act_str) != 1) {
        cli_abort("Activation string must be a single character value.")
    }
    
    # Check if it has parameters (contains parentheses)
    if (!grepl("\\(", act_str)) {
        return(list(name = act_str, params = list()))
    }
    
    match = regexec("^([a-z_0-9]+)\\((.*)\\)$", act_str, ignore.case = TRUE)
    matches = regmatches(act_str, match)[[1]]
    
    if (length(matches) != 3) {
        cli_abort(c(
            "Invalid activation string format: {.val {act_str}}",
            i = "Expected format: {.code 'function_name(param1 = value1, param2 = value2)'}"
        ))
    }
    
    act_name = matches[2]
    params_str = matches[3]
    
    if (nchar(trimws(params_str)) == 0) {
        params_list = list()
    } else {
        expr_str = paste0("list(", params_str, ")")
        
        tryCatch({
            params_list = eval(parse(text = expr_str))
        }, error = function(e) {
            cli_abort(c(
                "Failed to parse parameters in: {.val {act_str}}",
                i = "Error: {e$message}",
                i = "Parameters must be valid R expressions."
            ))
        })
    }
    
    list(name = act_name, params = params_list)
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
        # Only check named parameters
        if (!is.null(param_names)) {
            named_params = param_names[param_names != ""]
            invalid_params = setdiff(named_params, fn_formals)
            
            if (length(invalid_params) > 0) {
                valid_params_str = paste(fn_formals, collapse = ", ")
                cli_abort(c(
                    "Invalid parameter{?s} for {.fn {fn_name}}: {.arg {invalid_params}}.",
                    i = "Valid parameters are: {.code {valid_params_str}}."
                ), class = "activation_invalid_params_error")
            }
        }
        
        invisible(NULL)
    }
}

#' Activation Function Specifications Parser
#'
#' This function parses activation function specifications provided in various formats
#'
#' @param activations Activation function specifications. Can be:
#' - `NULL`: No activation functions.
#' - Character vector: e.g., `c("relu", "tanh")` or `c("relu", "softshrink(lambda = 0.1)")`.
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
        
        parsed_activation = lapply(activations, function(act_str) {
            parsed = parse_activation_string(act_str)
            list(name = parsed$name, param = parsed$params)
        })
        
        set_names(transpose(parsed_activation), c("names", "params"))
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
                elem_name = if (!is.null(names(activations))) names(activations)[i] else ""
                
                if (elem_name != "" && is.list(elem)) {
                    list(name = elem_name, param = elem)
                    
                } else if (elem_name == "" && is.character(elem) && length(elem) == 1L) {
                    parsed = parse_activation_string(elem)
                    list(name = parsed$name, param = parsed$params)
                    
                } else if (elem_name == "" && is.name(elem)) {
                    list(name = as.character(elem), param = list())
                    
                } else if (elem_name != "" && is.character(elem) && elem == "") {
                    list(name = elem_name, param = list())
                    
                } else {
                    cli_abort(c(
                        "Invalid activation specification at position {i}.",
                        i = "Use: {.code relu} or {.code 'relu'} for simple activations",
                        i = "Use: {.code 'softshrink(lambda = 0.1)'} for parameterized strings",
                        i = "Use: {.code softmax = args(dim = 2L)} for DSL"
                    ), class = "activation_syntax_error")
                }
            })
        
        set_names(transpose(parsed_activation), c("names", "params"))
    } else {
        cli_abort(c(
            "{.arg activations} must be character vector, list, or {.code act_funs()}.",
            i = "Examples:",
            "*" = "Character: {.code c('relu', 'tanh')}",
            "*" = "Parameterized: {.code c('relu', 'softshrink(lambda = 0.1)')}",
            "*" = "Bracket syntax: {.code act_funs(relu, softshrink[lambd = 0.2])}",
            "*" = "List: {.code list(relu, tanh, softmax = args(dim = 2L))}",
            "*" = "DSL: {.code act_funs(relu, tanh, softmax = args(dim = 2L))}"
        ), class = "activation_type_error")
    }
}

#' Activation Functions Processor
#'
#' This function processes activation function specifications into callable
#' expressions with proper torch namespace handling.
#'
#' @param activation_spec A list with two elements:
#' - `names`: Character vector of activation function names.
#' - `params`: List of parameter lists for each activation function.
#' @param prefix Prefix for activation function names (default: "nnf_").
#'
#' @return A list of functions that generate activation function calls with
#' proper torch:: namespace prefixing.
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
            fn_name = paste0(prefix, name)
            fn_call = call2("::", sym("torch"), sym(fn_name))
            
            if (is.null(params) || length(params) == 0) {
                function(x_expr) {
                    # rlang::call_standardise(call2(fn_call, x_expr))
                    call2(fn_call, x_expr)
                }
            } else {
                function(x_expr) {
                    # rlang::call_standardise(exec(call2, fn_call, x_expr, !!!params))
                    exec(call2, fn_call, x_expr, !!!params)
                }
            }
        }
    })
}

#' Activation Function Specs Evaluation
#' 
#' Helper function for `act_funcs()` argument.
#' 
#' @param activations Quosure containing the activations expression
#' @param output_activation Quosure containing the output_activation expression
#' 
#' @return A list with two elements: `activations` and `output_activation`
#' 
#' @importFrom rlang enquo quo_is_null eval_tidy
#' @keywords internal
eval_act_funs = function(activations, output_activation) {
    activations_quo = enquo(activations)
    output_activation_quo = enquo(output_activation)
    
    env_mask = list(
        act_funs = act_funs,
        args = args
    )
    
    activations = if (!quo_is_null(activations_quo)) {
        eval_tidy(activations_quo, data = env_mask)
    } else {
        NULL
    }
    
    output_activation = if (!quo_is_null(output_activation_quo)) {
        eval_tidy(output_activation_quo, data = env_mask)
    } else {
        NULL
    }
    
    list(
        activations = activations,
        output_activation = output_activation
    )
}
