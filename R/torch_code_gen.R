#' Generalized Neural Network Module Expression Generator
#' 
#' @description
#' `r lifecycle::badge("experimental")`
#' 
#' `nn_module_generator()` is a generalized function that generates neural network 
#' module expressions for various architectures. It provides a flexible framework for creating
#' custom neural network modules by parameterizing layer types, construction arguments, and
#' forward pass behavior.
#'
#' While designed primarily for `{torch}` modules, it can work with custom layer implementations
#' from the current environment, including user-defined layers like RBF networks, custom
#' attention mechanisms, or other novel architectures.
#'
#' This function serves as the foundation for specialized generators like `ffnn_generator()`
#' and `rnn_generator()`, but can be used directly to create custom architectures.
#'
#' @param nn_name Character string specifying the name of the generated neural network module class.
#'   Default is `"nnModule"`.
#'
#' @param nn_layer The type of neural network layer to use. Can be specified as:
#'   - `NULL` (default): Uses `nn_linear()` from `{torch}`
#'   - Character string: e.g., `"nn_linear"`, `"nn_gru"`, `"nn_lstm"`, `"some_custom_layer"`
#'   - Named function: A function object that constructs the layer
#'   - Anonymous function: e.g., `\() nn_linear()` or `function() nn_linear()`
#'   
#'   The layer constructor is first searched in the current environment, then in parent
#'   environments, and finally falls back to the `{torch}` namespace. This allows you to
#'   use custom layer implementations alongside standard torch layers.
#' 
#' @param out_nn_layer Default `NULL`. If supplied, it forces to be the neural network layer to be used 
#'   on the last layer. Can be specified as:
#'   - Character string, e.g. `"nn_linear"`, `"nn_gru"`, `"nn_lstm"`, `"some_custom_layer"`
#'   - Named function: A function object that constructs the layer
#'   - Formula interface, e.g. `~torch::nn_linear`, `~some_custom_layer`
#'   
#'   Internally, it almost works the same as `nn_layer` parameter. 
#' 
#' @param nn_layer_args Named list of additional arguments passed to the layer constructor
#'   specified by `nn_layer`. These arguments are applied to all layers. For layer-specific
#'   arguments, use `layer_arg_fn`. Default is an empty list.
#'
#' @param layer_arg_fn Optional function or formula that generates layer-specific construction arguments.
#'   Can be specified as:
#'   - **Formula**: `~ list(input_size = .in, hidden_size = .out)` where `.in`, `.out`, `.i`, and `.is_output` are available
#'   - **Function**: `function(i, in_dim, out_dim, is_output)` with signature as before
#'   
#'   The formula/function should return a named list of arguments to pass to the layer constructor.
#'   Available variables in formula context:
#'   
#'   - `.i` or `i`: Integer, the layer index (1-based)
#'   - `.in` or `in_dim`: Integer, input dimension for this layer
#'   - `.out` or `out_dim`: Integer, output dimension for this layer  
#'   - `.is_output` or `is_output`: Logical, whether this is the final output layer
#'   
#'   If `NULL`, defaults to FFNN-style arguments: `list(in_dim, out_dim, bias = bias)`.
#'
#' @param forward_extract Optional formula or function that processes layer outputs in the forward pass.
#'   Useful for layers that return complex structures (e.g., RNNs return `list(output, hidden)`).
#'   Can be specified as:
#'   
#'   - **Formula**: `~ .[[1]]` or `~ .$output` where `.` represents the layer output
#'   - **Function**: `function(expr)` that accepts/returns a language object
#'   
#'   Common patterns: 
#'   
#'   - Extract first element: `~ .[[1]]`
#'   - Extract named element: `~ .$output`
#'   - Extract with method: `~ .$get_output()`
#'   
#'   If `NULL`, layer outputs are used directly.
#'
#' @param before_output_transform Optional formula or function that transforms input before the output layer.
#'   This is applied after the last hidden layer (and its activation) but before the output layer.
#'   Can be specified as:
#'   
#'   - **Formula**: `~ .[, .$size(2), ]` where `.` represents the current tensor
#'   - **Function**: `function(expr)` that accepts/returns a language object
#'   
#'   Common patterns:
#'   
#'   - Extract last timestep: `~ .[, .$size(2), ]`
#'   - Flatten: `~ .$flatten(start_dim = 1)`
#'   - Global pooling: `~ .$mean(dim = 2)`
#'   - Extract token: `~ .[, 1, ]`
#'   
#'   If `NULL`, no transformation is applied.
#' 
#' @param after_output_transform Optional formula or function that transforms the output after the output layer.
#'   This is applied after `self$out(x)` (the final layer) but before returning the result.
#'   Can be specified as:
#'   
#'   - **Formula**: `~ .$mean(dim = 2)` where `.` represents the output tensor
#'   - **Function**: `function(expr)` that accepts/returns a language object
#'   
#'   Common patterns:
#'   
#'   - Global average pooling: `~ .$mean(dim = 2)`
#'   - Squeeze dimensions: `~ .$squeeze()`
#'   - Reshape output: `~ .$view(c(-1, 10))`
#'   - Extract specific outputs: `~ .[, , 1:5]`
#'   
#'   If `NULL`, no transformation is applied.
#' 
#' @param last_layer_args Optional named list or formula specifying additional arguments 
#'   for the output layer only. These arguments are appended to the output layer constructor
#'   after the arguments from `layer_arg_fn`. Can be specified as:
#'   
#'   - **Formula**: `~ list(kernel_size = 2L, bias = FALSE)` 
#'   - **Named list**: `list(kernel_size = 2L, bias = FALSE)`
#'   
#'   This is useful when you need to override or add specific parameters to the final layer
#'   without affecting hidden layers. For example, in CNNs you might want a different kernel
#'   size for the output layer, or in RNNs you might want to disable bias in the final linear
#'   projection. Arguments in `last_layer_args` will override any conflicting arguments from
#'   `layer_arg_fn` when `.is_output = TRUE`. Default is an empty list.
#' 
#' @param hd_neurons Integer vector specifying the number of neurons (hidden units) in each 
#'   hidden layer. The length determines the number of hidden layers in the network.
#'   Must contain at least one element.
#'
#' @param no_x Integer specifying the number of input features (input dimension).
#'
#' @param no_y Integer specifying the number of output features (output dimension).
#'
#' @param activations Activation function specifications for hidden layers. Can be:
#'   - `NULL`: No activation functions applied
#'   - Character vector: e.g., `c("relu", "sigmoid", "tanh")`
#'   - `activation_spec` object: Created using `act_funs()`, which allows
#'     specifying custom arguments. See examples.
#'   
#'   If a single activation is provided, it will be replicated across all hidden layers.
#'   Otherwise, the length should match the number of hidden layers.
#'
#' @param output_activation Optional activation function for the output layer.
#'   Same format as `activations`, but should specify only a single activation.
#'   Common choices include `"softmax"` for classification or `"sigmoid"` for 
#'   binary outcomes. Default is `NULL` (no output activation).
#'
#' @param bias Logical indicating whether to include bias terms in layers.
#'   Default is `TRUE`. Note that this is passed to `layer_arg_fn` if provided,
#'   so custom layer argument functions should handle this parameter appropriately.
#'
#' @param eval Logical indicating whether to evaluate the generated expression immediately.
#'   If `TRUE`, returns an instantiated `nn_module` class that can be called directly
#'   (e.g., `model()`). If `FALSE` (default), returns the unevaluated language expression
#'   that can be inspected or evaluated later with `eval()`. Default is `FALSE`.
#'
#' @param use_namespace Logical or character. Controls how layer functions are namespaced in 
#'   the generated code:
#'   - `TRUE` (default): Functions are namespaced to `{torch}` (e.g., `torch::nn_linear`)
#'   - `FALSE`: No namespace prefix is added (functions used as-is from current environment)
#'   - Character string: Custom namespace (e.g., `"mypackage"` produces `mypackage::my_layer`)
#'   
#'   When using custom layers from your environment, set to `FALSE` to avoid forcing
#'   torch namespace resolution.
#'
#' @param ... Additional arguments passed to layer constructors or for future extensions.
#'
#' @return 
#' If `eval = FALSE` (default): A language object (unevaluated expression) representing 
#' a `torch::nn_module` definition. This expression can be evaluated with `eval()` to 
#' create the module class, which can then be instantiated with `eval(result)()` to 
#' create a model instance.
#' 
#' If `eval = TRUE`: An instantiated `nn_module` class constructor that can be called 
#' directly to create model instances (e.g., `result()`).
#'
#' @examples
#' \dontrun{
#' \donttest{
#' if (torch::torch_is_installed()) {
#'     # Basic usage with formula interface
#'     nn_module_generator(
#'         nn_name = "MyGRU",
#'         nn_layer = "nn_gru",
#'         layer_arg_fn = ~ if (.is_output) {
#'             list(.in, .out)
#'         } else {
#'             list(input_size = .in, hidden_size = .out, 
#'                  num_layers = 1L, batch_first = TRUE)
#'         },
#'         forward_extract = ~ .[[1]],
#'         before_output_transform = ~ .[, .$size(2), ],
#'         hd_neurons = c(128, 64, 32),
#'         no_x = 20,
#'         no_y = 5,
#'         activations = "relu"
#'     )
#'     
#'     # LSTM with cleaner syntax
#'     nn_module_generator(
#'         nn_name = "MyLSTM",
#'         nn_layer = "nn_lstm",
#'         layer_arg_fn = ~ list(
#'             input_size = .in,
#'             hidden_size = .out,
#'             batch_first = TRUE
#'         ),
#'         forward_extract = ~ .[[1]],
#'         before_output_transform = ~ .[, .$size(2), ],
#'         hd_neurons = c(64, 32),
#'         no_x = 10,
#'         no_y = 2
#'     )
#'     
#'     # CNN with global average pooling
#'     nn_module_generator(
#'         nn_name = "SimpleCNN",
#'         nn_layer = "nn_conv1d",
#'         layer_arg_fn = ~ list(
#'             in_channels = .in,
#'             out_channels = .out,
#'             kernel_size = 3L,
#'             padding = 1L
#'         ),
#'         before_output_transform = ~ .$mean(dim = 2),
#'         hd_neurons = c(16, 32, 64),
#'         no_x = 1,
#'         no_y = 10,
#'         activations = "relu"
#'     )
#'     
#'     # CNN with after_output_transform (pooling applied AFTER output layer)
#'     nn_module_generator(
#'         nn_name = "CNN1DClassifier",
#'         nn_layer = "nn_conv1d",
#'         layer_arg_fn = ~ if (.is_output) {
#'             list(.in, .out)
#'         } else {
#'             list(
#'                 in_channels = .in,
#'                 out_channels = .out,
#'                 kernel_size = 3L,
#'                 stride = 1L,
#'                 padding = 1L 
#'             )
#'         },
#'         after_output_transform = ~ .$mean(dim = 2),
#'         last_layer_args = list(kernel_size = 1, stride = 2),
#'         hd_neurons = c(16, 32, 64),
#'         no_x = 1,
#'         no_y = 10,
#'         activations = "relu"
#'     )
#'     
#' } else {
#'   message("torch not installed - skipping examples")
#' }
#' }
#' }
#'
#' @importFrom rlang new_function call2 expr sym f_rhs is_formula
#' @importFrom purrr map map2
#' @importFrom glue glue
#' @importFrom cli cli_abort
#'
#' @export
nn_module_generator = 
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
        hd_neurons,
        no_x,
        no_y,
        activations = NULL,
        output_activation = NULL,
        bias = TRUE,
        eval = FALSE, 
        use_namespace = TRUE,
        ...
    ) {
    if (is.null(nn_layer)) nn_layer = "nn_linear"
    
    if (missing(hd_neurons) || is.null(hd_neurons) || length(hd_neurons) == 0L) {
        hd_neurons = integer(0) 
    }
    
    if (missing(no_x) || missing(no_y)) {
        cli::cli_abort("Both {.arg no_x} and {.arg no_y} must be specified.")
    }
    
    act_specs = eval_act_funs({{ activations }}, {{ output_activation }})
    activations = act_specs$activations
    output_activation = act_specs$output_activation
    
    # ---- INPUT PROCESSING ----
    layer_arg_fn = formula_to_function(
        layer_arg_fn,
        default_fn = function(i, in_dim, out_dim, is_output) {
            list(in_dim, out_dim, bias = bias)
        },
        arg_names = c("i", "in_dim", "out_dim", "is_output"),
        alias_map = list(
            i = ".i",
            in_dim = ".in", 
            out_dim = ".out",
            is_output = ".is_output"
        )
    )
    
    forward_extract = formula_to_expr_transformer(forward_extract)
    before_output_transform = formula_to_expr_transformer(before_output_transform)
    after_output_transform = formula_to_expr_transformer(after_output_transform)
    
    # ---- Process 1: Architecture setup ----
    nodes = c(no_x, hd_neurons, no_y)
    n_layers = length(nodes) - 1L
    n_hidden = length(hd_neurons)
    
    # ---- Process 2: Tweak activations input ----
    activation_spec = parse_activation_spec(activations, n_hidden)
    activation_calls = process_activations(activation_spec, prefix = "nnf_")
    
    if (!is.null(output_activation)) {
        output_spec = parse_activation_spec(output_activation, 1L)
        output_call = process_activations(output_spec, prefix = "nnf_")[[1]]
    } else {
        output_call = NULL
    }
    
    all_activation_calls = c(activation_calls, list(output_call))
    
    # ---- Build initialize() ----
    init_body = map(seq_len(n_layers), function(i) {
        is_output = (i == n_layers)
        layer_name = if (is_output) "out" else glue("{substring(nn_layer, 4)}_{i}")
        in_dim = nodes[i]
        out_dim = nodes[i + 1]
        
        layer_args = layer_arg_fn(i, in_dim, out_dim, is_output)
        
        current_layer = if (is_output && nn_layer %in% c("nn_linear", "nn_gru", "nn_lstm", "nn_rnn")) {
            "nn_linear"
        } else if (is_output && !is.null(out_nn_layer)) {
            out_nn_layer
        } else {
            nn_layer
        }
        
        additional_args = if (is_output && !is.null(last_layer_args)) {
            if (rlang::is_formula(last_layer_args)) {
                eval(rlang::f_rhs(last_layer_args))
            } else if (is.list(last_layer_args)) {
                last_layer_args
            } else {
                list()
            }
        } else {
            list()
        }
        
        layer_expr = if (is.function(current_layer)) {
            rlang::enexpr(current_layer)
        } else if (is.character(current_layer)) {
            rlang::parse_expr(current_layer)
        } else if (rlang::is_formula(current_layer)) {
            rlang::f_rhs(current_layer)
        } else if (is.symbol(current_layer) || is.call(current_layer)) {
            current_layer
        } else { 
            cli::cli_abort("{.arg {out_nn_layer}} must be a string, symbol, or function, got {class(current_layer)[1]}")
        }
        
        layer_call = call2(
            layer_expr,
            !!!c(layer_args, nn_layer_args, additional_args),
            .ns = if (rlang::is_true(use_namespace)) {
                "torch"
            } else if (rlang::is_false(use_namespace)) {
                NULL
            } else {
                use_namespace  
            }
        )
        
        call2("=", call2("$", expr(self), sym(layer_name)), layer_call)
    })
    
    init = new_function(
        args = pairlist(),
        body = call2("{", !!!init_body)
    )
    
    # ---- Build forward() ----
    forward_body_exprs = map(seq_len(n_layers), function(i) {
        is_output = (i == n_layers)
        is_last_hidden = (i == n_layers - 1L)
        layer_name = if (is_output) "out" else glue("{substring(nn_layer, 4)}_{i}")
        act_call_fn = all_activation_calls[[i]]
        
        layer_expr = call2(call2("$", expr(self), sym(layer_name)), expr(x))
        if (!is.null(forward_extract) && !is_output) {
            layer_expr = forward_extract(layer_expr)
        }
        
        line1 = call2("=", expr(x), layer_expr)
        
        out = list(line1)
        
        if (!is.null(act_call_fn)) {
            line2 = call2("=", expr(x), act_call_fn(expr(x)))
            out = c(out, list(line2))
        }
        
        # Apply before_output_transform after last hidden layer
        # This happens AFTER the last hidden layer's activation
        if (is_last_hidden && !is.null(before_output_transform)) {
            transform_line = call2("=", expr(x), before_output_transform(expr(x)))
            out = c(out, list(transform_line))
        }
        
        # Apply after_output_transform after output layer
        # This happens AFTER self$out(x)
        if (is_output && !is.null(after_output_transform)) {
            transform_line = call2("=", expr(x), after_output_transform(expr(x)))
            out = c(out, list(transform_line))
        }
        
        out
    })
    
    forward_body_exprs = c(
        unlist(forward_body_exprs, recursive = FALSE),
        list(expr(x)) 
    )
    
    forward = new_function(
        args = list(x = expr()),
        body = call2("{", !!!forward_body_exprs)
    )
    
    # ---- Build final nn_module call ----
    full_call = call2(
        expr(nn_module),
        nn_name,
        initialize = init,
        forward = forward,
        .ns = "torch"
    )
    
    if (eval) eval(full_call) else full_call
}


#' Formula to Function with Named Arguments
#' 
#' @param formula_or_fn A formula or function
#' @param default_fn Default function if `formula_or_fn` is `NULL`
#' @param arg_names Character vector of formal argument names
#' @param alias_map Named list mapping arg_names to formula aliases (e.g., list(in_dim = ".in"))
#' 
#' @return A function
#' @keywords internal
formula_to_function = function(formula_or_fn, default_fn = NULL, arg_names = NULL, alias_map = NULL) {
    if (is.null(formula_or_fn)) {
        return(default_fn)
    }
    
    if (rlang::is_formula(formula_or_fn)) {
        rhs = rlang::f_rhs(formula_or_fn)
        args_list = setNames(rep(list(rlang::missing_arg()), length(arg_names)), arg_names)
        
        if (!is.null(alias_map)) {
            alias_assignments = lapply(names(alias_map), function(arg) {
                alias = alias_map[[arg]]
                call2("=", sym(alias), sym(arg))
            })
            body_expr = call2("{", !!!alias_assignments, rhs)
        } else {
            body_expr = rhs
        }
        
        fn = rlang::new_function(
            args = args_list,
            body = body_expr,
            env = rlang::f_env(formula_or_fn)
        )
        return(fn)
    }
    
    if (is.function(formula_or_fn)) {
        return(formula_or_fn)
    }
    
    cli::cli_abort("Expected a formula or function, got {class(formula_or_fn)[1]}")
}


#' Convert Formula to Expression Transformer
#' 
#' @param formula_or_fn A formula like `~ .[[1]]` or a function that transforms expressions
#' 
#' @return A function that takes an expression and returns a transformed expression, or NULL
#' @keywords internal
formula_to_expr_transformer = function(formula_or_fn) {
    if (is.null(formula_or_fn)) {
        return(NULL)
    }
    
    if (rlang::is_formula(formula_or_fn)) {
        rhs = rlang::f_rhs(formula_or_fn)
        
        return(function(expr) {
            substitute_dot(rhs, expr)
        })
    }
    
    if (is.function(formula_or_fn)) {
        return(formula_or_fn)
    }
    
    cli::cli_abort("Expected a formula or function, got {class(formula_or_fn)[1]}")
}


#' Recursively Substitute . with Expression
#' 
#' @param expr Expression containing `.` placeholders
#' @param replacement Expression to substitute for `.`
#' 
#' @return Modified expression
#' @keywords internal
substitute_dot = function(expr, replacement) {
    if (is.symbol(expr) && identical(expr, quote(.))) {
        return(replacement)
    }
    
    if (is.call(expr)) {
        expr[] = lapply(expr, function(e) substitute_dot(e, replacement))
        return(expr)
    }
    
    expr
}

#' Layer argument pronouns for formula-based specifications
#' 
#' @description
#' These pronouns provide a cleaner, more readable way to reference layer parameters
#' in formula-based specifications for `nn_module_generator()` and related functions.
#' They work similarly to `rlang::.data` and `rlang::.env`.
#' 
#' @details
#' Available pronouns:
#' 
#' - `.layer`: Access all layer parameters as a list-like object
#' - `.i`: Layer index (1-based integer)
#' - `.in`: Input dimension for the layer
#' - `.out`: Output dimension for the layer
#' - `.is_output`: Logical indicating if this is the output layer
#' 
#' These pronouns can be used in formulas passed to:
#' 
#' - `layer_arg_fn` parameter
#' - Custom layer configuration functions
#' 
#' @section Usage:
#' 
#' ``` r
#' # Using individual pronouns
#' layer_arg_fn = ~ list(
#'     input_size = .in,
#'     hidden_size = .out,
#'     num_layers = if (.i == 1) 2L else 1L
#' )
#' 
#' # Using .layer pronoun (alternative syntax)
#' layer_arg_fn = ~ list(
#'     input_size = .layer$ind,
#'     hidden_size = .layer$out,
#'     is_first = .layer$i == 1
#' )
#' ```
#' 
#' @name layer_prs
#' @aliases .layer .i .in .out .is_output
NULL

#' @rdname layer_prs
#' @export
.layer = structure(
    list(),
    class = c("layer_pr", "list")
)

#' @rdname layer_prs
#' @export
.i = structure(
    list(),
    class = c("layer_index_pr", "layer_pr", "list")
)

#' @rdname layer_prs
#' @export
.in = structure(
    list(),
    class = c("layer_input_pr", "layer_pr", "list")
)

#' @rdname layer_prs
#' @export
.out = structure(
    list(),
    class = c("layer_output_pr", "layer_pr", "list")
)

#' @rdname layer_prs
#' @export
.is_output = structure(
    list(),
    class = c("layer_is_output_pr", "layer_pr", "list")
)

#' "Layer" attributes
#' 
#' @param x The .layer itself
#' @param name It could be the following: 
#' -  `i`: Layer index (1-based integer)
#' -  `ind`: Input dimension for the layer
#' -  `out`: Output dimension for the layer
#' -  `is_output`: Logical indicating if this is the output layer
#' 
#' @return A pronoun, it returns nothing
#' 
#' @name layer-attributes
#' @export
`$.layer_pr` = function(x, name) {
    if (inherits(x, "layer_index_pr"))  return(quote(i))
    if (inherits(x, c("layer_input_pr", "layer_output_pr", "layer_is_output_pr"))) {
        pr = switch(
            class(x),
            layer_input_pr  = quote(in_dim),
            layer_output_pr = quote(out_dim),
            layer_is_output_pr = quote(is_output)
        )
        
        return(pr)
    }
    
    switch(
        name,
        i = quote(i),
        ind = quote(in_dim),
        out = quote(out_dim),
        is_output = quote(is_output),
        cli::cli_abort("Unknown layer pronoun field: {name}")
    )
}

#' Print method for the pronouns
#' 
#' @param x An object of class "ffnn_fit"
#' @param ... Additional arguments (unused)
#'
#' @return No return value, prints out the type of pronoun to be used
#' 
#' @section For `.layer`: 
#' It displays what fields to be accessed by `$`.
#' 
#' @rdname print-layer_pronoun
#' @export
print.layer_pr = function(x, ...) {
    cat("<layer pronoun>\n")
    cat("Fields: i, ind, out, is_output\n")
    invisible(x)
}

#' @rdname print-layer_pronoun
#' @export
print.layer_index_pr = function(x, ...) {
    cat("<layer index pronoun>\n")
    invisible(x)
}

#' @rdname print-layer_pronoun
#' @export
print.layer_input_pr = function(x, ...) {
    cat("<layer input dimension pronoun>\n")
    invisible(x)
}

#' @rdname print-layer_pronoun
#' @export
print.layer_output_pr = function(x, ...) {
    cat("<layer output dimension pronoun>\n")
    invisible(x)
}

#' @rdname print-layer_pronoun
#' @export
print.layer_is_output_pr = function(x, ...) {
    cat("<layer is_output flag pronoun>\n")
    invisible(x)
}
