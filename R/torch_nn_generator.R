#' Functions to generate `nn_module` (language) expression
#'
#' @name nn_gens
#'
#' @section Feed-Forward Neural Network Module Generator:
#' The `ffnn_generator()` function generates a feed-forward neural network (FFNN) module expression
#' from the `torch` package in R. It allows customization of the FFNN architecture,
#' including the number of hidden layers, neurons, and activation functions.
#'
#' @param nn_name Character. Name of the generated FFNN module class. Default is `"DeepFFN"`.
#' @param hd_neurons Integer vector. Number of neurons in each hidden layer.
#' @param no_x Integer. Number of input features.
#' @param no_y Integer. Number of output features.
#' @param activations Activation function specifications for each hidden layer.
#' Can be:
#' - `NULL`: No activation functions.
#' - Character vector: e.g., `c("relu", "sigmoid")`.
#' - List: e.g., `act_funs(relu, elu, softshrink = args(lambd = 0.5))`.
#' - `activation_spec` object from `act_funs()`.
#'
#' If the length of `activations` is `1L`, this will be the activation throughout the architecture.
#'
#' @param output_activation Optional. Activation function for the output layer.
#' Same format as `activations` but should be a single activation.
#' @param bias Logical. Whether to use bias weights. Default is `TRUE`.
#'
#' @return A `torch` module expression representing the FFNN.
#'
#' @details The generated FFNN module will have the specified number of hidden layers,
#' with each layer containing the specified number of neurons. Activation functions
#' can be applied after each hidden layer as specified.
#' This can be used for both classification and regression tasks.
#'
#' The generated module properly namespaces all torch functions to avoid
#' polluting the global namespace.
#'
#' @examples
#' \donttest{
#' # FFNN
#' if (torch::torch_is_installed()) {
#'     # Generate an MLP module with 3 hidden layers
#'     ffnn_mod = ffnn_generator(
#'         nn_name = "MyFFNN",
#'         hd_neurons = c(64, 32, 16),
#'         no_x = 10,
#'         no_y = 1,
#'         activations = 'relu'
#'     )
#'
#'     # Evaluate and instantiate
#'     model = eval(ffnn_mod)()
#'
#'     # More complex: With different activations
#'     ffnn_mod2 = ffnn_generator(
#'         nn_name = "MyFFNN2",
#'         hd_neurons = c(128, 64, 32),
#'         no_x = 20,
#'         no_y = 5,
#'         activations = act_funs(
#'             relu,
#'             selu,
#'             sigmoid
#'         )
#'     )
#'
#'     # Even more complex: Different activations and customized argument
#'     # for the specific activation function
#'     ffnn_mod2 = ffnn_generator(
#'         nn_name = "MyFFNN2",
#'         hd_neurons = c(128, 64, 32),
#'         no_x = 20,
#'         no_y = 5,
#'         activations = act_funs(
#'             relu,
#'             selu,
#'             softshrink = args(lambd = 0.5)
#'         )
#'     )
#'
#'     # Customize output activation (softmax is useful for classification tasks)
#'     ffnn_mod3 = ffnn_generator(
#'         hd_neurons = c(64, 32),
#'         no_x = 10,
#'         no_y = 3,
#'         activations = 'relu',
#'         output_activation = act_funs(softmax = args(dim = 2L))
#'     )
#' } else {
#'     message("Torch not fully installed — skipping example")
#' }
#' }
#'
#' @importFrom rlang new_function call2 expr sym
#' @importFrom purrr map map2
#' @importFrom glue glue
#' @importFrom cli cli_abort
#'
#' @export
ffnn_generator = function(nn_name = "DeepFFN",
                          hd_neurons,
                          no_x,
                          no_y,
                          activations = NULL,
                          output_activation = NULL,
                          bias = TRUE) {

    nodes = c(no_x, hd_neurons, no_y)
    n_layers = length(nodes) - 1L
    n_hidden = length(hd_neurons)

    if (n_layers < 1L) {
        cli_abort(c(
            "{.arg hd_neurons} must contain at least one hidden layer size.",
            "i" = "You provided {.val {hd_neurons}} (length {n_hidden})."
        ), class = "nn_module_error")
    }

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
    init_body = map2(
        .x = seq_len(n_layers),
        .y = map2(nodes[-length(nodes)], nodes[-1], c),
        .f = function(i, dims) {
            layer_name = if (i == n_layers) "out" else glue("fc{i}")
            call2(
                "=",
                call2("$", expr(self), sym(layer_name)),
                call2(
                    sym("nn_linear"),
                    !!!dims,
                    bias = bias,
                    .ns = "torch"
                )
            )
        }
    )

    init = new_function(
        args = pairlist(),
        body = call2("{", !!!init_body)
    )

    # ---- Build forward() ----
    forward_body_exprs = map(seq_len(n_layers), function(i) {
        layer_name = if (i == n_layers) "out" else glue("fc{i}")
        act_call_fn = all_activation_calls[[i]]

        line1 = call2(
            "=", expr(x),
            call2(call2("$", expr(self), sym(layer_name)), expr(x))
        )

        if (is.null(act_call_fn)) {
            list(line1)
        } else {
            line2 = call2("=", expr(x), act_call_fn(expr(x)))
            list(line1, line2)
        }
    })

    forward_body_exprs = c(unlist(forward_body_exprs, recursive = FALSE), list(expr(x)))

    forward = new_function(
        args = list(x = expr()),
        body = call2("{", !!!forward_body_exprs)
    )

    call2(
        sym("nn_module"),
        nn_name, initialize = init, forward = forward,
        .ns = "torch"
    )
}

#' Check RNN Type Validity
#'
#' @param rnn_type Character. The RNN type to validate.
#' @param hd_neurons Integer vector. Hidden neurons (for error context).
#'
#' @importFrom cli cli_abort
#'
#' @noRd
check_rnn_type = function(rnn_type, hd_neurons) {
    valid_types = c("rnn", "lstm", "gru")

    rnn_type = tolower(rnn_type)

    if (!rnn_type %in% valid_types) {
        cli_abort(c(
            "{.arg rnn_type} must be one of {.val {valid_types}}.",
            x = "You provided {.val {rnn_type}}."
        ), class = "rnn_type_error")
    }

    n_rnn_layers = length(hd_neurons)
    if (n_rnn_layers == 0) {
        cli_abort(c(
            "{.arg hd_neurons} must contain at least one hidden layer size.",
            i = "RNNs require at least one recurrent layer."
        ), class = "rnn_module_error")
    }

    invisible(NULL)
}

#' @rdname nn_gens
#'
#' @section Recurrent Neural Network Module Generator:
#' The `rnn_generator()` function generates a recurrent neural network (RNN) module expression
#' from the `torch` package in R. It allows customization of the RNN architecture,
#' including the number of hidden layers, neurons, RNN type, activation functions,
#' and other parameters.
#'
#' @param nn_name Character. Name of the generated RNN module class. Default is `"DeepRNN"`.
#' @param hd_neurons Integer vector. Number of neurons in each hidden RNN layer.
#' @param no_x Integer. Number of input features.
#' @param no_y Integer. Number of output features.
#' @param rnn_type Character. Type of RNN to use. Must be one of `"rnn"`, `"lstm"`, or `"gru"`. Default is `"lstm"`.
#' @param activations Activation function specifications for each hidden layer.
#' Can be:
#' - `NULL`: No activation functions.
#' - Character vector: e.g., `c("relu", "sigmoid")`.
#' - List: e.g., `act_funs(relu, elu, softshrink = args(lambd = 0.5))`.
#' - `activation_spec` object from `act_funs()`.
#'
#' If the length of `activations` is `1L`, this will be the activation throughout the architecture.
#'
#' @param output_activation Optional. Activation function for the output layer.
#' Same format as `activations` but should be a single activation.
#' @param bias Logical. Whether to use bias weights. Default is `TRUE`
#' @param bidirectional Logical. Whether to use bidirectional RNN layers. Default is `TRUE`.
#' @param dropout Numeric. Dropout rate between RNN layers. Default is `0`.
#' @param ... Additional arguments (currently unused).
#'
#' @return A `torch` module expression representing the RNN.
#'
#' @details The generated RNN module will have the specified number of recurrent layers,
#' with each layer containing the specified number of hidden units. Activation functions
#' can be applied after each RNN layer as specified. The final output is taken from the
#' last time step and passed through a linear layer.
#'
#' The generated module properly namespaces all torch functions to avoid
#' polluting the global namespace.
#'
#' @examples
#' \donttest{
#' ## RNN
#' if (torch::torch_is_installed()) {
#'     # Basic LSTM with 2 layers
#'     rnn_mod = rnn_generator(
#'         nn_name = "MyLSTM",
#'         hd_neurons = c(64, 32),
#'         no_x = 10,
#'         no_y = 1,
#'         rnn_type = "lstm",
#'         activations = 'relu'
#'     )
#'
#'     # Evaluate and instantiate
#'     model = eval(rnn_mod)()
#'
#'     # GRU with different activations
#'     rnn_mod2 = rnn_generator(
#'         nn_name = "MyGRU",
#'         hd_neurons = c(128, 64, 32),
#'         no_x = 20,
#'         no_y = 5,
#'         rnn_type = "gru",
#'         activations = act_funs(relu, elu, relu),
#'         bidirectional = FALSE
#'     )
#'
#' } else {
#'     message("Torch not fully installed — skipping example")
#' }
#' }
#'
#' \dontrun{
#' ## Parameterized activation and dropout
#' # (Will throw an error due to `nnf_tanh()` not being available in `{torch}`)
#' # rnn_mod3 = rnn_generator(
#' #     hd_neurons = c(100, 50, 25),
#' #     no_x = 15,
#' #     no_y = 3,
#' #     rnn_type = "lstm",
#' #     activations = act_funs(
#' #         relu,
#' #         leaky_relu = args(negative_slope = 0.01),
#' #         tanh
#' #     ),
#' #     bidirectional = TRUE,
#' #     dropout = 0.3
#' # )
#' }
#'
#' @importFrom rlang new_function call2 expr sym
#' @importFrom purrr map map2
#' @importFrom glue glue
#' @importFrom cli cli_abort
#'
#' @export
rnn_generator = function(nn_name = "DeepRNN",
                         hd_neurons,
                         no_x,
                         no_y,
                         rnn_type = "lstm",
                         bias = TRUE,
                         activations = NULL,
                         output_activation = NULL,
                         bidirectional = TRUE,
                         dropout = 0,
                         ...) {

    check_rnn_type(rnn_type, hd_neurons)

    n_rnn_layers = length(hd_neurons)

    activation_spec = parse_activation_spec(activations, n_rnn_layers)
    activation_calls = process_activations(activation_spec, prefix = "nnf_")

    if (!is.null(output_activation)) {
        output_spec = parse_activation_spec(output_activation, 1L)
        output_call = process_activations(output_spec, prefix = "nnf_")[[1]]
    } else {
        output_call = NULL
    }

    # ---- Build initialize() ----
    input_sizes = c(no_x, hd_neurons[-n_rnn_layers] * (if (bidirectional) 2L else 1L))

    rnn_layers = map2(seq_len(n_rnn_layers), input_sizes, function(i, input_size) {
        layer_name = glue("rnn{i}")
        hidden_size = hd_neurons[i]
        layer_dropout = if (i < n_rnn_layers && dropout > 0) dropout else 0

        rnn_fn_name = paste0("nn_", rnn_type)
        rnn_call = call2(
            sym(rnn_fn_name),
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = 1L,
            bias = bias,
            batch_first = TRUE,
            bidirectional = bidirectional,
            dropout = layer_dropout,
            .ns = "torch"
        )

        call2("=", call2("$", expr(self), sym(layer_name)), rnn_call)
    })

    final_hidden = hd_neurons[n_rnn_layers] * (if (bidirectional) 2L else 1L)

    output_layer = call2(
        "=", call2("$", expr(self), sym("out")),
        call2(sym("nn_linear"), final_hidden, no_y, .ns = "torch")
    )

    init_body = c(rnn_layers, list(output_layer))

    init = new_function(
        args = pairlist(),
        body = call2("{", !!!init_body)
    )

    # ---- Build forward() ----
    rnn_forward_exprs = map(seq_len(n_rnn_layers), function(i) {
        layer_name = glue("rnn{i}")
        act_call_fn = activation_calls[[i]]

        rnn_call_expr = call2(
            "=", expr(x),
            call2("[[",
                  call2(call2("$", expr(self), sym(layer_name)), expr(x)),
                  1L)
        )

        if (is.null(act_call_fn)) {
            list(rnn_call_expr)
        } else {
            activation_expr = call2("=", expr(x), act_call_fn(expr(x)))
            list(rnn_call_expr, activation_expr)
        }
    })

    slice_expr = call2(
        "=", expr(x),
        call2(
            "[", expr(x), expr(),
            call2(call2("$", expr(x), sym("size")), 2L),
            expr()
        )
    )

    output_expr = call2(
        "=", expr(x), call2(call2("$", expr(self), sym("out")), expr(x))
    )

    if (!is.null(output_call)) {
        output_activation_expr = call2("=", expr(x), output_call(expr(x)))
        forward_body_exprs = c(
            unlist(rnn_forward_exprs, recursive = FALSE),
            list(slice_expr, output_expr, output_activation_expr, expr(x))
        )
    } else {
        forward_body_exprs = c(
            unlist(rnn_forward_exprs, recursive = FALSE),
            list(slice_expr, output_expr, expr(x))
        )
    }

    forward = new_function(
        args = alist(x = ),
        body = call2("{", !!!forward_body_exprs)
    )

    call2(
        sym("nn_module"),
        nn_name,
        initialize = init,
        forward = forward,
        .ns = "torch"
    )
}
