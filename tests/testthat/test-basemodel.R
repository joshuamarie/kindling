test_that("FFNN generator uses torch namespace explicitly", {
    skip_if_not_installed("torch")

    expr = ffnn_generator(
        nn_name = "TestFFNN",
        hd_neurons = c(10, 5),
        no_x = 3,
        no_y = 1,
        activations = "relu"
    )
    code_str = paste(deparse(expr), collapse = " ")

    expect_match(code_str, "torch::nn_module", fixed = TRUE)
    expect_match(code_str, "torch::nn_linear", fixed = TRUE)
    expect_match(code_str, "torch::nnf_relu", fixed = TRUE)
    expect_false(grepl("(?<!torch::)\\bnn_linear\\s*\\(", code_str, perl = TRUE))
    expect_false(grepl("(?<!torch::)\\bnnf_relu\\s*\\(", code_str, perl = TRUE))
})

test_that("RNN generator uses torch namespace explicitly", {
    skip_if_not_installed("torch")

    expr = rnn_generator(
        nn_name = "TestRNN",
        hd_neurons = c(20, 10, 15, 8),
        no_x = 5,
        no_y = 2,
        rnn_type = "lstm",
        activations = act_funs(relu, selu, softshrink = args(lambd = 0.5), gelu)
    )

    code_str = paste(deparse(expr), collapse = " ")

    expect_match(code_str, "torch::nn_module", fixed = TRUE)
    expect_match(code_str, "torch::nn_lstm", fixed = TRUE)
    expect_match(code_str, "torch::nn_linear", fixed = TRUE)
    expect_match(code_str, "torch::nnf_relu", fixed = TRUE)
    expect_match(code_str, "torch::nnf_selu", fixed = TRUE)
    expect_match(code_str, "torch::nnf_softshrink", fixed = TRUE)
    expect_match(code_str, "torch::nnf_gelu", fixed = TRUE)
})

test_that("Invalid activation 'tanh' throws error in FFNN generator", {
    skip_if_not_installed("torch")

    expect_error(
        ffnn_generator(
            hd_neurons = c(10, 5),
            no_x = 3,
            no_y = 1,
            activations = act_funs(relu, tanh)
        ),
        class = "activation_not_found_error"
    )
})

test_that("Invalid activation 'tanh' throws error in RNN generator", {
    skip_if_not_installed("torch")

    expect_error(
        rnn_generator(
            hd_neurons = c(20, 10),
            no_x = 5,
            no_y = 2,
            rnn_type = "lstm",
            activations = act_funs(relu, tanh)
        ),
        class = "activation_not_found_error"
    )
})

test_that("Generated model can be evaluated without library(torch)", {
    skip_if_not_installed("torch")

    test_env = new.env()
    expr = ffnn_generator(
        hd_neurons = c(10),
        no_x = 5,
        no_y = 1,
        activations = "relu"
    )

    expect_error({
        model_class = eval(expr, envir = test_env)
        model = model_class()
    }, NA)
})

test_that("Activation functions are properly namespaced", {
    skip_if_not_installed("torch")

    spec = parse_activation_spec(
        act_funs(relu, selu, leaky_relu = args(negative_slope = 0.01)),
        n_layers = 3
    )

    calls = process_activations(spec)
    relu_call = calls[[1]](quote(x))

    expect_equal(as.character(relu_call[[1]][[1]]), "::")
    expect_equal(as.character(relu_call[[1]][[2]]), "torch")
    expect_equal(as.character(relu_call[[1]][[3]]), "nnf_relu")
})

test_that("Complex activations maintain namespace", {
    skip_if_not_installed("torch")

    expr = ffnn_generator(
        hd_neurons = c(128, 64, 32, 16),
        no_x = 20,
        no_y = 10,
        activations = act_funs(
            relu,
            leaky_relu = args(negative_slope = 0.01),
            elu = args(alpha = 1.0),
            softshrink = args(lambd = 0.5)
        )
    )

    code_str = paste(deparse(expr), collapse = " ")

    expect_match(code_str, "torch::nnf_relu")
    expect_match(code_str, "torch::nnf_leaky_relu")
    expect_match(code_str, "torch::nnf_elu")
    expect_match(code_str, "torch::nnf_softshrink")
})

test_that("Generated model works in isolated environment", {
    skip_if_not_installed("torch")

    isolated_env = new.env(parent = asNamespace("kindling"))
    expr = ffnn_generator(
        hd_neurons = c(10, 5),
        no_x = 4,
        no_y = 1,
        activations = "relu"
    )

    model_class = eval(expr, envir = isolated_env)

    expect_s3_class(model_class, "nn_module_generator")

    model = model_class()
    expect_s3_class(model, "nn_module")
})

test_that("torch functions not in kindling namespace", {
    kindling_exports = getNamespaceExports("kindling")

    expect_false("nn_module" %in% kindling_exports)
    expect_false("nn_linear" %in% kindling_exports)
    expect_false("nn_lstm" %in% kindling_exports)
    expect_false("nnf_relu" %in% kindling_exports)
    expect_false("nnf_tanh" %in% kindling_exports)
})

test_that("Generated code doesn't require torch in global env", {
    skip_if_not_installed("torch")

    if ("package:torch" %in% search()) {
        detach("package:torch", unload = FALSE)
    }

    expr = ffnn_generator(
        hd_neurons = c(10),
        no_x = 5,
        no_y = 1
    )
    expect_error({
        model_class = eval(expr)
        model = model_class()
    }, NA)
    expect_false("package:torch" %in% search())
})

test_that("RNN types use correct namespace", {
    skip_if_not_installed("torch")

    for (rnn_type in c("rnn", "lstm", "gru")) {
        expr = rnn_generator(
            hd_neurons = c(10),
            no_x = 5,
            no_y = 1,
            rnn_type = rnn_type
        )

        code_str = paste(deparse(expr), collapse = " ")
        expected_call = paste0("torch::nn_", rnn_type)

        expect_match(code_str, expected_call, fixed = TRUE,
                    info = paste("RNN type:", rnn_type))
    }
})

test_that("Bidirectional RNN maintains namespace", {
    skip_if_not_installed("torch")

    expr = rnn_generator(
        hd_neurons = c(20, 10),
        no_x = 5,
        no_y = 1,
        rnn_type = "lstm",
        bidirectional = TRUE
    )

    code_str = paste(deparse(expr), collapse = " ")
    expect_match(code_str, "bidirectional = TRUE")
    expect_match(code_str, "torch::nn_lstm")
})

test_that("Dropout in RNN maintains namespace", {
    skip_if_not_installed("torch")

    expr = rnn_generator(
        hd_neurons = c(30, 20, 10),
        no_x = 5,
        no_y = 1,
        rnn_type = "lstm",
        dropout = 0.3
    )

    code_str = paste(deparse(expr), collapse = " ")
    expect_match(code_str, "dropout")
    expect_match(code_str, "torch::nn_lstm")
})

test_that("No torch imports in generated forward pass", {
    skip_if_not_installed("torch")

    expr = ffnn_generator(
        hd_neurons = c(10, 5),
        no_x = 3,
        no_y = 1,
        activations = c("relu", "elu")
    )

    code_str = paste(deparse(expr), collapse = "\n")
    expect_false(grepl("library\\(torch\\)", code_str))
    expect_false(grepl("require\\(torch\\)", code_str))
    expect_false(grepl("@import torch", code_str))
    expect_false(grepl("@importFrom torch", code_str))
})

test_that("Real model training works without explicit torch load", {
    skip_if_not_installed("torch")
    if ("package:torch" %in% search()) {
        detach("package:torch", unload = FALSE)
    }
    expect_error({
        model = ffnn(
            Sepal.Length ~ Petal.Length + Petal.Width,
            data = iris[1:50, ],
            hidden_neurons = c(10, 5),
            activations = "relu",
            epochs = 5,
            verbose = FALSE
        )
    }, NA)
    expect_s3_class(model, "ffnn_fit")
    expect_false("package:torch" %in% search())
})

test_that("Multiple models don't interfere", {
    skip_if_not_installed("torch")

    expr1 = ffnn_generator(
        nn_name = "Model1",
        hd_neurons = c(10),
        no_x = 5,
        no_y = 1
    )

    expr2 = ffnn_generator(
        nn_name = "Model2",
        hd_neurons = c(20, 10),
        no_x = 8,
        no_y = 3
    )

    model1_class = eval(expr1)
    model2_class = eval(expr2)

    model1 = model1_class()
    model2 = model2_class()

    expect_s3_class(model1, "nn_module")
    expect_s3_class(model2, "nn_module")

    expect_false(identical(model1, model2))
})
