skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

# ---- ordinal_gen() ----

test_that("ordinal_gen() handles standard cases", {
    expect_equal(ordinal_gen(1), "1st")
    expect_equal(ordinal_gen(2), "2nd")
    expect_equal(ordinal_gen(3), "3rd")
    expect_equal(ordinal_gen(4), "4th")
})

test_that("ordinal_gen() handles teens (11th, 12th, 13th)", {
    expect_equal(ordinal_gen(11), "11th")
    expect_equal(ordinal_gen(12), "12th")
    expect_equal(ordinal_gen(13), "13th")
})

test_that("ordinal_gen() warns on values below 1", {
    expect_warning(ordinal_gen(0), "Values below 1")
})

# ---- print.nn_fit() ----

test_that("print.nn_fit() runs without error for regression", {
    skip_if_no_torch()
    fit = train_nn(
        x = Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 3
    )
    expect_output(print(fit))
})

test_that("print.nn_fit() runs without error for classification", {
    skip_if_no_torch()
    fit = train_nn(
        x = Species ~ .,
        data = iris,
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 3
    )
    expect_output(print(fit))
})

test_that("print.nn_fit() shows regularization info when penalty > 0", {
    skip_if_no_torch()
    fit = train_nn(
        x = Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = 16L,
        activations = "relu",
        epochs = 3,
        penalty = 0.01,
        mixture = 0.5
    )
    expect_output(print(fit), "\u03BB")
})

test_that("print.nn_fit() handles NULL hidden_neurons (no hidden layers)", {
    skip_if_no_torch()
    fit = train_nn(
        x = Sepal.Length ~ .,
        data = iris[, 1:4],
        epochs = 3
    )
    expect_output(print(fit), "Number of Hidden Layers\\s*:\\s*0")
})

test_that("print.nn_fit() shows arch info when nn_arch is provided", {
    skip_if_no_torch()
    arch = nn_arch(nn_name = "mlp_model")
    fit = train_nn(
        x = Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = c(16, 8),
        activations = "relu",
        architecture = arch,
        epochs = 3
    )
    expect_output(print(fit), "mlp_model")
})

# ---- print.ffnn_fit() ----

test_that("print.ffnn_fit() runs without error for regression", {
    skip_if_no_torch()
    fit = ffnn(
        Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 3
    )
    expect_output(print(fit), "Feedforward Neural Network")
})

test_that("print.ffnn_fit() runs without error for classification", {
    skip_if_no_torch()
    fit = ffnn(
        Species ~ .,
        data = iris,
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 3
    )
    expect_output(print(fit), "classification")
})

test_that("print.ffnn_fit() shows regularization when penalty > 0", {
    skip_if_no_torch()
    fit = ffnn(
        Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = 16L,
        activations = "relu",
        epochs = 3,
        penalty = 0.01,
        mixture = 0.5
    )
    expect_output(print(fit), "\u03BB")
})

test_that("print.ffnn_fit() shows 'Not specified' when hidden_neurons is NULL", {
    skip_if_no_torch()
    fit = ffnn(
        Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = 8,
        epochs = 3
    )
    fit$hidden_neurons = NULL
    expect_output(print(fit), "Not specified")
})

# ---- print.rnn_fit() ----

test_that("print.rnn_fit() runs without error for lstm regression", {
    skip_if_no_torch()
    fit = rnn(
        Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = c(16, 8),
        rnn_type = "lstm",
        activations = "relu",
        epochs = 3
    )
    expect_output(print(fit), "LSTM")
})

test_that("print.rnn_fit() runs without error for gru", {
    skip_if_no_torch()
    fit = rnn(
        Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = 16L,
        rnn_type = "gru",
        epochs = 3
    )
    expect_output(print(fit), "GRU")
})

test_that("print.rnn_fit() runs without error for plain rnn", {
    skip_if_no_torch()
    fit = rnn(
        Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = 16L,
        rnn_type = "rnn",
        epochs = 3
    )
    expect_output(print(fit), "Recurrent Neural Network")
})

test_that("print.rnn_fit() shows bidirectional status", {
    skip_if_no_torch()
    fit_bi = rnn(
        Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = 16L,
        rnn_type = "lstm",
        bidirectional = TRUE,
        epochs = 3
    )
    fit_uni = rnn(
        Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = 16L,
        rnn_type = "lstm",
        bidirectional = FALSE,
        epochs = 3
    )
    expect_output(print(fit_bi),  "Yes")
    expect_output(print(fit_uni), "No")
})

test_that("print.rnn_fit() shows regularization when penalty > 0", {
    skip_if_no_torch()
    fit = rnn(
        Sepal.Length ~ .,
        data = iris[, 1:4],
        hidden_neurons = 16L,
        rnn_type = "lstm",
        epochs = 3,
        penalty = 0.01,
        mixture = 0.5
    )
    expect_output(print(fit), "\u03BB")
})
