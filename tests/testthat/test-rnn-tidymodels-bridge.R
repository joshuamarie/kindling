test_that("rnn_kindling creates valid model specification", {
    skip_if_not_installed("parsnip")

    spec = rnn_kindling(
        mode = "classification",
        hidden_neurons = c(64, 32),
        rnn_type = "lstm",
        activations = "relu",
        epochs = 50
    )

    expect_s3_class(spec, "rnn_kindling")
    expect_s3_class(spec, "model_spec")
    expect_equal(spec$mode, "classification")
    expect_equal(spec$engine, "kindling")
})

test_that("rnn_kindling accepts all RNN-specific parameters", {
    skip_if_not_installed("parsnip")

    spec = rnn_kindling(
        mode = "regression",
        hidden_neurons = c(128, 64),
        rnn_type = "gru",
        activations = c("relu", "elu"),
        bidirectional = TRUE,
        dropout = 0.3,
        epochs = 100
    )

    expect_s3_class(spec, "rnn_kindling")
    expect_equal(spec$mode, "regression")
})

test_that("rnn_kindling print method works", {
    skip_if_not_installed("parsnip")

    spec = rnn_kindling(mode = "classification", rnn_type = "lstm")

    expect_output(print(spec), "Recurrent Neural Network")
    expect_output(print(spec), "classification")
})

test_that("rnn_kindling update method works", {
    skip_if_not_installed("parsnip")

    spec = rnn_kindling(
        mode = "classification",
        rnn_type = "lstm",
        epochs = 50
    )

    updated_spec = update(spec, epochs = 100, dropout = 0.2)

    expect_s3_class(updated_spec, "rnn_kindling")
})
