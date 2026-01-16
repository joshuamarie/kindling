test_that("mlp_kindling creates valid model specification", {
    skip_if_not_installed("parsnip")

    spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = c(64, 32),
        activations = "relu",
        epochs = 50
    )

    expect_s3_class(spec, "mlp_kindling")
    expect_s3_class(spec, "model_spec")
    expect_equal(spec$mode, "classification")
    expect_equal(spec$engine, "kindling")
})

test_that("mlp_kindling defaults to unknown mode", {
    skip_if_not_installed("parsnip")

    spec = mlp_kindling()

    expect_equal(spec$mode, "unknown")
    expect_false(spec$user_specified_mode)
})

test_that("mlp_kindling accepts all valid parameters", {
    skip_if_not_installed("parsnip")

    spec = mlp_kindling(
        mode = "regression",
        hidden_neurons = c(128, 64, 32),
        activations = c("relu", "elu", "selu"),
        output_activation = "sigmoid",
        bias = TRUE,
        epochs = 100,
        batch_size = 32,
        learn_rate = 0.001,
        optimizer = "adam",
        loss = "mse",
        validation_split = 0.2,
        device = "cpu",
        verbose = FALSE
    )

    expect_s3_class(spec, "mlp_kindling")
    expect_equal(spec$mode, "regression")
})

test_that("mlp_kindling print method works", {
    skip_if_not_installed("parsnip")

    spec = mlp_kindling(mode = "classification")

    expect_output(print(spec), "Kindling Multi-Layer Perceptron")
    expect_output(print(spec), "classification")
})

test_that("mlp_kindling update method works", {
    skip_if_not_installed("parsnip")

    spec = mlp_kindling(
        mode = "classification",
        epochs = 50
    )

    updated_spec = update(spec, epochs = 100, learn_rate = 0.01)

    expect_s3_class(updated_spec, "mlp_kindling")
})

test_that("mlp_kindling update method works", {
    skip_if_not_installed("parsnip")

    spec = mlp_kindling(
        mode = "classification",
        epochs = 50
    )

    updated_spec = update(spec, epochs = 100, learn_rate = 0.01)

    expect_s3_class(updated_spec, "mlp_kindling")
})

test_that("mlp_kindling supports tune placeholders", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("tune")

    spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = tune::tune(),
        epochs = tune::tune(),
        learn_rate = tune::tune()
    )

    expect_s3_class(spec, "mlp_kindling")
})

