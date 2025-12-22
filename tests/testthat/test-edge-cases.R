test_that("mlp_kindling with multiple activation functions works", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = c(20, 10),
        activations = c("relu", "elu"),
        epochs = 5,
        verbose = FALSE
    )

    fitted = parsnip::fit(
        spec,
        Species ~ .,
        data = iris[1:100, ]
    )

    expect_s3_class(fitted, "model_fit")

    preds = predict(fitted, new_data = iris[101:110, ])
    expect_equal(nrow(preds), 10)
})

test_that("mlp_kindling handles single hidden layer and accepts using `list()`", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = list(20),
        epochs = 5,
        verbose = FALSE
    )

    fitted = parsnip::fit(spec, Species ~ ., data = iris[1:100, ])
    preds = predict(fitted, new_data = iris[101:110, ])

    expect_equal(nrow(preds), 10)
})

test_that("mlp_kindling handles deep networks", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    spec = mlp_kindling(
        mode = "regression",
        hidden_neurons = c(64, 32, 16, 8),
        activations = "relu",
        epochs = 5,
        verbose = FALSE
    )

    expect_error({
        fitted = parsnip::fit(
            spec,
            Sepal.Length ~ .,
            data = iris[1:100, ]
        )
    }, NA)
})

test_that("mlp_kindling handles deep neural networks and accepts both using `list()` and a stringed argument for the activation function", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = list(5, 10, 7),
        activations = list('relu', 'softshrink(lambd = 0.5)', 'celu(alpha = 0.8)'),
        epochs = 5,
        verbose = FALSE
    )

    fitted = parsnip::fit(spec, Species ~ ., data = iris[1:100, ])
    preds = predict(fitted, new_data = iris[101:110, ])

    expect_no_warning(fitted)
    expect_no_error(fitted)
    expect_no_warning(preds)
    expect_no_error(preds)
    expect_equal(nrow(preds), 10)
})

test_that("predictions work with single observation", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = 10,
        epochs = 5,
        verbose = FALSE
    )

    fitted = parsnip::fit(spec, Species ~ ., data = iris[1:100, ])
    preds = predict(fitted, new_data = iris[101, ])

    expect_equal(nrow(preds), 1)
    expect_s3_class(preds$.pred_class, "factor")
})

test_that("augment method works correctly", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = list(10),
        epochs = 5,
        verbose = FALSE
    )

    fitted = parsnip::fit(spec, Species ~ ., data = iris[1:100, ])
    augmented = parsnip::augment(fitted, new_data = iris[101:110, ])

    expect_s3_class(augmented, "tbl_df")
    expect_equal(nrow(augmented), 10)
    expect_true(".pred_class" %in% names(augmented))
    expect_true("Species" %in% names(augmented))
    expect_true("Sepal.Length" %in% names(augmented))
})
