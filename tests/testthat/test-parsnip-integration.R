test_that("mlp_kindling works with parsnip fit - regression", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    spec = mlp_kindling(
        mode = "regression",
        hidden_neurons = c(10, 5),
        activations = "relu",
        epochs = 5,
        verbose = FALSE
    )

    expect_error({
        fitted = parsnip::fit(
            spec,
            Sepal.Length ~ Petal.Length + Petal.Width,
            data = iris[1:50, ]
        )
    }, NA)

    expect_s3_class(fitted, "model_fit")
    expect_s3_class(fitted$fit, "ffnn_fit")
})

test_that("mlp_kindling works with parsnip fit - classification", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = c(10, 5),
        activations = "relu",
        epochs = 5,
        verbose = FALSE
    )

    expect_error({
        fitted = parsnip::fit(
            spec,
            Species ~ .,
            data = iris[1:50, ]
        )
    }, NA)

    expect_s3_class(fitted, "model_fit")
    expect_s3_class(fitted$fit, "ffnn_fit")
})

test_that("rnn_kindling works with parsnip fit - classification", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    spec = rnn_kindling(
        mode = "classification",
        hidden_neurons = c(10),
        rnn_type = "lstm",
        activations = "relu",
        epochs = 5,
        verbose = FALSE
    )

    expect_error({
        fitted = parsnip::fit(
            spec,
            Species ~ .,
            data = iris[1:50, ]
        )
    }, NA)

    expect_s3_class(fitted, "model_fit")
})

test_that("mlp predictions return correct format - regression", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    spec = mlp_kindling(
        mode = "regression",
        hidden_neurons = c(10),
        epochs = 5,
        verbose = FALSE
    )

    fitted = parsnip::fit(
        spec,
        Sepal.Length ~ Petal.Length + Petal.Width,
        data = iris[1:50, ]
    )

    preds = predict(fitted, new_data = iris[51:60, ])

    expect_s3_class(preds, "tbl_df")
    expect_true(".pred" %in% names(preds))
    expect_equal(nrow(preds), 10)
    expect_type(preds$.pred, "double")
})

test_that("mlp predictions return correct format - classification class", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = c(10),
        epochs = 5,
        verbose = FALSE
    )

    fitted = parsnip::fit(
        spec,
        Species ~ .,
        data = iris[1:100, ]
    )

    preds = predict(fitted, new_data = iris[101:110, ])

    expect_s3_class(preds, "tbl_df")
    expect_true(".pred_class" %in% names(preds))
    expect_equal(nrow(preds), 10)
    expect_s3_class(preds$.pred_class, "factor")
})

test_that("'tanh' should still throws an error from `activations` in `mlp_kindling()`", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    expect_error({
        mlp_kindling(
            mode = "classification",
            hidden_neurons = 15,
            activations = act_funs(tanh),
            epochs = 5,
            verbose = FALSE
        ) |>
        parsnip::fit(Species ~ ., data = iris)
    })
})

test_that("mlp predictions return correct format - classification prob", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("torch")

    spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = c(10),
        epochs = 5,
        verbose = FALSE
    )

    fitted = parsnip::fit(
        spec,
        Species ~ .,
        data = iris[1:100, ]
    )

    preds = predict(fitted, new_data = iris[101:110, ], type = "prob")

    expect_s3_class(preds, "tbl_df")
    expect_equal(nrow(preds), 10)
    expect_true(all(grepl("^\\.pred_", names(preds))))
    expect_true(all(abs(rowSums(preds) - 1) < 1e-6))
})
