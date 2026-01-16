skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

test_that("classification metrics work with mlp predictions", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("yardstick")
    skip_if_no_torch()

    spec = mlp_kindling(
        mode = "classification",
        hidden_neurons = c(20),
        epochs = 10,
        verbose = FALSE
    )

    fitted = parsnip::fit(
        spec,
        Species ~ .,
        data = iris
    )

    results = parsnip::augment(fitted, new_data = iris)

    metrics = yardstick::metrics(
        results,
        truth = Species,
        estimate = .pred_class
    )

    expect_s3_class(metrics, "tbl_df")
    expect_true("accuracy" %in% metrics$.metric)
    expect_true("kap" %in% metrics$.metric)
})

test_that("regression metrics work with mlp predictions", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("yardstick")
    skip_if_no_torch()

    spec = mlp_kindling(
        mode = "regression",
        hidden_neurons = c(20),
        epochs = 10,
        verbose = FALSE
    )

    fitted = parsnip::fit(
        spec,
        Sepal.Length ~ .,
        data = iris
    )

    results = parsnip::augment(fitted, new_data = iris)

    metrics = yardstick::metrics(
        results,
        truth = Sepal.Length,
        estimate = .pred
    )

    expect_s3_class(metrics, "tbl_df")
    expect_true("rmse" %in% metrics$.metric)
    expect_true("rsq" %in% metrics$.metric)
    expect_true("mae" %in% metrics$.metric)
})
