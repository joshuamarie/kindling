skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

test_that("garson method works for ffnn_fit", {
    skip_if_no_torch()
    skip_if_not_installed("NeuralNetTools")

    model = ffnn(
        Species ~ .,
        data = iris,
        hidden_neurons = c(32, 16),
        activations = "relu",
        epochs = 20,
        verbose = FALSE,
        cache_weights = TRUE
    )

    imp = garson(model, bar_plot = FALSE)

    expect_s3_class(imp, c("garson", "data.frame"))
    expect_equal(nrow(imp), 4)
    expect_true(all(c("x_names", "y_names", "rel_imp") %in% names(imp)))
    expect_equal(sum(imp$rel_imp), 100, tolerance = 1e-6)
    expect_true(all(imp$rel_imp >= 0))
})

test_that("olden method works for ffnn_fit", {
    skip_if_no_torch()
    skip_if_not_installed("NeuralNetTools")

    model = ffnn(
        mpg ~ cyl + disp + hp + wt,
        data = mtcars,
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 20,
        loss = "mse",
        verbose = FALSE,
        cache_weights = TRUE
    )

    imp = olden(model, bar_plot = FALSE)

    expect_s3_class(imp, c("olden", "data.frame"))
    expect_equal(nrow(imp), 4)
    expect_true(all(c("x_names", "y_names", "rel_imp") %in% names(imp)))
    expect_true(all(imp$x_names %in% c("cyl", "disp", "hp", "wt")))
})

test_that("vi_model method works with ffnn_fit", {
    skip_if_no_torch()
    skip_if_not_installed("vip")

    model = ffnn(
        Species ~ .,
        data = iris,
        hidden_neurons = c(16),
        activations = "relu",
        epochs = 15,
        verbose = FALSE,
        cache_weights = TRUE
    )

    # Test with olden
    imp_olden = vi_model(model, type = "olden")
    expect_true(all(c("Variable", "Importance") %in% names(imp_olden)))
    expect_equal(nrow(imp_olden), 4)

    # Test with garson
    imp_garson = vi_model(model, type = "garson")
    expect_equal(nrow(imp_garson), 4)
    expect_true(all(imp_garson$Importance >= 0))
})

test_that("vip integration works", {
    skip_if_no_torch()
    skip_if_not_installed("vip")

    model = ffnn(
        Species ~ Sepal.Length + Sepal.Width,
        data = iris,
        hidden_neurons = c(8),
        activations = "relu",
        epochs = 10,
        verbose = FALSE,
        cache_weights = TRUE
    )

    imp = vip::vi(model)

    expect_equal(nrow(imp), 2)
    expect_true("Variable" %in% names(imp))
    expect_true("Importance" %in% names(imp))
})

test_that("variable importance works without cached weights", {
    skip_if_no_torch()
    skip_if_not_installed("NeuralNetTools")

    model = ffnn(
        Species ~ .,
        data = iris,
        hidden_neurons = c(16),
        activations = "relu",
        epochs = 10,
        verbose = FALSE,
        cache_weights = FALSE
    )

    expect_null(model$cached_weights)

    imp = garson(model, bar_plot = FALSE)
    expect_s3_class(imp, "garson")
    expect_equal(nrow(imp), 4)
})

test_that("variable importance handles multi-layer networks", {
    skip_if_no_torch()
    skip_if_not_installed("NeuralNetTools")

    model = ffnn(
        Species ~ .,
        data = iris,
        hidden_neurons = c(32, 16, 8),
        activations = "relu",
        epochs = 15,
        verbose = FALSE,
        cache_weights = TRUE
    )

    imp_garson = garson(model, bar_plot = FALSE)
    imp_olden = olden(model, bar_plot = FALSE)

    expect_equal(nrow(imp_garson), 4)
    expect_equal(nrow(imp_olden), 4)
    expect_equal(sum(imp_garson$rel_imp), 100, tolerance = 1e-6)
})
