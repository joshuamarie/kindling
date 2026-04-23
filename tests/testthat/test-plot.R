skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

iris_x = as.matrix(iris[, 2:4])
iris_y = iris$Sepal.Length
iris_cls_x = as.matrix(iris[, 1:4])
iris_cls_y = iris$Species

make_reg_ds = function() {
    torch::dataset(
        initialize = function() {
            self$x = torch::torch_tensor(
                as.matrix(iris[, 2:4]),
                dtype = torch::torch_float32()
            )
            self$y = torch::torch_tensor(
                as.matrix(iris[, 1, drop = FALSE]),
                dtype = torch::torch_float32()
            )
        },
        .getitem = function(i) list(self$x[i, ], self$y[i]),
        .length = function() self$x$size(1)
    )()
}

test_that("autoplot() returns a ggplot for a basic fit (no validation)", {
    skip_if_no_torch()
    skip_if_not_installed("ggplot2")
    m = train_nn(iris_x, iris_y, epochs = 5)
    p = ggplot2::autoplot(m)
    expect_s3_class(p, "ggplot")
})

test_that("autoplot() returns a ggplot for a fit with validation_split", {
    skip_if_no_torch()
    skip_if_not_installed("ggplot2")
    m = train_nn(iris_x, iris_y, epochs = 5, validation_split = 0.2)
    p = ggplot2::autoplot(m)
    expect_s3_class(p, "ggplot")
})

test_that("autoplot() includes a geom_vline layer when early stopping fires", {
    skip_if_no_torch()
    skip_if_not_installed("ggplot2")
    es = early_stop(patience = 2, min_delta = 1e10, monitor = "val_loss")
    m = train_nn(iris_x, iris_y, epochs = 50, validation_split = 0.2, early_stopping = es)
    expect_false(is.na(m$stopped_epoch))
    p = ggplot2::autoplot(m)
    layer_classes = vapply(p$layers, function(l) class(l$geom)[1], character(1))
    expect_true("GeomVline" %in% layer_classes)
})

test_that("plot.nn_fit() returns the fit object invisibly", {
    skip_if_no_torch()
    skip_if_not_installed("ggplot2")
    m = train_nn(iris_x, iris_y, epochs = 5)
    pdf(NULL)
    on.exit(dev.off(), add = TRUE)
    result = plot(m)
    expect_identical(result, m)
})

# ---- autoplot_diagnostics() ----

test_that("autoplot_diagnostics() returns a named list of ggplots for regression", {
    skip_if_no_torch()
    skip_if_not_installed("ggplot2")
    m = train_nn(iris_x, iris_y, epochs = 5)
    result = suppressMessages(autoplot_diagnostics(m, actual = iris_y))
    expect_type(result, "list")
    expect_named(result, c("residuals_vs_fitted", "actual_vs_fitted"))
    expect_s3_class(result$residuals_vs_fitted, "ggplot")
    expect_s3_class(result$actual_vs_fitted, "ggplot")
})

test_that("autoplot_diagnostics() returns a ggplot for classification", {
    skip_if_no_torch()
    skip_if_not_installed("ggplot2")
    m = train_nn(iris_cls_x, iris_cls_y, epochs = 5)
    p = autoplot_diagnostics(m, actual = iris_cls_y)
    expect_s3_class(p, "ggplot")
})

test_that("autoplot_diagnostics() errors cleanly on length mismatch", {
    skip_if_no_torch()
    skip_if_not_installed("ggplot2")
    m = train_nn(iris_x, iris_y, epochs = 5)
    expect_error(
        autoplot_diagnostics(m, actual = iris_y[1:10]),
        class = "rlang_error"
    )
})

test_that("autoplot_diagnostics() errors cleanly for nn_fit_ds (no fitted values)", {
    skip_if_no_torch()
    skip_if_not_installed("ggplot2")
    ds = make_reg_ds()
    m_ds = train_nn(ds, hidden_neurons = 8L, epochs = 5)
    expect_error(
        autoplot_diagnostics(m_ds, actual = iris_y),
        class = "rlang_error"
    )
})

test_that("plot_diagnostics() returns the fit object invisibly (regression)", {
    skip_if_no_torch()
    skip_if_not_installed("ggplot2")
    m = train_nn(iris_x, iris_y, epochs = 5)
    pdf(NULL)
    on.exit(dev.off(), add = TRUE)
    result = suppressMessages(plot_diagnostics(m, actual = iris_y))
    expect_identical(result, m)
})
