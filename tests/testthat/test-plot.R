skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

iris_x = as.matrix(iris[, 2:4])
iris_y = iris$Sepal.Length

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
    result = plot(m)
    dev.off()
    expect_identical(result, m)
})
