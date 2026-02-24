skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

# ---- torch dataset ----

make_iris_dataset = function() {
    torch::dataset(
        initialize = function() {
            self$x = torch::torch_tensor(
                as.matrix(iris[, 1:4]),
                dtype = torch::torch_float32()
            )
            self$y = torch::torch_tensor(
                as.integer(iris$Species),
                dtype = torch::torch_long()
            )
        },
        .getitem = function(i) list(self$x[i, ], self$y[i]),
        .length = function() self$x$size(1)
    )()
}

make_reg_dataset = function() {
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

# ---- More for nn_arch() ----

test_that("nn_arch() returns correct classes and defaults", {
    arch = nn_arch()
    expect_s3_class(arch, "nn_arch")
    expect_s3_class(arch, "kindling_arch")
    expect_equal(arch$nn_name, "nnModule")
    expect_null(arch$nn_layer)
    expect_null(arch$out_nn_layer)
    expect_null(arch$input_transform)
    expect_true(is.environment(attr(arch, "env")))
})

test_that("nn_arch() stores all supplied arguments", {
    arch = nn_arch(
        nn_name = "MyGRU",
        nn_layer = "torch::nn_gru",
        out_nn_layer = "torch::nn_linear",
        nn_layer_args = list(batch_first = TRUE),
        input_transform = ~ .$unsqueeze(2)
    )
    expect_equal(arch$nn_name, "MyGRU")
    expect_equal(arch$nn_layer, "torch::nn_gru")
    expect_equal(arch$out_nn_layer, "torch::nn_linear")
    expect_equal(arch$nn_layer_args, list(batch_first = TRUE))
    expect_false(is.null(arch$input_transform))
})

test_that("nn_arch() captures caller environment", {
    my_env = environment()
    arch = nn_arch()
    expect_true(is.environment(attr(arch, "env")))
})

# ---- train_nn.dataset() ----

test_that("train_nn.dataset() trains a classification model", {
    skip_if_no_torch()
    ds = make_iris_dataset()
    m = train_nn(ds, hidden_neurons = c(16L, 8L), activations = "relu",
                 epochs = 5, n_classes = 3)
    expect_s3_class(m, "nn_fit_ds")
    expect_s3_class(m, "nn_fit")
    expect_true(m$is_classification)
    expect_equal(m$n_classes, 3L)
    expect_length(m$loss_history, 5)
})

test_that("train_nn.dataset() trains a regression model", {
    skip_if_no_torch()
    ds = make_reg_dataset()
    m = train_nn(ds, hidden_neurons = 16L, activations = "relu", epochs = 5)
    expect_s3_class(m, "nn_fit_ds")
    expect_false(m$is_classification)
})

test_that("train_nn.dataset() auto-switches loss to cross_entropy", {
    skip_if_no_torch()
    ds = make_iris_dataset()
    m = train_nn(ds, epochs = 5, n_classes = 3, loss = "mse")
    expect_s3_class(m, "nn_fit_ds")
})

test_that("train_nn.dataset() errors without n_classes for classification", {
    skip_if_no_torch()
    ds = make_iris_dataset()
    expect_error(train_nn(ds, epochs = 5), class = "rlang_error")
})

test_that("train_nn.dataset() warns when y is supplied", {
    skip_if_no_torch()
    ds = make_iris_dataset()
    expect_warning(
        train_nn(ds, y = 1:150, epochs = 5, n_classes = 3),
        class = "rlang_warning"
    )
})

test_that("train_nn.dataset() supports validation_split", {
    skip_if_no_torch()
    ds = make_reg_dataset()
    m = train_nn(ds, epochs = 5, validation_split = 0.2)
    expect_length(m$val_loss_history, 5)
})

test_that("train_nn.dataset() supports cache_weights", {
    skip_if_no_torch()
    ds = make_reg_dataset()
    m = train_nn(ds, epochs = 5, cache_weights = TRUE)
    expect_type(m$cached_weights, "list")
})

test_that("predict.nn_fit_ds() works with a dataset", {
    skip_if_no_torch()
    ds = make_iris_dataset()
    m = train_nn(ds, hidden_neurons = 16L, epochs = 5, n_classes = 3)
    preds = predict(m, newdata = ds)
    expect_s3_class(preds, "factor")
    expect_length(preds, 150)
})

test_that("predict.nn_fit_ds() type = 'prob' returns valid probability matrix", {
    skip_if_no_torch()
    ds = make_iris_dataset()
    m = train_nn(ds, hidden_neurons = 16L, epochs = 5, n_classes = 3)
    probs = predict(m, newdata = ds, type = "prob")
    expect_true(is.matrix(probs))
    expect_equal(ncol(probs), 3L)
    expect_equal(rowSums(probs), rep(1, 150), tolerance = 1e-5)
})

test_that("predict.nn_fit_ds() works with a matrix as newdata", {
    skip_if_no_torch()
    ds = make_reg_dataset()
    m = train_nn(ds, epochs = 5)
    preds = predict(m, newdata = as.matrix(iris[, 2:4]))
    expect_length(preds, 150)
})

test_that("predict.nn_fit_ds() errors when newdata is NULL", {
    skip_if_no_torch()
    ds = make_reg_dataset()
    m = train_nn(ds, epochs = 5)
    expect_error(predict(m), class = "rlang_error")
})

test_that("predict.nn_fit_ds() errors on type = 'prob' for regression", {
    skip_if_no_torch()
    ds = make_reg_dataset()
    m = train_nn(ds, epochs = 5)
    expect_error(
        predict(m, newdata = make_reg_dataset(), type = "prob"),
        class = "rlang_error"
    )
})

test_that("predict.nn_fit_ds() errors on invalid type", {
    skip_if_no_torch()
    ds = make_iris_dataset()
    m = train_nn(ds, epochs = 5, n_classes = 3)
    expect_error(
        predict(m, newdata = ds, type = "bad"),
        class = "rlang_error"
    )
})

test_that("train_nn.dataset() with nn_arch and flatten_input = FALSE", {
    skip_if_no_torch()
    gru_arch = nn_arch(
        nn_name = "GRU",
        nn_layer = "torch::nn_gru",
        layer_arg_fn = ~ if (.is_output) {
            list(.in, .out)
        } else {
            list(input_size = .in, hidden_size = .out, batch_first = TRUE)
        },
        out_nn_layer = "torch::nn_linear",
        forward_extract = ~ .[[1]],
        before_output_transform = ~ .[, .$size(2), ],
        input_transform = ~ .$unsqueeze(2)
    )
    ds = make_reg_dataset()
    m = train_nn(ds, hidden_neurons = 16L, epochs = 3,
                 architecture = gru_arch, flatten_input = FALSE)
    expect_s3_class(m, "nn_fit_ds")
})

test_that("train_nn.dataset() errors with flatten_input = FALSE and no arch", {
    skip_if_no_torch()
    ds = make_reg_dataset()
    expect_error(
        train_nn(ds, epochs = 3, flatten_input = FALSE),
        class = "rlang_error"
    )
})
