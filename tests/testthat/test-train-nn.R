skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

iris_x = as.matrix(iris[, 2:4])
iris_y = iris$Sepal.Length
iris_cls_x = as.matrix(iris[, 1:4])
iris_cls_y = iris$Species

test_that("train_nn() dispatches on matrix, data.frame, and formula", {
    skip_if_no_torch()
    expect_s3_class(
        train_nn(iris_x, iris_y, epochs = 5),
        "nn_fit"
    )
    expect_s3_class(
        train_nn(iris[, 2:4], iris_y, epochs = 5),
        c("nn_fit_tab", "nn_fit"), exact = TRUE
    )
    expect_s3_class(
        train_nn(Sepal.Length ~ ., data = iris[, 1:4], epochs = 5),
        c("nn_fit_tab", "nn_fit"), exact = TRUE
    )
})

# ---- Classification -----

test_that("train_nn() handles classification correctly", {
    skip_if_no_torch()
    m = train_nn(iris_cls_x, iris_cls_y, epochs = 5)
    expect_true(m$is_classification)
    expect_equal(m$y_levels, levels(iris_cls_y))
    expect_s3_class(m$fitted, "factor")
    expect_equal(levels(m$fitted), levels(iris_cls_y))
})

test_that("train_nn() errors on unsupported input types", {
    skip_if_no_torch()
    expect_error(train_nn("not valid"), class = "rlang_error")
    expect_error(train_nn(Sepal.Length ~ ., data = NULL), class = "rlang_error")
    expect_error(
        train_nn(iris_x, iris_y, arch = list(nn_name = "bad"), epochs = 5),
        class = "rlang_error"
    )
})

test_that("train_nn() return object has correct structure", {
    skip_if_no_torch()
    m = train_nn(iris_x, iris_y, epochs = 10, validation_split = 0.2)
    expect_named(m, c(
        "model", "fitted", "loss_history", "val_loss_history",
        "n_epochs", "stopped_epoch", "hidden_neurons", "activations",
        "output_activation", "penalty", "mixture", "feature_names",
        "response_name", "no_x", "no_y", "is_classification",
        "y_levels", "n_classes", "device", "cached_weights", "arch"
    ), ignore.order = TRUE)
    expect_length(m$loss_history, 10)
    expect_length(m$val_loss_history, 10)
    expect_equal(m$no_x, ncol(iris_x))
    expect_null(m$cached_weights)
    expect_true(is.na(m$stopped_epoch))
})

test_that("cache_weights stores weight matrices when TRUE", {
    skip_if_no_torch()
    m = train_nn(iris_x, iris_y, epochs = 5, cache_weights = TRUE)
    expect_type(m$cached_weights, "list")
})

test_that("train_nn() accepts various act_funs() syntaxes", {
    skip_if_no_torch()
    expect_no_error(
        train_nn(
            iris_x, 
            iris_y, 
            hidden_neurons = c(16, 8),
            activations = act_funs(relu, ), 
            epochs = 5
        )
    )
    expect_no_error(
        train_nn(
            iris_x, 
            iris_y, 
            hidden_neurons = 16,
            activations = act_funs(elu[alpha = 0.5]), 
            epochs = 5
        )
    )
    expect_no_error(
        train_nn(
            iris_x, 
            iris_y, 
            hidden_neurons = 16,
            activations = act_funs(new_act_fn(\(x) torch::torch_tanh(x))),
            epochs = 5
        )
    )
})

test_that("act_funs() errors on bad activation specs", {
    skip_if_no_torch()
    expect_error(act_funs(not_a_real_fn), class = "activation_not_found_error")
    expect_error(act_funs(relu[bad_param = 1]), class = "purrr_error_indexed")
})

# ---- Loss functions ----

test_that("train_nn() accepts built-in and custom loss functions", {
    skip_if_no_torch()
    expect_no_error(train_nn(iris_x, iris_y, loss = "mae", epochs = 5))
    expect_no_error(
        train_nn(
            iris_x, 
            iris_y,
            loss = \(input, target) torch::nnf_mse_loss(input, target),
            epochs = 5
        )
    )
    expect_error(train_nn(iris_x, iris_y, loss = "not_a_loss", epochs = 5))
    expect_error(
        train_nn(iris_x, iris_y, loss = \(input, target) 42, epochs = 5),
        class = "loss_fn_output_error"
    )
    expect_error(
        train_nn(iris_x, iris_y, loss = \(x) torch::nnf_mse_loss(x, x), epochs = 5),
        class = "loss_fn_arity_error"
    )
})

# ---- Early stopping ----

describe("train_nn() early stopping", {
    it("runs cleanly and trims loss_history when triggered", {
        skip_if_no_torch()
        # min_delta = 1e10 forces early stopping to fire reliably
        es = early_stop(patience = 2, min_delta = 1e10, monitor = "val_loss")
        m = train_nn(iris_x, iris_y, epochs = 50,
                     validation_split = 0.2, early_stopping = es)
        expect_lt(length(m$loss_history), 50)
        expect_false(is.na(m$stopped_epoch))
    })
    
    it("errors when val_loss monitor is used without validation_split", {
        skip_if_no_torch()
        expect_error(
            train_nn(
                iris_x, 
                iris_y, 
                epochs = 10,
                early_stopping = early_stop(patience = 3, monitor = "val_loss")
            ),
            class = "rlang_error"
        )
    })
    
    it("errors when early_stopping is not an early_stop_spec", {
        skip_if_no_torch()
        expect_error(
            train_nn(iris_x, iris_y, epochs = 5, early_stopping = list(patience = 5)),
            class = "rlang_error"
        )
    })
})

test_that("predict.nn_fit() returns correct output types", {
    skip_if_no_torch()
    m_reg = train_nn(iris_x, iris_y, epochs = 5)
    m_cls = train_nn(iris_cls_x, iris_cls_y, epochs = 5)
    
    expect_equal(predict(m_reg), m_reg$fitted)
    expect_type(predict(m_reg, newdata = iris_x), "double")
    expect_s3_class(predict(m_cls, newdata = iris_cls_x), "factor")
    
    probs = predict(m_cls, newdata = iris_cls_x, type = "prob")
    expect_true(is.matrix(probs))
    expect_equal(rowSums(probs), rep(1, nrow(iris_cls_x)), tolerance = 1e-5)
    
    expect_error(predict(m_reg, newdata = iris_x, type = "prob"), class = "rlang_error")
    expect_error(predict(m_reg, newdata = iris_x, type = "bad"), class = "rlang_error")
})

test_that("new_data is accepted as alias for newdata", {
    skip_if_no_torch()
    m = train_nn(iris_x, iris_y, epochs = 5)
    expect_equal(predict(m, newdata = iris_x), predict(m, new_data = iris_x))
})

test_that("train_nn() handles edge case inputs", {
    skip_if_no_torch()
    m = train_nn(iris_x, iris_y, epochs = 5)
    expect_length(predict(m, newdata = iris_x[1, , drop = FALSE]), 1)
    
    expect_no_error(
        train_nn(iris_x[1:10, ], iris_y[1:10], batch_size = 50, epochs = 5)
    )
    
    expect_length(train_nn(iris_x, iris_y, epochs = 1)$loss_history, 1)
    
    m_multi = train_nn(as.matrix(iris[, 3:4]), as.matrix(iris[, 1:2]), epochs = 5)
    expect_equal(m_multi$no_y, 2L)
})
