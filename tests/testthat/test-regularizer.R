skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

test_that("FFNN regularization parameters are validated", {
    skip_if_no_torch()
    
    # Negative penalty should error
    expect_error(
        ffnn(
            Sepal.Length ~ ., 
            data = iris[1:50, ], 
            hidden_neurons = 10, 
            penalty = -0.01, 
            epochs = 5
        ),
        "non-negative"
    )
    
    # Invalid mixture (> 1)
    expect_error(
        ffnn(
            Sepal.Length ~ ., 
            data = iris[1:50, ], 
            hidden_neurons = 10, 
            penalty = 0.01, 
            mixture = 1.5, 
            epochs = 5
        ),
        "between 0 and 1"
    )
    
    # Invalid mixture (< 0)
    expect_error(
        ffnn(
            Sepal.Length ~ ., 
            data = iris[1:50, ], 
            hidden_neurons = 10, 
            penalty = 0.01, 
            mixture = -0.1, 
            epochs = 5
        ),
        "between 0 and 1"
    )
})

test_that("FFNN trains with L1 regularization (Lasso)", {
    skip_if_no_torch()
    
    data = iris[1:50, 1:4]
    
    # Pure L1 (LASSO regression)
    model = ffnn(
        Sepal.Length ~ ., 
        data = data, 
        hidden_neurons = c(16, 8), 
        penalty = 0.01,
        mixture = 1,
        epochs = 10,
        verbose = FALSE
    )
    
    expect_s3_class(model, "ffnn_fit")
    expect_equal(model$penalty, 0.01)
    expect_equal(model$mixture, 1)
    expect_length(model$fitted, nrow(data))
})

test_that("FFNN trains with L2 regularization (Ridge)", {
    skip_if_no_torch()
    
    # Pure L2 (Ridge regression)
    model = ffnn(
        Sepal.Length ~ ., 
        data = iris[1:50, 1:4], 
        hidden_neurons = 16, 
        penalty = 0.001,
        mixture = 0,      
        epochs = 10,
        verbose = FALSE
    )
    
    expect_s3_class(model, "ffnn_fit")
    expect_equal(model$penalty, 0.001)
    expect_equal(model$mixture, 0)
    expect_true(length(model$loss_history) == 10)
})

test_that("FFNN trains with elastic net regularization", {
    skip_if_no_torch()
    
    # Elastic net
    # With mixture = 0.5
    model = ffnn(
        Species ~ ., 
        data = iris[1:50, ], 
        hidden_neurons = 16, 
        penalty = 0.005,
        mixture = 0.5,    
        epochs = 10,
        verbose = FALSE
    )
    
    expect_s3_class(model, "ffnn_fit")
    expect_equal(model$penalty, 0.005)
    expect_equal(model$mixture, 0.5)
    expect_true(model$is_classification)
})

test_that("RNN trains with regularization", {
    skip_if_no_torch()
    
    model = rnn(
        Sepal.Length ~ ., 
        data = iris[1:50, 1:4], 
        hidden_neurons = c(16, 8), 
        rnn_type = "lstm",
        penalty = 0.01,
        mixture = 0,    
        epochs = 5,
        verbose = FALSE
    )
    
    expect_s3_class(model, "rnn_fit")
    expect_equal(model$penalty, 0.01)
    expect_equal(model$mixture, 0)
})

test_that("Regularization reduces to no penalty when penalty = 0", {
    skip_if_no_torch()
    
    model_no_reg = ffnn(
        Sepal.Length ~ ., 
        data = iris[1:30, 1:4], 
        hidden_neurons = 10, 
        penalty = 0,
        epochs = 10,
        verbose = FALSE
    )
    
    model_with_reg = ffnn(
        Sepal.Length ~ ., 
        data = iris[1:30, 1:4], 
        hidden_neurons = 10, 
        penalty = 0,
        mixture = 0,
        epochs = 10,
        verbose = FALSE
    )
    
    expect_s3_class(model_no_reg, "ffnn_fit")
    expect_s3_class(model_with_reg, "ffnn_fit")
    expect_equal(model_no_reg$penalty, 0)
    expect_equal(model_with_reg$penalty, 0)
})

test_that("Different mixture values produce different models", {
    skip_if_no_torch()
    
    set.seed(123)
    model_l2 = ffnn(
        Sepal.Length ~ ., 
        data = iris[1:30, 1:4], 
        hidden_neurons = 10, 
        penalty = 0.01,
        mixture = 0,      # L2
        epochs = 20,
        verbose = FALSE
    )
    
    set.seed(123)
    model_l1 = ffnn(
        Sepal.Length ~ ., 
        data = iris[1:30, 1:4], 
        hidden_neurons = 10, 
        penalty = 0.01,
        mixture = 1,      # L1
        epochs = 20,
        verbose = FALSE
    )
    
    expect_false(isTRUE(all.equal(model_l2$fitted, model_l1$fitted)))
})
