skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

test_that("ffnn works with optimizer_args", {
    skip_if_no_torch()
    
    model1 = ffnn(
        Sepal.Length ~ .,
        data = iris[1:50, 1:4],
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 5,
        optimizer = "adam",
        optimizer_args = list(weight_decay = 0.01),
        verbose = FALSE
    )
    
    expect_s3_class(model1, "ffnn_fit")
    expect_true(length(model1$loss_history) == 5)
    
    # Test with momentum (SGD)
    model2 = ffnn(
        Species ~ .,
        data = iris[1:50, ],
        hidden_neurons = 16,
        activations = "relu",
        epochs = 5,
        optimizer = "sgd",
        learn_rate = 0.01,
        optimizer_args = list(momentum = 0.9),
        verbose = FALSE
    )
    
    expect_s3_class(model2, "ffnn_fit")
    expect_true(model2$is_classification)
    
    # Test with multiple optimizer args
    model3 = ffnn(
        Sepal.Length ~ .,
        data = iris[1:50, 1:4],
        hidden_neurons = 8,
        epochs = 3,
        optimizer = "adam",
        optimizer_args = list(weight_decay = 0.001, amsgrad = TRUE),
        verbose = FALSE
    )
    
    expect_s3_class(model3, "ffnn_fit")
})

test_that("rnn works with optimizer_args", {
    skip_if_no_torch()
    
    # Test LSTM with weight_decay
    model1 = rnn(
        Sepal.Length ~ .,
        data = iris[1:50, 1:4],
        hidden_neurons = c(16, 8),
        rnn_type = "lstm",
        epochs = 5,
        optimizer = "adam",
        optimizer_args = list(weight_decay = 0.01),
        verbose = FALSE
    )
    
    expect_s3_class(model1, "rnn_fit")
    expect_equal(model1$rnn_type, "lstm")
    
    # Test GRU with momentum
    model2 = rnn(
        Species ~ .,
        data = iris[1:50, ],
        hidden_neurons = 16,
        rnn_type = "gru",
        epochs = 5,
        optimizer = "sgd",
        optimizer_args = list(momentum = 0.9, dampening = 0.1),
        verbose = FALSE
    )
    
    expect_s3_class(model2, "rnn_fit")
    expect_true(model2$is_classification)
    expect_equal(model2$rnn_type, "gru")
})

test_that("optimizer_args defaults to empty list", {
    skip_if_no_torch()
    
    # Should work without optimizer_args
    model = ffnn(
        Sepal.Length ~ Sepal.Width,
        data = iris[1:30, ],
        hidden_neurons = 8,
        epochs = 3,
        verbose = FALSE
    )
    
    expect_s3_class(model, "ffnn_fit")
})

test_that("invalid optimizer_args throw errors", {
    skip_if_no_torch()
    
    expect_error(
        ffnn(
            Sepal.Length ~ .,
            data = iris[1:30, 1:4],
            hidden_neurons = 8,
            epochs = 2,
            optimizer = "adam",
            optimizer_args = list(invalid_param = 999)
        )
    )
})
