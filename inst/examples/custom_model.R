box::use(
    kindling[train_nn, nn_arch, ffnn],
    torch[
        torch_randn, torch_zeros, torch_exp, 
        nn_module, nn_parameter, nn_linear
    ]
)

rbf_gaussian = function(alpha) {
    torch_exp(-1 * alpha$pow(2))
}

rbf_layer = nn_module(
    "RBFLayer",
    initialize = function(in_features, out_features, basis_func = \(alpha) torch_exp(-alpha)) {
        self$in_features = in_features
        self$out_features = out_features
        self$centres = nn_parameter(torch_randn(out_features, in_features) * 0.1)
        self$log_sigma = nn_parameter(torch_zeros(out_features))
        self$basis_func = basis_func
    },
    
    forward = function(input) {
        batch_size = input$size(1)
        
        x = input$unsqueeze(2)
        c = self$centres$unsqueeze(1)
        distances = (x - c)$pow(2)$sum(3)
        
        # Use log_sigma for numerical stability
        # beta = 1 / (2 * exp(2 * log_sigma)) = exp(-2 * log_sigma) / 2
        beta = torch_exp(-2 * self$log_sigma)$unsqueeze(1) / 2
        
        self$basis_func(distances * beta)
    }
)

model_rbf = train_nn(
    Species ~ .,
    data = iris,
    hidden_neurons = c(32, 16),
    epochs = 150,
    batch_size = 32,
    learn_rate = 0.01,
    loss = "cross_entropy",
    validation_split = 0.2,
    arch = nn_arch(
        nn_name = "RBFNet",
        nn_layer = ~ rbf_layer,
        out_nn_layer = ~ torch::nn_linear,
        layer_arg_fn = ~ if (.is_output) {
            list(.in, .out)
        } else {
            list(in_features = .in, out_features = .out)
        }
    )
)

model_nn = train_nn(
    Species ~ .,
    data = iris,
    hidden_neurons = c(32, 16),
    epochs = 150,
    batch_size = 32,
    learn_rate = 0.01,
    loss = "cross_entropy",
    validation_split = 0.2
)

model_nn2 = ffnn(
    Species ~ .,
    data = iris,
    hidden_neurons = c(32, 16),
    epochs = 150,
    batch_size = 32,
    learn_rate = 0.01,
    loss = "cross_entropy",
    validation_split = 0.2
)

preds_rbf = predict(model_rbf, newdata = iris)
preds_nn = predict(model_nn, newdata = iris)
preds_nn2 = predict(model_nn2, newdata = iris)

cat("\n\nPrediction of RBF Layer:\n\n")
table(actual = iris$Species, prediction = preds_rbf)

cat("\n\nPrediction of MLP:\n\n")
table(actual = iris$Species, prediction = preds_nn)

cat("\n\nPrediction of MLP (from `ffnn()` function):\n\n")
table(actual = iris$Species, prediction = preds_nn2)
