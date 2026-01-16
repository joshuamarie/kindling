box::use(
    kindling[ffnn, act_funs, args],
    NeuralNetTools[garson, olden],
    vip[vip, vi]
)

cat("\n\n=== Example 1: FFNN Classification ===\n")

model_iris = ffnn(
    Species ~ .,
    data = iris,
    hidden_neurons = c(64, 32),
    activations = "relu",
    epochs = 100,
    verbose = FALSE,
    cache_weights = TRUE
)

cat("\nGarson's Algorithm:\n")

imp_garson = garson(model_iris, bar_plot = FALSE)
imp_garson

cat("\nOlden's Algorithm:\n")
imp_olden = olden(model_iris, bar_plot = FALSE)
imp_olden

cat("\nvip Integration:\n")
imp_vip = vi(model_iris)
imp_vip

cat("\n\n=== Example 2: FFNN Regression (mtcars dataset) ===\n")

model_mtcars = ffnn(
    mpg ~ cyl + disp + hp + drat + wt + qsec,
    data = mtcars,
    hidden_neurons = c(32, 16),
    activations = "relu",
    epochs = 150,
    loss = "mse",
    verbose = FALSE,
    cache_weights = TRUE
)

cat("\nVariable Importance (Olden's method):\n")
imp_reg = olden(model_mtcars, bar_plot = FALSE)
imp_reg

cat("\nVariable Importance (Garson's method):\n")
imp_reg = garson(model_mtcars, bar_plot = FALSE)
imp_reg

cat("\nvip Integration:\n")
imp_vip = vi(model_iris)
imp_vip
