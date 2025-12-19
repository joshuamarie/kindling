#' `{kindling}`: Higher-level interface of torch package in training neural networks
#'
#' The `{kindling}` package provides a unified, high-level interface that bridges
#' the **{torch}** and **{tidymodels}** ecosystems, making it easy to define, train,
#' and tune deep learning models using the familiar `tidymodels` workflow.
#'
#' @description
#' `{kindling}` enables R users to build and train deep neural networks such as:
#'
#' - Deep Neural Networks / (Deep) Feedforward Neural Networks (DNN / FFNN)
#' - Recurrent Neural Networks (RNN)
#'
#'
#' It is designed to reduce boilerplate `{torch}` code for FFNN and RNN. It also
#' integrate seamlessly with `{tidymodels}` components like `{parsnip}`, `{recipes}`,
#' and `{workflows}`, allowing flexibility and a consistent interface for model
#' specification, training, and evaluation.
#'
#' Thus, the package supports hyperparameter tuning for:
#'
#' - Number of hidden layers
#' - Number of units per layer
#' - Choice of activation functions
#'
#' Note: The hyperparameter tuning support is not currently implemented.
#'
#' @section How to use:
#' The following uses of this package has 3 levels:
#'
#' Level 1: Code generation
#'
#' ``` r
#' ffnn_generator(
#'    nn_name = "MyFFNN",
#'    hd_neurons = c(64, 32, 16),
#'    no_x = 10,
#'    no_y = 1,
#'    activations = 'relu'
#' )
#' ```
#'
#' Level 2: Direct Execution
#'
#' ``` r
#' ffnn(
#'    Species ~ .,
#'    data = iris,
#'    hidden_neurons = c(128, 64, 32),
#'    activations = 'relu',
#'    loss = "cross_entropy",
#'    epochs = 100
#' )
#' ```
#'
#' Level 3: tidymodels interface part 1
#'
#' ``` r
#' # library(parsnip)
#' # library(kindling)
#' box::use(
#'    kindling[mlp_kindling, rnn_kindling, act_funs, args],
#'    parsnip[fit, augment],
#'    yardstick[metrics],
#'    mlbench[Ionosphere] # data(Ionosphere, package = "mlbench")
#' )
#'
#' # Remove V2 as it's all zeros
#' ionosphere_data = Ionosphere[, -2]
#'
#' # MLP example
#' mlp_kindling(
#'    mode = "classification",
#'    hidden_neurons = c(128, 64),
#'    activations = act_funs(relu, softshrink = args(lambd = 0.5)),
#'    epochs = 100
#' ) |>
#'    fit(Class ~ ., data = ionosphere_data) |>
#'    augment(new_data = ionosphere_data) |>
#'    metrics(truth = Class, estimate = .pred_class)
#' #> A tibble: 2 × 3
#' #>   .metric  .estimator .estimate
#' #>   <chr>    <chr>          <dbl>
#' #> 1 accuracy binary         0.989
#' #> 2 kap      binary         0.975
#'
#' # RNN example (toy usage on non-sequential data)
#' rnn_kindling(
#'    mode = "classification",
#'    hidden_neurons = c(128, 64),
#'    activations = act_funs(relu, elu),
#'    epochs = 100,
#'    rnn_type = "gru"
#' ) |>
#'    fit(Class ~ ., data = ionosphere_data) |>
#'    augment(new_data = ionosphere_data) |>
#'    metrics(truth = Class, estimate = .pred_class)
#' #> A tibble: 2 × 3
#' #>   .metric  .estimator .estimate
#' #>   <chr>    <chr>          <dbl>
#' #> 1 accuracy binary         0.641
#' #> 2 kap      binary         0
#' ```
#'
#' Level 4: tidymodels interface part 2 - tuning (not yet implemented)
#'
#' @section Key Features:
#' - Define neural network models using `parsnip::set_engine("kindling")`
#' - Integrate deep learning into `{tidymodels}` workflows
#' - Support for multiple architectures (DNN, RNN)
#' - Hyperparameter tuning for architecture depth, units, and activation
#' - Compatible with `{torch}` tensors for GPU acceleration
#'
#' @references
#'
#' Falbel D, Luraschi J (2023). _torch: Tensors and Neural Networks with 'GPU'
#' Acceleration_. R package version 0.13.0,
#' \url{https://torch.mlverse.org}, \url{https://github.com/mlverse/torch}.
#'
#' Wickham H (2019). _Advanced R_, 2nd edition. Chapman and Hall/CRC.
#' ISBN 978-0815384571, \url{https://adv-r.hadley.nz/}.
#'
#' Goodfellow I, Bengio Y, Courville A (2016). _Deep Learning_. MIT Press.
#' \url{https://www.deeplearningbook.org/}.
#'
#' @section License:
#' MIT + file LICENSE
#'
#' @docType package
#' @name kindling
#' @keywords internal
"_PACKAGE"

# Suppress R CMD check notes for NSE variables used in torch modules
# These variables (self, x) are used in non-standard evaluation contexts
# within torch::nn_module definitions and are intentional
utils::globalVariables(c("self", "x"))
