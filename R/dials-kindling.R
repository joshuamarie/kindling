#' Tunable hyperparameters for `kindling` models
#'
#' These parameters extend the **dials** framework to support hyperparameter
#' tuning of neural networks built with the `{kindling}` package. They control
#' network architecture, activation functions, optimization, and training
#' behavior.
#'
#' @section Architecture Strategy:
#' Since tidymodels tuning works with independent parameters, we use a simplified
#' approach where:
#' - `hidden_neurons` specifies a single value that will be used for ALL layers
#' - `activations` specifies a single activation that will be used for ALL layers
#' - `n_hlayers` controls the depth
#'
#' For more complex architectures with different neurons/activations per layer,
#' users should manually specify these after tuning or use custom tuning logic.
#'
#' @section Parameters:
#'
#' \describe{
#'   \item{`n_hlayers`}{Number of hidden layers in the network.}
#'   \item{`hidden_neurons`}{Number of units per hidden layer (applied to all layers).}
#'   \item{`activation`}{Single activation function applied to all hidden layers.}
#'   \item{`output_activation`}{Activation function for the output layer.}
#'   \item{`optimizer`}{Optimizer algorithm.}
#'   \item{`bias`}{Whether to include bias terms in layers.}
#'   \item{`validation_split`}{Proportion of training data held out for validation.}
#'   \item{`bidirectional`}{Whether RNN layers are bidirectional.}
#' }
#' 
#' @return
#' Each function returns a `dials` parameter object:
#' \describe{
#'     \item{`n_hlayers()`}{A quantitative parameter for the number of hidden layers}
#'     \item{`hidden_neurons()`}{A quantitative parameter for hidden units per layer}
#'     \item{`activations()`}{A qualitative parameter for activation function names}
#'     \item{`output_activation()`}{A qualitative parameter for output activation}
#'     \item{`optimizer()`}{A qualitative parameter for optimizer type}
#'     \item{`bias()`}{A qualitative parameter for bias inclusion}
#'     \item{`validation_split()`}{A quantitative parameter for validation proportion}
#'     \item{`bidirectional()`}{A qualitative parameter for bidirectional RNN}
#' }
#'
#' @examples
#' \donttest{
#' library(dials)
#' library(tune)
#'
#' # Create a tuning grid
#' grid = grid_regular(
#'     n_hlayers(range = c(1L, 4L)),
#'     hidden_neurons(range = c(32L, 256L)),
#'     activations(c('relu', 'elu', 'selu')),
#'     levels = c(4, 5, 3)
#' )
#'
#' # Use in a model spec
#' mlp_spec = mlp_kindling(
#'     mode = "classification",
#'     hidden_neurons = tune(),
#'     activations = tune(),
#'     epochs = tune(),
#'     learn_rate = tune()
#' )
#' }
#'
#' @name dials-kindling
#' @keywords internal
NULL

#' @section Number of Hidden Layers:
#' Controls the depth of the network. When tuning, this will determine how many
#' layers are created, each with `hidden_neurons` units and `activations` function.
#'
#' @param range A two-element integer vector with the default lower and upper bounds.
#' @param trans An optional transformation (e.g., `transform_log()`); `NULL` for none.
#'
#' @rdname dials-kindling
#' @export
n_hlayers = function(range = c(1L, 5L), trans = NULL) {
    dials::new_quant_param(
        type = "integer",
        range = range,
        inclusive = c(TRUE, TRUE),
        trans = trans,
        label = c(n_hlayers = "Number of Hidden Layers"),
        finalize = NULL
    )
}

#' @section Hidden Units per Layer:
#' Specifies the number of units per hidden layer.
#'
#' @param range A two-element integer vector with the default lower and upper bounds.
#' @param trans An optional transformation; `NULL` for none.
#'
#' @rdname dials-kindling
#' @export
hidden_neurons = function(range = c(8L, 512L), trans = NULL) {
    dials::new_quant_param(
        type = "integer",
        range = range,
        inclusive = c(TRUE, TRUE),
        trans = trans,
        label = c(hidden_neurons = "Hidden Units per Layer"),
        finalize  = NULL
    )
}

#' @section Activation Function (Hidden Layers):
#' Activation functions for hidden layers.
#'
#' @param values Character vector of possible activation names.
#'
#' @rdname dials-kindling
#' @export
activations = function(values = c(
    "relu", "relu6", "elu", "selu", "celu",
    "leaky_relu", "gelu", "softplus", "softshrink",
    "softsign", "tanhshrink", "hardtanh",
    "hardshrink", "hardswish", "hardsigmoid",
    "silu", "mish", "logsigmoid"
)) {
    dials::new_qual_param(
        type = "character",
        values = values,
        label = c(activations = "Activation Function (Hidden Layers)"),
        finalize = NULL
    )
}

#' @section Output Activation Function:
#' Activation function applied to the output layer. Values must correspond to
#' `torch::nnf_*` functions.
#'
#' @param values Character vector of possible output activation names.
#'
#' @rdname dials-kindling
#' @export
output_activation = function(values = c(
    "relu", "elu", "selu", "softplus",
    "softmax", "log_softmax", "logsigmoid",
    "hardtanh", "hardsigmoid", "silu"
)) {
    dials::new_qual_param(
        type = "character",
        values = values,
        label = c(output_activation = "Output Activation Function"),
        finalize = NULL
    )
}

#' @section Optimizer Type:
#' The optimization algorithm used during training.
#'
#' @param values Character vector of supported optimizers.
#'
#' @rdname dials-kindling
#' @export
optimizer = function(values = c("adam", "sgd", "rmsprop", "adamw")) {
    dials::new_qual_param(
        type = "character",
        values = values,
        label = c(optimizer = "Optimizer Type"),
        finalize = NULL
    )
}

#' @section Include Bias Terms:
#' Whether layers should include bias parameters.
#'
#' @param values Logical vector of possible values.
#'
#' @rdname dials-kindling
#' @export
bias = function(values = c(TRUE, FALSE)) {
    dials::new_qual_param(
        type = "logical",
        values = values,
        label = c(bias = "Include Bias Terms"),
        finalize = NULL
    )
}

#' @section Validation Split Proportion:
#' Fraction of the training data to use as a validation set during training.
#'
#' @param range A two-element numeric vector with the default lower and upper bounds.
#' @param trans An optional transformation; `NULL` for none.
#'
#' @rdname dials-kindling
#' @export
validation_split = function(range = c(0, 0.5), trans = NULL) {
    dials::new_quant_param(
        type = "double",
        range = range,
        inclusive = c(TRUE, TRUE),
        trans = trans,
        label = c(validation_split = "Validation Split Proportion"),
        finalize  = NULL
    )
}

#' @section Bidirectional RNN:
#' Whether recurrent layers should process sequences in both directions.
#'
#' @param values Logical vector of possible values.
#'
#' @rdname dials-kindling
#' @export
bidirectional = function(values = c(TRUE, FALSE)) {
    dials::new_qual_param(
        type = "logical",
        values = values,
        label = c(bidirectional = "Bidirectional RNN"),
        finalize = NULL
    )
}
