#' kindling: A High-Level Interface for Deep Learning with Torch and Tidymodels
#'
#' The `kindling` package provides a unified, high-level interface that bridges
#' the **{torch}** and **{tidymodels}** ecosystems, making it easy to define, train,
#' and tune deep learning models using the familiar `tidymodels` workflow.
#'
#' @description
#' `kindling` enables R users to build and train deep neural networks such as:
#'
#' - Feedforward Neural Networks (DNN)
#' - Convolutional Neural Networks (CNN)
#' - Recurrent Neural Networks (RNN)
#'
#' The package supports hyperparameter tuning for:
#'
#' - Number of hidden layers
#' - Number of units per layer
#' - Choice of activation functions
#'
#' It is designed to integrate seamlessly with `tidymodels` components like
#' `parsnip`, `recipes`, and `workflows`, allowing a consistent interface for
#' model specification, training, and evaluation.
#'
#' @section Key Features:
#' - Define neural network models using `parsnip::set_engine("kindling")`
#' - Integrate deep learning into `tidymodels` workflows
#' - Support for multiple architectures (DNN, CNN, RNN, LSTM)
#' - Hyperparameter tuning for architecture depth, units, and activation
#' - Compatible with `torch` tensors for GPU acceleration
#'
#' @section License:
#' MIT + file LICENSE
#'
#' @docType package
#' @name kindling
#' @keywords internal
"_PACKAGE"
