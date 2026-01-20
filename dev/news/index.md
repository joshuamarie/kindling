# Changelog

## kindling (development version)

### New features

- Added regularization support for neural network models

  - L1 regularization (Lasso) for feature selection via `mixture = 1`
  - L2 regularization (Ridge) for weight decay via `mixture = 0`
  - Elastic Net combining L1 and L2 penalties via `0 < mixture < 1`
  - Controlled via `penalty` (regularization strength) and `mixture`
    (L1/L2 balance) parameters
  - Follows tidymodels conventions for consistency with `glmnet` and
    other packages

## kindling 0.1.0

- Initial CRAN release
- Higher-level interface for torch package to define, train, and tune
  neural networks
- Support for feedforward (multi-layer perceptron) and recurrent
  networks (RNN, LSTM, GRU)
- Integration with tidymodels ecosystem (parsnip, workflows, recipes,
  tuning)
- Variable importance plots and network visualization tools
