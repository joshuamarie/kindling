# Package index

## Model Specifications

Define neural network architectures for tidymodels

- [`mlp_kindling()`](https://kindling.joshuamarie.com/reference/mlp_kindling.md)
  : Multi-Layer Perceptron (Feedforward Neural Network) via kindling
- [`rnn_kindling()`](https://kindling.joshuamarie.com/reference/rnn_kindling.md)
  : Recurrent Neural Network via kindling

## Training Functions

Direct training interface (Level 2)

- [`ffnn()`](https://kindling.joshuamarie.com/reference/kindling-basemodels.md)
  [`rnn()`](https://kindling.joshuamarie.com/reference/kindling-basemodels.md)
  : Base models for Neural Network Training in kindling

## Code Generators

Generate torch nn_module code (Level 1)

- [`ffnn_generator()`](https://kindling.joshuamarie.com/reference/nn_gens.md)
  [`rnn_generator()`](https://kindling.joshuamarie.com/reference/nn_gens.md)
  :

  Functions to generate `nn_module` (language) expression

## Variable Importance

Interpret neural network models

- [`garson(`*`<ffnn_fit>`*`)`](https://kindling.joshuamarie.com/reference/kindling-varimp.md)
  [`olden(`*`<ffnn_fit>`*`)`](https://kindling.joshuamarie.com/reference/kindling-varimp.md)
  [`vi_model(`*`<ffnn_fit>`*`)`](https://kindling.joshuamarie.com/reference/kindling-varimp.md)
  : Variable Importance Methods for kindling Models

## Tuning Parameters

Hyperparameter specifications for tidymodels

- [`n_hlayers()`](https://kindling.joshuamarie.com/reference/dials-kindling.md)
  [`hidden_neurons()`](https://kindling.joshuamarie.com/reference/dials-kindling.md)
  [`activations()`](https://kindling.joshuamarie.com/reference/dials-kindling.md)
  [`output_activation()`](https://kindling.joshuamarie.com/reference/dials-kindling.md)
  [`optimizer()`](https://kindling.joshuamarie.com/reference/dials-kindling.md)
  [`bias()`](https://kindling.joshuamarie.com/reference/dials-kindling.md)
  [`validation_split()`](https://kindling.joshuamarie.com/reference/dials-kindling.md)
  [`bidirectional()`](https://kindling.joshuamarie.com/reference/dials-kindling.md)
  :

  Tunable hyperparameters for `kindling` models

## Helper Functions

Utilities for model configuration

- [`act_funs()`](https://kindling.joshuamarie.com/reference/act_funs.md)
  : Activation Functions Specification Helper
- [`args()`](https://kindling.joshuamarie.com/reference/args.md) :
  Activation Function Arguments Helper
- [`grid_depth()`](https://kindling.joshuamarie.com/reference/grid_depth.md)
  : Depth-Aware Grid Generation for Neural Networks
- [`table_summary()`](https://kindling.joshuamarie.com/reference/table_summary.md)
  : Summarize and Display a Two-Column Data Frame as a Formatted Table
- [`ordinal_gen()`](https://kindling.joshuamarie.com/reference/ordinal_gen.md)
  : Ordinal Suffixes Generator

## Basemodels-tidymodels wrappers

Functions for tidymodels integration (not for direct use)

- [`ffnn_wrapper()`](https://kindling.joshuamarie.com/reference/kindling-nn-wrappers.md)
  [`rnn_wrapper()`](https://kindling.joshuamarie.com/reference/kindling-nn-wrappers.md)
  : Basemodels-tidymodels wrappers
