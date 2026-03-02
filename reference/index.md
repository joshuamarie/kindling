# Package index

## Model Specifications

Define neural network architectures for tidymodels

- [`mlp_kindling()`](https://kindling.joshuamarie.com/reference/mlp_kindling.md)
  : Multi-Layer Perceptron (Feedforward Neural Network) via kindling

- [`rnn_kindling()`](https://kindling.joshuamarie.com/reference/rnn_kindling.md)
  : Recurrent Neural Network via kindling

- [`train_nnsnip()`](https://kindling.joshuamarie.com/reference/train_nnsnip.md)
  :

  Parsnip Interface of
  [`train_nn()`](https://kindling.joshuamarie.com/reference/gen-nn-train.md)

## Training Functions

Direct training interface (Level 2)

### Generalized Neural Network Trainer

- [`train_nn()`](https://kindling.joshuamarie.com/reference/gen-nn-train.md)
  **\[experimental\]** : Generalized Neural Network Trainer
- [`nn_arch()`](https://kindling.joshuamarie.com/reference/nn_arch.md) :
  Architecture specification for train_nn()

### Base Models

- [`ffnn()`](https://kindling.joshuamarie.com/reference/kindling-basemodels.md)
  [`rnn()`](https://kindling.joshuamarie.com/reference/kindling-basemodels.md)
  : Base models for Neural Network Training in kindling

### Utility Functions

- [`nn_arch()`](https://kindling.joshuamarie.com/reference/nn_arch.md) :
  Architecture specification for train_nn()
- [`early_stop()`](https://kindling.joshuamarie.com/reference/early_stop.md)
  : Early Stopping Specification

## Code Generators

Generate
[`torch::nn_module()`](https://torch.mlverse.org/docs/reference/nn_module.html)
code (Lowest Level)

### General-purpose / low-level generator, including the layer utilities & pronouns

- [`nn_module_generator()`](https://kindling.joshuamarie.com/reference/nn_module_generator.md)
  **\[experimental\]** : Generalized Neural Network Module Expression
  Generator
- [`.layer`](https://kindling.joshuamarie.com/reference/layer_prs.md)
  [`.i`](https://kindling.joshuamarie.com/reference/layer_prs.md)
  [`.in`](https://kindling.joshuamarie.com/reference/layer_prs.md)
  [`.out`](https://kindling.joshuamarie.com/reference/layer_prs.md)
  [`.is_output`](https://kindling.joshuamarie.com/reference/layer_prs.md)
  : Layer argument pronouns for formula-based specifications

### High-level generators

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
- [`args()`](https://kindling.joshuamarie.com/reference/args.md)
  **\[superseded\]** : Activation Function Arguments Helper
- [`new_act_fn()`](https://kindling.joshuamarie.com/reference/new_act_fn.md)
  : Custom Activation Function Constructor
- [`grid_depth()`](https://kindling.joshuamarie.com/reference/grid_depth.md)
  : Depth-Aware Grid Generation for Neural Networks
- [`table_summary()`](https://kindling.joshuamarie.com/reference/table_summary.md)
  : Summarize and Display a Two-Column Data Frame as a Formatted Table
- [`ordinal_gen()`](https://kindling.joshuamarie.com/reference/ordinal_gen.md)
  : Ordinal Suffixes Generator
