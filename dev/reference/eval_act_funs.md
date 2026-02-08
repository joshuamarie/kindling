# Evaluate Activation Function Specifications in DSL Environment

Helper function to evaluate activation specifications with act_funs()
and args() available without namespace prefixes.

## Usage

``` r
eval_act_funs(activations, output_activation)
```

## Arguments

- activations_quo:

  Quosure containing the activations expression

- output_activation_quo:

  Quosure containing the output_activation expression

## Value

A list with two elements: `activations` and `output_activation`
