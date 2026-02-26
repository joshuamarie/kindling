# Custom Activation Function Constructor

Wraps a user-supplied function into a validated custom activation,
ensuring it accepts and returns a `torch_tensor`. Performs an eager
dry-run probe at *definition time* so errors surface early, and wraps
the function with a *call-time* type guard for safety.

## Usage

``` r
new_act_fn(fn, probe = TRUE, .name = "<custom>")
```

## Arguments

- fn:

  A function taking a single tensor argument and returning a tensor.
  E.g. `\(x) torch::torch_tanh(x)`.

- probe:

  Logical. If `TRUE` (default), runs a dry-run with a small dummy tensor
  at definition time to catch obvious errors early.

- .name:

  A string. Default is `"<custom>"`.

## Value

An object of class `c("custom_activation", "parameterized_activation")`,
compatible with
[`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md).

## Examples
