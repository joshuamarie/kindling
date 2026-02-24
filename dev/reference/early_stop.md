# Early Stopping Specification

`early_stop()` is a helper function to be supplied on `early_stopping`
arguments.

## Usage

``` r
early_stop(
  patience = 5L,
  min_delta = 1e-04,
  restore_best_weights = TRUE,
  monitor = "val_loss"
)
```

## Arguments

- patience:

  Integer. Epochs to wait after last improvement. Default `5`.

- min_delta:

  Numeric. Minimum improvement to qualify as better. Default `1e-4`.

- restore_best_weights:

  Logical. Restore weights from best epoch. Default `TRUE`.

- monitor:

  Character. Metric to monitor. One of `"val_loss"` (default) or
  `"train_loss"`.

## Value

An object of class `"early_stop_spec"`.
