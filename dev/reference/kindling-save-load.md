# Save and reload trained kindling models

`save_kindling()` writes a trained kindling model (an `nn_fit`,
`ffnn_fit`, or `rnn_fit` object) to a file, and `load_kindling()`
restores it. Base [`saveRDS()`](https://rdrr.io/r/base/readRDS.html) is
**not** appropriate for these objects: the underlying
[`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
holds external pointers into LibTorch that do not survive ordinary R
serialization. These functions split the object instead – the torch
module is serialized with
[`torch::torch_serialize()`](https://torch.mlverse.org/docs/reference/torch_serialize.html)
and everything else with R's serializer – so the reloaded model is fully
functional.

The reloaded model lives on the CPU (`$device` is set to `"cpu"`),
regardless of the device it was trained on; move it with
`object$model$to(device = ...)` if needed. Reloaded fits can be used
directly with [`predict()`](https://rdrr.io/r/stats/predict.html), and
re-submitted to
[`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md)
via its `initial_model` argument to continue training.

Note for custom architectures: the *instantiated* module is saved, so
prediction and continued training work after reload even for
[`nn_arch()`](https://kindling.joshuamarie.com/dev/reference/nn_arch.md)-based
models; only re-building the architecture from scratch would require the
original constructor objects.

## Usage

``` r
save_kindling(object, path)

load_kindling(path)
```

## Arguments

- object:

  A trained model of class `nn_fit`, `ffnn_fit`, or `rnn_fit`.

- path:

  File path to write to / read from.

## Value

`save_kindling()` returns `path` invisibly; `load_kindling()` returns
the restored model object.

## See also

[`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md)
(`initial_model` argument) for continuing training from a reloaded
model.

## Examples

``` r
# \donttest{
if (torch::torch_is_installed()) {
    fit = train_nn(
        x = as.matrix(iris[, 2:4]),
        y = iris$Sepal.Length,
        hidden_neurons = 8,
        epochs = 5
    )

    path = tempfile(fileext = ".rds")
    save_kindling(fit, path)

    fit2 = load_kindling(path)
    head(predict(fit2, newdata = as.matrix(iris[, 2:4])))

    # continue training from the reloaded model
    fit3 = train_nn(
        x = as.matrix(iris[, 2:4]),
        y = iris$Sepal.Length,
        initial_model = fit2,
        epochs = 5
    )

    unlink(path)
}
# }
```
