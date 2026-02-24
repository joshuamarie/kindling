skip_if_no_torch = function() {
    testthat::skip_if_not_installed("torch")
    testthat::skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

make_dataset = function(n = 10L) {
    force(n)

    torch::dataset(
        name = "toy_cls_dataset",
        initialize = function() {
            self$x = torch::torch_tensor(
                matrix(rnorm(n * 4L), ncol = 4L),
                dtype = torch::torch_float32()
            )
            self$y = torch::torch_tensor(
                rep(c(1L, 2L), length.out = n),
                dtype = torch::torch_long()
            )
        },
        .getitem = function(i) {
            list(self$x[i, ], self$y[i])
        },
        .length = function() {
            self$x$size(1)
        }
    )()
}

test_that("dataset method trains and predicts", {
    skip_if_no_torch()

    set.seed(1)
    ds = make_dataset(10L)

    fit = kindling::train_nn(
        x = ds,
        hidden_neurons = c(8L),
        epochs = 2,
        batch_size = 4,
        learn_rate = 0.01,
        n_classes = 2,
        verbose = FALSE
    )

    expect_s3_class(fit, "nn_fit_ds")

    cls_pred = stats::predict(fit, ds)
    prob_pred = stats::predict(fit, ds, type = "prob")

    expect_s3_class(cls_pred, "factor")
    expect_length(cls_pred, length(ds))
    expect_equal(dim(prob_pred), c(length(ds), 2L))
    expect_equal(colnames(prob_pred), c("1", "2"))
})

test_that("dataset classification requires n_classes", {
    skip_if_no_torch()

    set.seed(2)
    ds = make_dataset(8L)

    expect_error(
        kindling::train_nn(
            x = ds,
            hidden_neurons = c(4L),
            epochs = 1,
            batch_size = 4,
            learn_rate = 0.01,
            verbose = FALSE
        ),
        "n_classes"
    )
})

test_that("dataset validation split guards against empty partitions", {
    skip_if_no_torch()

    set.seed(3)
    ds = make_dataset(1L)

    expect_error(
        kindling::train_nn(
            x = ds,
            hidden_neurons = c(4L),
            epochs = 1,
            batch_size = 1,
            learn_rate = 0.01,
            n_classes = 2,
            validation_split = 0.5,
            verbose = FALSE
        ),
        "empty train/validation"
    )
})
