skip_if_no_torch = function() {
    skip_if_not_installed("torch")
    skip_if_not(torch::torch_is_installed(), "Torch backend not available")
}

# ---- Classification: spec and fit -------------------------------------------

test_that("train_nnsnip() returns correct class and mode - classification", {
    skip_if_not_installed("parsnip")
    
    spec = train_nnsnip(
        mode = "classification",
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 5
    )
    
    expect_s3_class(spec, "train_nnsnip")
    expect_equal(spec$mode, "classification")
    expect_equal(spec$engine, "kindling")
})

test_that("train_nnsnip() fits classification workflow without error", {
    skip_if_not_installed("parsnip")
    skip_if_no_torch()
    
    spec = train_nnsnip(
        mode = "classification",
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 5,
        verbose = FALSE
    )
    
    expect_no_error({
        fitted = parsnip::fit(
            spec,
            Species ~ .,
            data = iris
        )
    })
    
    expect_s3_class(fitted, "model_fit")
    expect_s3_class(fitted$fit, "nn_fit_tab")
})

test_that("train_nnsnip() classification fit sets is_classification = TRUE", {
    skip_if_not_installed("parsnip")
    skip_if_no_torch()
    
    spec = train_nnsnip(
        mode = "classification",
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 5,
        verbose = FALSE
    )
    
    fitted = parsnip::fit(spec, Species ~ ., data = iris)
    
    expect_true(fitted$fit$is_classification)
    expect_equal(fitted$fit$y_levels, levels(iris$Species))
})

# ---- Regression: spec and fit -----------------------------------------------

test_that("train_nnsnip() returns correct class and mode - regression", {
    skip_if_not_installed("parsnip")
    
    spec = train_nnsnip(
        mode = "regression",
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 5
    )
    
    expect_s3_class(spec, "train_nnsnip")
    expect_equal(spec$mode, "regression")
})

test_that("train_nnsnip() fits regression workflow without error", {
    skip_if_not_installed("parsnip")
    skip_if_no_torch()
    
    spec = train_nnsnip(
        mode = "regression",
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 5,
        verbose = FALSE
    )
    
    expect_no_error({
        fitted = parsnip::fit(
            spec,
            Sepal.Length ~ .,
            data = iris[, 1:4]
        )
    })
    
    expect_s3_class(fitted, "model_fit")
    expect_s3_class(fitted$fit, "nn_fit_tab")
    expect_false(fitted$fit$is_classification)
})

# ---- Predictions: classification --------------------------------------------

test_that("train_nnsnip() predict type = 'class' returns correct format", {
    skip_if_not_installed("parsnip")
    skip_if_no_torch()
    
    spec = train_nnsnip(
        mode = "classification",
        hidden_neurons = c(10),
        epochs = 5,
        verbose = FALSE
    )
    
    fitted = parsnip::fit(spec, Species ~ ., data = iris[1:100, ])
    preds = predict(fitted, new_data = iris[101:110, ])
    
    expect_s3_class(preds, "tbl_df")
    expect_true(".pred_class" %in% names(preds))
    expect_equal(nrow(preds), 10)
    expect_s3_class(preds$.pred_class, "factor")
    expect_equal(levels(preds$.pred_class), levels(iris$Species))
})

test_that("train_nnsnip() predict type = 'prob' returns correct format", {
    skip_if_not_installed("parsnip")
    skip_if_no_torch()
    
    spec = train_nnsnip(
        mode = "classification",
        hidden_neurons = c(10),
        epochs = 5,
        verbose = FALSE
    )
    
    fitted = parsnip::fit(spec, Species ~ ., data = iris[1:100, ])
    preds = predict(fitted, new_data = iris[101:110, ], type = "prob")
    
    expect_s3_class(preds, "tbl_df")
    expect_equal(nrow(preds), 10)
    expect_true(all(grepl("^\\.pred_", names(preds))))
    expect_true(all(abs(rowSums(preds) - 1) < 1e-6))
})

# ---- Predictions: regression ------------------------------------------------

test_that("train_nnsnip() predict returns correct format - regression", {
    skip_if_not_installed("parsnip")
    skip_if_no_torch()
    
    spec = train_nnsnip(
        mode = "regression",
        hidden_neurons = c(10),
        epochs = 5,
        verbose = FALSE
    )
    
    fitted = parsnip::fit(
        spec,
        Sepal.Length ~ Petal.Length + Petal.Width,
        data = iris[1:50, ]
    )
    
    preds = predict(fitted, new_data = iris[51:60, ])
    
    expect_s3_class(preds, "tbl_df")
    expect_true(".pred" %in% names(preds))
    expect_equal(nrow(preds), 10)
    expect_type(preds$.pred, "double")
})

# ---- Engine args passthrough ------------------------------------------------

test_that("validation_split is passed through and produces val_loss_history", {
    skip_if_not_installed("parsnip")
    skip_if_no_torch()
    
    spec = train_nnsnip(
        mode = "classification",
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 5,
        validation_split = 0.2,
        verbose = FALSE
    )
    
    fitted = parsnip::fit(spec, Species ~ ., data = iris)
    
    expect_false(is.null(fitted$fit$val_loss_history))
    expect_length(fitted$fit$val_loss_history, fitted$fit$n_epochs)
})

test_that("penalty and mixture are passed through correctly", {
    skip_if_not_installed("parsnip")
    skip_if_no_torch()
    
    spec = train_nnsnip(
        mode = "classification",
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 5,
        penalty = 0.01,
        mixture = 0.5,
        verbose = FALSE
    )
    
    fitted = parsnip::fit(spec, Species ~ ., data = iris)
    
    expect_equal(fitted$fit$penalty, 0.01)
    expect_equal(fitted$fit$mixture, 0.5)
})

test_that("optimizer arg is passed through without error", {
    skip_if_not_installed("parsnip")
    skip_if_no_torch()
    
    spec = train_nnsnip(
        mode = "classification",
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 5,
        optimizer = "sgd",
        verbose = FALSE
    )
    
    expect_no_error(parsnip::fit(spec, Species ~ ., data = iris))
})

# ---- Early stopping ---------------------------------------------------------

test_that("early_stopping halts training before max epochs", {
    skip_if_not_installed("parsnip")
    skip_if_no_torch()
    
    spec = train_nnsnip(
        mode = "classification",
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 200,
        validation_split = 0.2,
        early_stopping = early_stop(patience = 3, min_delta = 1e10),
        verbose = FALSE
    )
    
    fitted = parsnip::fit(spec, Species ~ ., data = iris)
    
    expect_lt(fitted$fit$n_epochs, 200)
    expect_false(is.na(fitted$fit$stopped_epoch))
})

test_that("without early_stopping all epochs are run", {
    skip_if_not_installed("parsnip")
    skip_if_no_torch()
    
    spec = train_nnsnip(
        mode = "classification",
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 5,
        verbose = FALSE
    )
    
    fitted = parsnip::fit(spec, Species ~ ., data = iris)
    
    expect_equal(fitted$fit$n_epochs, 5)
    expect_true(is.na(fitted$fit$stopped_epoch))
})

# ---- Custom architecture ----------------------------------------------------

test_that("nn_arch object is passed through and stored in fit", {
    skip_if_not_installed("parsnip")
    skip_if_no_torch()
    
    arch = nn_arch(nn_name = "CustomMLP")
    
    spec = train_nnsnip(
        mode = "classification",
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 5,
        architecture = arch,
        verbose = FALSE
    )
    
    fitted = parsnip::fit(spec, Species ~ ., data = iris)
    
    expect_false(is.null(fitted$fit$arch))
    expect_equal(fitted$fit$arch$nn_name, "CustomMLP")
})

# ---- update() ---------------------------------------------------------------

test_that("update.train_nnsnip() updates args and preserves others", {
    skip_if_not_installed("parsnip")
    
    spec = train_nnsnip(
        mode = "classification",
        hidden_neurons = c(16, 8),
        activations = "relu",
        epochs = 5
    )
    
    spec2 = update(spec, epochs = 20, learn_rate = 0.01)
    
    expect_equal(rlang::quo_get_expr(spec2$args$epochs), 20)
    expect_equal(rlang::quo_get_expr(spec2$args$learn_rate), 0.01)
    expect_equal(rlang::quo_get_expr(spec2$args$hidden_neurons), quote(c(16, 8)))
})

# ---- tunable() --------------------------------------------------------------

test_that("tunable.train_nnsnip() returns expected parameter names", {
    skip_if_not_installed("parsnip")
    skip_if_not_installed("tune")
    
    spec = train_nnsnip(mode = "classification")
    tbl = tune::tunable(spec)
    
    expect_s3_class(tbl, "tbl_df")
    expect_setequal(tbl$name, c(
        "hidden_neurons", "activations", "output_activation", "bias",
        "epochs", "batch_size", "penalty", "mixture",
        "learn_rate", "optimizer", "validation_split"
    ))
})

# ---- prepare_kindling_args() ------------------------------------------------

test_that("prepare_kindling_args() evaluates one-sided formulas and drops NULLs", {
    args = list(epochs = ~50L, loss = ~"cross_entropy", device = ~NULL)
    out = prepare_kindling_args(args)
    
    expect_equal(out$epochs, 50L)
    expect_equal(out$loss, "cross_entropy")
    expect_false("device" %in% names(out))
})

test_that("prepare_kindling_args() evaluates quosures", {
    args = list(
        epochs = rlang::quo(100L),
        optimizer = rlang::quo("adam")
    )
    out = prepare_kindling_args(args)
    
    expect_equal(out$epochs, 100L)
    expect_equal(out$optimizer, "adam")
})

test_that("prepare_kindling_args() unwraps single-element hidden_neurons list", {
    args = list(hidden_neurons = list(c(32L, 16L)))
    out = prepare_kindling_args(args)
    
    expect_equal(out$hidden_neurons, c(32L, 16L))
})

test_that("prepare_kindling_args() passes plain values through unchanged", {
    args = list(epochs = 50L, learn_rate = 0.001)
    out = prepare_kindling_args(args)
    
    expect_equal(out$epochs, 50L)
    expect_equal(out$learn_rate, 0.001)
})
