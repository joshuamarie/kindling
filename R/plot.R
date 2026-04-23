#' Plot training loss history for a fitted neural network
#'
#' Creates a line plot of training (and optionally validation) loss over epochs
#' for a model of class `nn_fit`.
#'
#' @param object A fitted model of class `nn_fit`, as returned by
#'   [train_nn()], [ffnn()], or [rnn()].
#' @param ... Additional arguments (currently unused).
#'
#' @return A [ggplot2::ggplot()] object showing loss vs epoch.
#'
#' @examples
#' \donttest{
#'   if (torch::torch_is_installed()) {
#'     m = train_nn(
#'       as.matrix(iris[, 2:4]), iris$Sepal.Length,
#'       epochs = 5
#'     )
#'     ggplot2::autoplot(m)
#'   }
#' }
#'
#' @export
#' @method autoplot nn_fit
autoplot.nn_fit = function(object, ...) {
    n = length(object$loss_history)
    epochs = seq_len(n)

    if (!is.null(object$val_loss_history)) {
        long = data.frame(
            epoch = c(epochs, epochs),
            loss = c(object$loss_history, object$val_loss_history),
            set = rep(c("Training", "Validation"), each = n)
        )
    } else {
        long = data.frame(
            epoch = epochs,
            loss = object$loss_history,
            set = "Training"
        )
    }

    p = ggplot2::ggplot(long, ggplot2::aes(x = epoch, y = loss, colour = set)) +
        ggplot2::geom_line() +
        ggplot2::labs(
            x = "Epoch",
            y = "Loss",
            title = "Training History",
            colour = NULL
        ) +
        ggplot2::theme_minimal()

    if (is.null(object$val_loss_history)) {
        p = p + ggplot2::guides(colour = "none")
    }

    if (!is.na(object$stopped_epoch)) {
        p = p +
            ggplot2::geom_vline(
                xintercept = object$stopped_epoch,
                linetype = "dashed",
                colour = "grey40"
            ) +
            ggplot2::annotate(
                "text",
                x = object$stopped_epoch,
                y = max(long$loss, na.rm = TRUE),
                label = paste("early stop:", object$stopped_epoch),
                hjust = -0.05,
                size = 3,
                colour = "grey40"
            )
    }

    p
}

#' @rdname autoplot.nn_fit
#' @param x A fitted model of class `nn_fit`, as returned by
#'   [train_nn()], [ffnn()], or [rnn()].
#' @export
#' @method plot nn_fit
plot.nn_fit = function(x, ...) {
    print(ggplot2::autoplot(x, ...))
    invisible(x)
}

#' Plot prediction diagnostics for a fitted neural network
#'
#' @description
#' Produces diagnostic plots comparing fitted values against actual (true)
#' response values for a fitted `nn_fit` model.
#'
#' - **Regression (single output)**: returns a named list with two panels --
#'   residuals vs fitted and actual vs fitted.
#' - **Regression (multi-output)**: returns a named list with one actual vs
#'   fitted panel per output column.
#' - **Classification**: returns a single confusion matrix heatmap.
#'
#' @param object A fitted model of class `nn_fit`, as returned by
#'   [train_nn()], [ffnn()], or [rnn()].
#' @param actual A vector of true response values, the same length as the
#'   training data used to fit `object`.
#' @param ... Additional arguments (currently unused).
#'
#' @return For regression, a named list of [ggplot2::ggplot()] objects (one per
#'   diagnostic panel). For classification, a single [ggplot2::ggplot()]
#'   confusion matrix heatmap.
#'
#' @examples
#' \donttest{
#'   if (torch::torch_is_installed()) {
#'     # Regression
#'     m = train_nn(
#'       as.matrix(iris[, 2:4]), iris$Sepal.Length,
#'       epochs = 5
#'     )
#'     autoplot_diagnostics(m, actual = iris$Sepal.Length)
#'
#'     # Classification
#'     m_cls = train_nn(
#'       as.matrix(iris[, 1:4]), iris$Species,
#'       epochs = 5
#'     )
#'     autoplot_diagnostics(m_cls, actual = iris$Species)
#'   }
#' }
#'
#' @export
autoplot_diagnostics = function(object, actual, ...) {
    if (missing(actual)) {
        cli::cli_abort(
            "Argument {.arg actual} is required: supply the vector of true response values."
        )
    }
    if (is.null(object$fitted)) {
        cli::cli_abort(c(
            "Prediction diagnostics require fitted values, which are not available for dataset fits ({.cls nn_fit_ds}).",
            "i" = "Use {.fn predict} with held-out data instead."
        ))
    }

    n_fitted = if (is.matrix(object$fitted)) nrow(object$fitted) else length(object$fitted)
    if (length(actual) != n_fitted) {
        cli::cli_abort(
            "{.arg actual} has length {length(actual)} but {.field fitted} has length {n_fitted}: they must match."
        )
    }

    if (object$is_classification) {
        actual_fac = factor(actual, levels = object$y_levels)
        tbl = table(actual = actual_fac, predicted = object$fitted)
        df = as.data.frame(tbl, stringsAsFactors = FALSE)
        names(df)[names(df) == "Freq"] = "n"
        row_totals = tapply(df$n, df$actual, sum)
        df$prop = df$n / row_totals[as.character(df$actual)]

        ggplot2::ggplot(df, ggplot2::aes(x = predicted, y = actual, fill = prop)) +
            ggplot2::geom_tile() +
            ggplot2::geom_text(ggplot2::aes(label = n), colour = "black") +
            ggplot2::scale_fill_gradient(low = "white", high = "steelblue", guide = "none") +
            ggplot2::labs(x = "Predicted", y = "Actual", title = "Confusion Matrix") +
            ggplot2::theme_minimal()
    } else if (object$no_y > 1L) {
        actual_mat = if (is.matrix(actual)) actual else matrix(as.numeric(actual), ncol = 1L)
        col_labels = colnames(object$fitted)
        if (is.null(col_labels)) col_labels = paste0("output_", seq_len(object$no_y))

        plots = lapply(seq_len(object$no_y), function(j) {
            df = data.frame(
                fitted_val = as.numeric(object$fitted[, j]),
                actual_val = as.numeric(actual_mat[, j])
            )
            ggplot2::ggplot(df, ggplot2::aes(x = fitted_val, y = actual_val)) +
                ggplot2::geom_point() +
                ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
                ggplot2::labs(
                    x = "Fitted values",
                    y = "Actual values",
                    title = col_labels[j]
                ) +
                ggplot2::theme_minimal()
        })
        names(plots) = col_labels

        cli::cli_inform(
            "Returning one plot per output column. Install {.pkg patchwork} to combine them."
        )
        plots
    } else {
        fitted_vals = as.numeric(object$fitted)
        df = data.frame(
            fitted_val = fitted_vals,
            residuals = as.numeric(actual) - fitted_vals,
            actual_val = as.numeric(actual)
        )

        p1 = ggplot2::ggplot(df, ggplot2::aes(x = fitted_val, y = residuals)) +
            ggplot2::geom_point() +
            ggplot2::geom_hline(yintercept = 0, linetype = "dashed") +
            ggplot2::labs(x = "Fitted values", y = "Residuals") +
            ggplot2::theme_minimal()

        p2 = ggplot2::ggplot(df, ggplot2::aes(x = fitted_val, y = actual_val)) +
            ggplot2::geom_point() +
            ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
            ggplot2::labs(x = "Fitted values", y = "Actual values") +
            ggplot2::theme_minimal()

        cli::cli_inform(
            "Returning two plots. Install {.pkg patchwork} and use {.code p[[1]] + p[[2]]} to combine them."
        )
        list(residuals_vs_fitted = p1, actual_vs_fitted = p2)
    }
}

#' @rdname autoplot_diagnostics
#' @export
plot_diagnostics = function(object, actual, ...) {
    result = autoplot_diagnostics(object, actual, ...)
    if (inherits(result, "gg")) {
        print(result)
    } else if (is.list(result)) {
        lapply(result, print)
    }
    invisible(object)
}
