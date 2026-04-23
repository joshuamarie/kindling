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
