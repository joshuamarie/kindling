#' Early Stopping Specification
#' 
#' @description
#' `early_stop()` is a helper function to be supplied on `early_stopping` arguments. 
#' 
#' @param patience Integer. Epochs to wait after last improvement. Default `5`.
#' @param min_delta Numeric. Minimum improvement to qualify as better. Default `1e-4`.
#' @param restore_best_weights Logical. Restore weights from best epoch. Default `TRUE`.
#' @param monitor Character. Metric to monitor. One of `"val_loss"` (default) or `"train_loss"`.
#'
#' @return An object of class `"early_stop_spec"`.
#' @export
early_stop = 
    function(
        patience = 5L,
        min_delta = 1e-4,
        restore_best_weights = TRUE,
        monitor = "val_loss"
    ) 
{
    monitor = rlang::arg_match(monitor, c("val_loss", "train_loss"))
    patience = vctrs::vec_cast(patience, integer())
    
    if (length(patience) != 1L || patience < 1L) {
        cli::cli_abort("{.arg patience} must be a single positive integer.")
    }
    if (!vctrs::vec_is(min_delta, double()) || length(min_delta) != 1L || min_delta < 0) {
        cli::cli_abort("{.arg min_delta} must be a single non-negative number.")
    }
    if (!vctrs::vec_is(restore_best_weights, logical()) || length(restore_best_weights) != 1L) {
        cli::cli_abort("{.arg restore_best_weights} must be a single logical value.")
    }
    
    vctrs::new_vctr(
        list(
            patience = patience,
            min_delta = min_delta,
            restore_best_weights = restore_best_weights,
            monitor = monitor
        ),
        class = "early_stop_spec"
    )
}

#' @export
print.early_stop_spec = function(x, ...) {
    cli::cli_h3("Early Stopping Spec")
    cli::cli_bullets(c(
        "*" = "Monitor: {x$monitor}",
        "*" = "Patience: {x$patience}",
        "*" = "Min delta: {x$min_delta}",
        "*" = "Restore best weights: {x$restore_best_weights}"
    ))
    invisible(x)
}
