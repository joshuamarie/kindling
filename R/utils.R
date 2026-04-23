#' @importFrom stats reorder setNames
#' @importFrom rlang :=
#' @importFrom lifecycle badge
#' @importFrom ggplot2 autoplot
NULL

# To avoid R CMD check notes about undefined global variables
utils::globalVariables(c(
    "x_names", "rel_imp", "object", "new_data", "nn_module", "fit_class",
    "epoch", "loss", "set",
    "fitted_val", "residuals", "actual_val",
    "actual", "predicted", "n", "prop"
))
