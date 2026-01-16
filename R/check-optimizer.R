#' Validate Optimizer Exists
#'
#' @param act_name Character. Activation function name (without prefix).
#'
#' @importFrom cli cli_abort
#' @importFrom rlang is_installed
#'
#' @noRd
validate_optimizer = function(optimizer) {
    if (!is_installed("torch")) {
        cli_abort(c(
            "{.pkg torch} package is required but not installed.",
            i = "Install it with: {.code install.packages('torch')}"
        ), class = "torch_missing_error")
    }

    prefix = "optim_"
    fn_name = paste0(prefix, optimizer)

    if (!exists(fn_name, where = asNamespace("torch"), mode = "function")) {
        cli_abort(c(
            "Optimizer {.fn {fn_name}} does not exist in {.pkg torch}.",
            i = "Available functions start with {.code {prefix}}.",
            i = "Check {.code ?torch::optim_adam} for examples."
        ), class = "optimizer_not_found_error")
    }
}
