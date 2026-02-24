#' Custom Activation Function Constructor
#'
#' Wraps a user-supplied function into a validated custom activation,
#' ensuring it accepts and returns a `torch_tensor`. Performs an eager
#' dry-run probe at *definition time* so errors surface early, and
#' wraps the function with a *call-time* type guard for safety.
#'
#' @param fn A function taking a single tensor argument and returning a tensor.
#'   E.g. `\(x) torch::torch_tanh(x)`.
#' @param probe Logical. If `TRUE` (default), runs a dry-run with a small
#'   dummy tensor at definition time to catch obvious errors early.
#' @param .name A string. Default is `"<custom>"`. 
#'
#' @return An object of class `c("custom_activation", "parameterized_activation")`,
#'   compatible with `act_funs()`.
#'
#' @examples
#' act_funs(relu, elu, new_act_fn(\(x) torch::torch_tanh(x)))
#' act_funs(new_act_fn(\(x) torch::nnf_silu(x)))
#'
#' @export
new_act_fn = function(fn, probe = TRUE, .name = "<custom>") {
    if (!is.function(fn)) {
        cli::cli_abort(c(
            "{.arg fn} must be a function.",
            i = "Use a lambda like {.code \\(x) torch::torch_tanh(x)}."
        ), class = "custom_activation_type_error")
    }
    
    fn_formals = formals(fn)
    if (length(fn_formals) < 1L) {
        cli_abort(c(
            "{.arg fn} must accept at least one argument (the input tensor).",
            i = "Use a lambda like {.code \\(x) torch::torch_tanh(x)}."
        ), class = "custom_activation_arity_error")
    }
    
    # ---- Dry-run probe at definition time ----
    if (probe) {
        if (!requireNamespace("torch", quietly = TRUE)) {
            cli::cli_warn(c(
                "{.pkg torch} is not installed; skipping dry-run probe for {.fn new_act_fn}.",
                i = "Type safety will only be enforced at call time."
            ))
        } else {
            # Prepare a tiny 1-D tensor to validate
            dummy = torch::torch_zeros(2L)  
            out = torch::with_no_grad(
                tryCatch(
                    fn(dummy),
                    error = function(e) {
                        cli_abort(c(
                            "Dry-run of custom activation function failed.",
                            x = "{e$message}",
                            i = "Ensure {.arg fn} accepts a {.cls torch_tensor} and returns one."
                        ), class = "custom_activation_probe_error")
                    }
                )
            )
            .assert_tensor_output(out, context = "Dry-run")
        }
    }
    
    # ---- Build the call-time type-guarded wrapper ----
    guarded_fn = function(x) {
        out = fn(x)
        .assert_tensor_output(out, context = "Custom activation output")
        out
    }
    
    structure(
        list(),
        act_fn = guarded_fn, 
        act_name = .name,
        class = c("custom_activation", "parameterized_activation")
    )
}

#' New custom activation function validator
#' @keywords internal
#' @noRd
.assert_tensor_output = function(x, context = "Output") {
    if (!inherits(x, "torch_tensor")) {
        cli_abort(c(
            "{context} must be a {.cls torch_tensor}.",
            x = "Got {.cls {class(x)}}.",
            i = "Ensure your function returns the result of a {.pkg torch} operation."
        ), class = "custom_activation_output_error")
    }
    invisible(x)
}