#' @keywords internal
.validate_loss_fn = function(fn, call = rlang::caller_env()) {
    # ---- Arity check ----
    if (length(formals(fn)) < 2L) {
        cli::cli_abort(c(
            "Custom loss function must accept at least 2 arguments: {.arg input} and {.arg target}.",
            i = "Use a lambda like {.code \\(input, target) torch::nnf_mse_loss(input, target)}."
        ), class = "loss_fn_arity_error", call = call)
    }
    
    # ---- Dry-run probe ----
    if (requireNamespace("torch", quietly = TRUE)) {
        dummy_input = torch::torch_randn(c(2L, 1L))
        dummy_target = torch::torch_randn(c(2L, 1L))
        
        out = tryCatch(
            fn(dummy_input, dummy_target),
            error = function(e) {
                cli::cli_abort(c(
                    "Dry-run of custom loss function failed.",
                    x = "{e$message}",
                    i = "Ensure your function accepts two {.cls torch_tensor}s and returns a scalar tensor."
                ), class = "loss_fn_probe_error", call = call)
            }
        )
        
        if (!inherits(out, "torch_tensor")) {
            cli::cli_abort(c(
                "Custom loss function must return a {.cls torch_tensor}.",
                x = "Got {.cls {class(out)}}.",
                i = "Ensure your function returns the result of a {.pkg torch} operation."
            ), class = "loss_fn_output_error", call = call)
        }
        
        if (out$numel() != 1L) {
            cli::cli_warn(c(
                "Custom loss function returned a non-scalar tensor with {out$numel()} elements.",
                i = "Loss functions should return a scalar. Did you forget to reduce (e.g. {.code $mean()})?"
            ), class = "loss_fn_shape_warning", call = call)
        }
    }
    
    # ---- Build the call-time type-guarded wrapper ----
    function(input, target) {
        out = fn(input, target)
        if (!inherits(out, "torch_tensor")) {
            cli::cli_abort(c(
                "Custom loss function must return a {.cls torch_tensor}.",
                x = "Got {.cls {class(out)}}."
            ), class = "loss_fn_output_error", call = rlang::caller_env())
        }
        out
    }
}
