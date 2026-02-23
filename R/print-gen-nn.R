#' Print method for nn_fit objects
#'
#' @param x An object of class `"nn_fit"`
#' @param ... Additional arguments (unused)
#'
#' @return No return value, called for side effects (printing model summary)
#'
#' @keywords internal
#' @export
print.nn_fit = function(x, ...) {
    # ---- Title ----
    title = "Generalized Neural Network"
    title_block = style_bold(rule(center = title, line = "="))
    cat_line("\n", title_block, "\n")
    
    hidden_units_str = if (is.null(x$hidden_neurons)) {
        "Not specified"
    } else {
        paste(as.character(x$hidden_neurons), collapse = ", ")
    }
    
    nn_model_type = if (!is.null(x$arch)) {
        nn_layer = x$arch$nn_layer %||% "nn_linear"
        glue("{x$arch$nn_name} ({nn_layer})")
    } else {
        "FFNN"
    }
    
    # ---- Table 1: Model Summary ----
    summary_data = data.frame(
        type = c(
            "NN Model Type",
            "Number of Epochs",
            "Hidden Layer Units",
            "Number of Hidden Layers",
            "Pred. Type",
            "n_inputs",
            "n_outputs",
            "reg.",
            "Device"
        ),
        res = c(
            nn_model_type,
            as.character(x$n_epochs),
            hidden_units_str,
            as.character(length(x$hidden_neurons)),
            if (x$is_classification) "classification" else "regression",
            as.character(x$no_x),
            as.character(x$no_y),
            if (x$penalty == 0) "None" else glue("[\u03BB = {x$penalty}, \u03B1 = {x$mixture}]"),
            x$device
        ),
        stringsAsFactors = FALSE
    )
    
    heading1 = style_italic(rule(left = "Model Summary", line = "-"))
    cat_line("\n", heading1, "\n\n")
    table_summary(summary_data, l = 5, center_table = TRUE, style = list(sep = ":  "))
    cat("\n\n")
    
    # ---- Table 2: Activation Functions ----
    inner_acts = if (is.list(x$activations)) {
        vapply(x$activations, concat, character(1))
    } else if (!is.null(x$activations)) {
        if (length(x$activations) == 1) {
            rep(as.character(x$activations), length(x$hidden_neurons))
        } else {
            as.character(x$activations)
        }
    } else {
        rep("None", length(x$hidden_neurons))
    }
    outer_acts = concat(x$output_activation)
    
    act_data = data.frame(
        layer = c(
            paste(
                ordinal_gen(seq_along(inner_acts)), "Layer",
                paste0("{", x$hidden_neurons, "}")
            ),
            "Output Activation"
        ),
        infos = c(inner_acts, outer_acts),
        stringsAsFactors = FALSE
    )
    
    heading2 = style_italic(rule(left = "Activation Functions", line = "-"))
    cat_line("\n", heading2, "\n\n")
    table_summary(act_data, l = 5, center_table = TRUE, style = list(sep = ":  "))
    cat("\n\n")
    
    # ---- Table 3: Architecture Spec ----
    flag = function(val) if (!is.null(val)) "yes" else "N/A"
    
    arch = x$arch
    arch_data = data.frame(
        type = c(
            "nn_layer",
            "out_nn_layer",
            "nn_layer_args",
            "layer_arg_fn",
            "forward_extract",
            "before_output_transform",
            "after_output_transform",
            "last_layer_args",
            "input_transform"
        ),
        res = if (is.null(arch)) {
            rep("N/A", 9L)
        } else {
            c(
                arch$nn_layer %||% "nn_linear (default)",
                arch$out_nn_layer %||% "N/A",
                if (length(arch$nn_layer_args) > 0) "yes" else "N/A",
                flag(arch$layer_arg_fn),
                flag(arch$forward_extract),
                flag(arch$before_output_transform),
                flag(arch$after_output_transform),
                if (length(arch$last_layer_args) > 0) "yes" else "N/A",
                flag(arch$input_transform)
            )
        },
        stringsAsFactors = FALSE
    )
    
    heading3 = style_italic(rule(left = "Architecture Spec", line = "-"))
    cat_line("\n", heading3, "\n\n")
    table_summary(arch_data, l = 5, center_table = TRUE, style = list(sep = ":  "))
    cat("\n")
    
    invisible(x)
}
