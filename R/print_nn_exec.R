#' @importFrom cli rule style_bold cat_line style_italic

concat = function(x) {
    if (is.null(x)) return("No act function applied")

    if (inherits(x, "parameterized_activation")) {
        fname = attr(x, "act_name")
        params = paste(
            names(x),
            format(unlist(x), trim = TRUE),
            sep = " = ",
            collapse = ", "
        )
        glue::glue("{fname}({params})")
    } else {
        as.character(x)
    }
}

#' Ordinal Suffixes Generator
#'
#' @description
#' This function is originally from `numform::f_ordinal()`.
#'
#' @param x Vector of numbers. Could be a string equivalent
#'
#' @references
#'
#' Rinker, T. W. (2021). numform: A publication style number and plot formatter
#' version 0.7.0. \url{https://github.com/trinker/numform}
#'
#' @examples
#' ordinal_gen(1:10)
#'
#' @rdname ordinal_gen
ordinal_gen = function(x) {
    if (is.numeric(x) & any(x < 1))
        warning("Values below 1 found.\nMay yield incorrect results")
    x = as.character(x)
    regs = c(th = "^1[1:2]$|[0456789]$", st = "(?<!^1)1$", nd = "(?<!^1)2$",
             rd = "(?<!^1)3$")
    for (i in seq_along(regs)) {
        locs = grepl(regs[i], x, perl = TRUE)
        x[locs] = paste0(x[locs], names(regs)[i])
    }
    x
}

#' Print method for ffnn_fit objects
#'
#' @param x An object of class "ffnn_fit"
#' @param ... Additional arguments (unused)
#' @export
print.ffnn_fit = function(x, ...) {
    # ---Title section---
    title = "Feedforward Neural Networks (MLP)"
    title_block = rule(center = title, line = "=")
    title_block = style_bold(title_block)

    cat_line("\n", title_block, "\n")

    hidden_units_str = if (is.null(x$hidden_neurons)) {
        "Not specified"
    } else {
        paste(as.character(x$hidden_neurons), collapse = ", ")
    }

    summary_data = data.frame(
        type = c(
            "NN Model Type",
            "Number of Epochs",
            "Hidden Layer Units",
            "Number of Hidden Layers",
            "Pred. Type",
            "n_predictors",
            "n_response"
        ),
        res = c(
            "FFNN",
            as.character(x$n_epochs),
            hidden_units_str,
            as.character(length(x$activations)),
            if (x$is_classification) "classification" else "regression",
            as.character(x$no_x),
            as.character(x$no_y)
        ),
        stringsAsFactors = FALSE
    )

    # Activation function details
    inner_acts = if (is.list(x$activations)) {
        vapply(x$activations, concat, character(1))
    } else if (!is.null(x$activations)) {
        rep(as.character(x$activations), length(x$activations))
    } else {
        "None"
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

    # ---Display summary table---
    heading1 = rule(left = "FFNN Model Summary", line = "-")
    heading1_block = style_italic(heading1)
    cat_line("\n", heading1_block, "\n\n")
    table_summary(summary_data, l = 5, center_table = TRUE)
    cat("\n\n")

    # ---Activation function summary---
    heading2 = rule(left = "Activation function", line = "-")
    heading2_block = style_italic(heading2)
    cat_line("\n", heading2_block, "\n\n")
    table_summary(act_data, l = 5, center_table = TRUE)

    invisible(x)
}

#' Print method for rnn_fit objects
#'
#' @param x An object of class "rnn_fit"
#' @param ... Additional arguments (unused)
#' @export
print.rnn_fit = function(x, ...) {
    rnn_type = x$rnn_type

    # ---Title section---
    title = switch(
        rnn_type,
        "rnn" = "Recurrent Neural Networks",
        "lstm" = "Long Short-Term Memory (RNN)",
        "gru" = "Gated Recurrent Unit (RNN)"
    )
    title_block = rule(center = title, line = "=")
    title_block = style_bold(title_block)

    cat_line("\n", title_block, "\n")

    hidden_units_str = if (is.null(x$hidden_neurons)) {
        "Not specified"
    } else {
        paste(as.character(x$hidden_neurons), collapse = ", ")
    }

    summary_data = data.frame(
        type = c(
            "NN Model Type",
            "RNN Type",
            "Bidirectional",
            "Number of Epochs",
            "Hidden Layer Units",
            "Number of Hidden Layers",
            "Pred. Type",
            "n_predictors",
            "n_response"
        ),
        res = c(
            "RNN",
            toupper(rnn_type),
            if (x$bidirectional) "Yes" else "No",
            as.character(x$n_epochs),
            hidden_units_str,
            as.character(length(x$activations)),
            if (x$is_classification) "classification" else "regression",
            as.character(x$no_x),
            as.character(x$no_y)
        ),
        stringsAsFactors = FALSE
    )

    inner_acts = if (is.list(x$activations)) {
        vapply(x$activations, concat, character(1))
    } else if (!is.null(x$activations)) {
        rep(as.character(x$activations), length(x$activations))
    } else {
        "None"
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

    # ---Display summary table---
    heading1 = rule(left = "RNN Model Summary", line = "-")
    heading1_block = style_italic(heading1)
    cat_line("\n", heading1_block, "\n\n")
    table_summary(summary_data, l = 7, center_table = TRUE)
    cat("\n\n")

    # ---Activation function summary---
    heading2 = rule(left = "Activation function", line = "-")
    heading2_block = style_italic(heading2)
    cat_line("\n", heading2_block, "\n\n")
    table_summary(act_data, l = 5, center_table = TRUE)
    cat("\n")

    invisible(x)
}
