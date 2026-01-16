#' Summarize and Display a Two-Column Data Frame as a Formatted Table
#'
#' This function takes a two-column data frame and formats it into a summary-like table.
#' The table can be optionally split into two parts, centered, and given a title.
#' It is useful for displaying summary information in a clean, tabular format.
#' The function also supports styling with ANSI colors and text formatting through
#' the `{cli}` package and column alignment options.
#'
#' @param data A data frame with exactly two columns. The data to be summarized and displayed.
#' @param title A character string. An optional title to be displayed above the table.
#' @param l An integer. The number of rows to include in the left part of a split table.
#'           If `NULL`, the table is not split.
#' @param header A logical value. If `TRUE`, the column names of `data` are displayed as a header.
#' @param center_table A logical value. If `TRUE`, the table is centered in the terminal.
#' @param border_char Character used for borders. Default is `"\u2500"`.
#' @param style A list controlling the visual styling of table elements using ANSI formatting.
#'   Can include the following components:
#'   - `left_col`: Styling for the left column values.
#'   - `right_col`: Styling for the right column values.
#'   - `border_text`: Styling for the border.
#'   - `title`: Styling for the title.
#'   - `sep`: Separator character between left and right column.
#'
#'   Each style component can be either a predefined style string (e.g., "blue", "red_italic", "bold")
#'   or a function that takes a context list with/without a `value` element and returns the styled text.
#' @param align Controls the alignment of column values.
#'   Can be specified in three ways:
#'   - A single string: affects only the left column (e.g., "left", "center", "right").
#'   - A vector of two strings: affects both columns in order (e.g., c("left", "right")).
#'   - A list with named components: explicitly specifies alignment for each column
#'
#' @param ... Additional arguments (currently unused).
#'
#' @return This function does not return a value. It prints the formatted table to the console.
#'
#' @examples
#' # Create a sample data frame
#' df = data.frame(
#'     Category = c("A", "B", "C", "D", "E"),
#'     Value = c(10, 20, 30, 40, 50)
#' )
#'
#' # Display the table with a title and header
#' table_summary(df, title = "Sample Table", header = TRUE)
#'
#' # Split the table after the second row and center it
#' table_summary(df, l = 2, center_table = TRUE)
#'
#' # Use styling and alignment
#' table_summary(
#'     df, header = TRUE,
#'     style = list(
#'         left_col = "blue_bold",
#'         right_col = "red",
#'         title = "green",
#'         border_text = "yellow"
#'     ),
#'     align = c("center", "right")
#' )
#'
#' # Use custom styling with lambda functions
#' table_summary(
#'     df, header = TRUE,
#'     style = list(
#'         left_col = \(ctx) cli::col_red(ctx), # ctx$value is another option
#'         right_col = \(ctx) cli::col_blue(ctx)
#'     ),
#'     align = list(left_col = "left", right_col = "right")
#' )
#'
#' @export
table_summary = function(
    data,
    title = NULL,
    l = NULL,
    header = FALSE,
    center_table = FALSE,
    border_char = "-",
    style = list(),
    align = NULL,
    ...
) {
    if (!is.data.frame(data) || ncol(data) != 2) {
        stop("Input must be a data frame with exactly 2 columns")
    }

    col_names = colnames(data)
    data_matrix = as.matrix(data)

    n_rows = nrow(data)
    is_split = !is.null(l) && l < n_rows

    sep_width = 4
    if (!is.null(style) && !is.null(style$sep)) {
        sep_width = nchar(paste0(" ", style$sep, " "))
    }

    if (is_split) {
        left_table = data_matrix[1:l, ]
        right_table = data_matrix[(l+1):n_rows, ]

        left_left_width = max(nchar(left_table[, 1]), nchar(col_names[1]))
        left_right_width = max(nchar(left_table[, 2]), nchar(col_names[2]))
        right_left_width = max(nchar(right_table[, 1]), nchar(col_names[1]))
        right_right_width = max(nchar(right_table[, 2]), nchar(col_names[2]))

        left_table_width = left_left_width + left_right_width + 4 + sep_width
        right_table_width = right_left_width + right_right_width + 4 + sep_width
        full_width = left_table_width + right_table_width
    } else {
        left_width = max(nchar(data_matrix[, 1]), nchar(col_names[1]))
        right_width = max(nchar(data_matrix[, 2]), nchar(col_names[2]))
        full_width = left_width + right_width + 4 + sep_width
    }

    horizontal_line = strrep(border_char, full_width)
    styled_horizontal_line = horizontal_line

    if (!is.null(style) && !is.null(style$border_text)) {
        if (is.function(style$border_text)) {
            styled_horizontal_line = style$border_text(horizontal_line)
        } else if (is.character(style$border_text)) {
            styled_horizontal_line = switch(
                style$border_text,
                bold = cli::style_bold(horizontal_line),
                italic = cli::style_italic(horizontal_line),
                blue = cli::col_blue(horizontal_line),
                red = cli::col_red(horizontal_line),
                green = cli::col_green(horizontal_line),
                yellow = cli::col_yellow(horizontal_line),
                blue_bold = cli::col_blue(cli::style_bold(horizontal_line)),
                red_italic = cli::col_red(cli::style_italic(horizontal_line)),
                horizontal_line
            )
        }
    }

    prefix = ""
    if (center_table) {
        term_width = tryCatch({
            as.numeric(system("tput cols", intern = TRUE))
        }, error = function(e) {
            as.double(options("width"))
        })

        left_padding = max(0, floor((term_width - full_width) / 2))
        prefix = strrep(" ", left_padding)
    }

    if (!is.null(title)) {
        formatted_title = align_test(title, full_width)

        if (!is.null(style) && !is.null(style$title)) {
            if (is.function(style$title)) {
                formatted_title = style$title(formatted_title)
            } else if (is.character(style$title)) {
                formatted_title = switch(
                    style$title,
                    bold = cli::style_bold(formatted_title),
                    italic = cli::style_italic(formatted_title),
                    blue = cli::col_blue(formatted_title),
                    red = cli::col_red(formatted_title),
                    green = cli::col_green(formatted_title),
                    yellow = cli::col_yellow(formatted_title),
                    blue_bold = cli::col_blue(cli::style_bold(formatted_title)),
                    formatted_title
                )
            }
        }

        if (center_table) {
            cat("\n", prefix, formatted_title, "\n", sep = "")
        } else {
            cat("\n", formatted_title, "\n", sep = "")
        }
    }

    cat(prefix, styled_horizontal_line, "\n", sep = "")

    if (header) {
        if (is_split) {
            header_row = paste0(
                format_row_summary(col_names[1], col_names[2], left_left_width, left_right_width, style = style),
                "  ",
                format_row_summary(col_names[1], col_names[2], right_left_width, right_right_width, style = style)
            )
        } else {
            header_row = format_row_summary(col_names[1], col_names[2], left_width, right_width, style = style)
        }
        cat(prefix, header_row, "\n", sep = "")
        cat(prefix, styled_horizontal_line, "\n", sep = "")
    }

    if (!is_split) {
        for (i in 1:n_rows) {
            row_output = format_row_summary(
                data_matrix[i, 1],
                data_matrix[i, 2],
                left_width,
                right_width,
                align = align,
                style = style
            )
            cat(prefix, row_output, "\n", sep = "")
        }
    } else {
        for (i in 1:max(l, nrow(right_table))) {
            left_row = if (i <= l) left_table[i, ] else c("", "")
            right_row = if (i <= nrow(right_table)) right_table[i, ] else c("", "")

            left_output = format_row_summary(
                left_row[1],
                left_row[2],
                left_left_width,
                left_right_width,
                align = align,
                style = style
            )

            right_output = format_row_summary(
                right_row[1],
                right_row[2],
                right_left_width,
                right_right_width,
                align = align,
                style = style
            )

            row_output = paste0(left_output, "  ", right_output)
            cat(prefix, row_output, "\n", sep = "")
        }
    }

    cat(prefix, styled_horizontal_line, "\n", sep = "")
}

align_test = function(text, width) {
    padding = width - nchar(text)
    left_pad = floor(padding / 2)
    right_pad = ceiling(padding / 2)
    paste0(strrep(" ", left_pad), text, strrep(" ", right_pad))
}

format_row_summary = function(
    left,
    right,
    left_width,
    right_width,
    align = NULL,
    style = NULL
) {

    format_aligned = function(text, width, alignment) {
        switch(
            alignment,
            left = sprintf("%-*s", width, text),
            right = sprintf("%*s",  width, text),
            center = align_test(text, width),
            sprintf("%-*s", width, text)
        )
    }

    apply_cli_style = function(text, style_name) {
        switch(
            style_name,
            bold = cli::style_bold(text),
            italic = cli::style_italic(text),
            blue = cli::col_blue(text),
            red = cli::col_red(text),
            green = cli::col_green(text),
            yellow = cli::col_yellow(text),
            blue_bold = cli::col_blue(cli::style_bold(text)),
            red_italic = cli::col_red(cli::style_italic(text)),
            text
        )
    }

    left_align  = "left"
    right_align = "right"

    if (!is.null(align)) {
        if (is.character(align) && length(align) == 1) {
            left_align = right_align = align
        } else if (is.character(align) && length(align) == 2) {
            left_align  = align[1]
            right_align = align[2]
        } else if (is.list(align)) {
            if (!is.null(align$left_col))  left_align  = align$left_col
            if (!is.null(align$right_col)) right_align = align$right_col
        }
    }

    left_formatted = format_aligned(left,  left_width,  left_align)
    right_formatted = format_aligned(right, right_width, right_align)

    if (!is.null(style)) {
        if (!is.null(style$left_col)) {
            if (is.function(style$left_col)) {
                left_formatted = style$left_col(list(value = left_formatted))
            } else if (is.character(style$left_col)) {
                left_formatted = apply_cli_style(left_formatted, style$left_col)
            }
        }

        if (!is.null(style$right_col)) {
            if (is.function(style$right_col)) {
                right_formatted = style$right_col(list(value = right_formatted))
            } else if (is.character(style$right_col)) {
                right_formatted = apply_cli_style(right_formatted, style$right_col)
            }
        }
    }

    sep = "    "
    if (!is.null(style) && !is.null(style$sep)) {
        sep = paste0(" ", style$sep, " ")
    }

    paste0("  ", left_formatted, sep, right_formatted)
}
