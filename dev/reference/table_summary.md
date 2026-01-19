# Summarize and Display a Two-Column Data Frame as a Formatted Table

This function takes a two-column data frame and formats it into a
summary-like table. The table can be optionally split into two parts,
centered, and given a title. It is useful for displaying summary
information in a clean, tabular format. The function also supports
styling with ANSI colors and text formatting through the `{cli}` package
and column alignment options.

## Usage

``` r
table_summary(
  data,
  title = NULL,
  l = NULL,
  header = FALSE,
  center_table = FALSE,
  border_char = "-",
  style = list(),
  align = NULL,
  ...
)
```

## Arguments

- data:

  A data frame with exactly two columns. The data to be summarized and
  displayed.

- title:

  A character string. An optional title to be displayed above the table.

- l:

  An integer. The number of rows to include in the left part of a split
  table. If `NULL`, the table is not split.

- header:

  A logical value. If `TRUE`, the column names of `data` are displayed
  as a header.

- center_table:

  A logical value. If `TRUE`, the table is centered in the terminal.

- border_char:

  Character used for borders. Default is `"\u2500"`.

- style:

  A list controlling the visual styling of table elements using ANSI
  formatting. Can include the following components:

  - `left_col`: Styling for the left column values.

  - `right_col`: Styling for the right column values.

  - `border_text`: Styling for the border.

  - `title`: Styling for the title.

  - `sep`: Separator character between left and right column.

  Each style component can be either a predefined style string (e.g.,
  "blue", "red_italic", "bold") or a function that takes a context list
  with/without a `value` element and returns the styled text.

- align:

  Controls the alignment of column values. Can be specified in three
  ways:

  - A single string: affects only the left column (e.g., "left",
    "center", "right").

  - A vector of two strings: affects both columns in order (e.g.,
    c("left", "right")).

  - A list with named components: explicitly specifies alignment for
    each column

- ...:

  Additional arguments (currently unused).

## Value

This function does not return a value. It prints the formatted table to
the console.

## Examples

``` r
# Create a sample data frame
df = data.frame(
    Category = c("A", "B", "C", "D", "E"),
    Value = c(10, 20, 30, 40, 50)
)

# Display the table with a title and header
table_summary(df, title = "Sample Table", header = TRUE)
#> 
#>     Sample Table     
#> ---------------------
#>   Category    Value
#> ---------------------
#>   A              10
#>   B              20
#>   C              30
#>   D              40
#>   E              50
#> ---------------------

# Split the table after the second row and center it
table_summary(df, l = 2, center_table = TRUE)
#> Warning: running command 'tput cols' had status 2
#> ------------------------------------------
#>   A              10    C              30
#>   B              20    D              40
#>                        E              50
#> ------------------------------------------

# Use styling and alignment
table_summary(
    df, header = TRUE,
    style = list(
        left_col = "blue_bold",
        right_col = "red",
        title = "green",
        border_text = "yellow"
    ),
    align = c("center", "right")
)
#> ---------------------
#>   Category    Value
#> ---------------------
#>      A           10
#>      B           20
#>      C           30
#>      D           40
#>      E           50
#> ---------------------

# Use custom styling with lambda functions
table_summary(
    df, header = TRUE,
    style = list(
        left_col = \(ctx) cli::col_red(ctx), # ctx$value is another option
        right_col = \(ctx) cli::col_blue(ctx)
    ),
    align = list(left_col = "left", right_col = "right")
)
#> ---------------------
#>   Category    Value
#> ---------------------
#>   A              10
#>   B              20
#>   C              30
#>   D              40
#>   E              50
#> ---------------------
```
