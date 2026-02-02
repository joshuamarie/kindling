#' Depth-Aware Grid Generation for Neural Networks
#'
#' `grid_depth()` extends standard grid generation to support multi-layer
#' neural network architectures. It creates heterogeneous layer configurations
#' by generating list columns for `hidden_neurons` and `activations`.
#'
#' @param x A `parameters` object, list, workflow, or model spec. Can also be
#'   a single `param` object if `...` contains additional param objects.
#' @param ... One or more `param` objects (e.g., `hidden_neurons()`, `epochs()`).
#'   If `x` is a `parameters` object, `...` is ignored. None of the objects can
#'   have `unknown()` values.
#' @param n_hlayer Integer vector specifying number of hidden layers to generate
#'   (e.g., `2:4` for 2, 3, or 4 layers), or a `param` object created with `n_hlayers()`.
#'   Default is 2.
#' @param size Integer. Number of parameter combinations to generate.
#' @param type Character. Type of grid: "regular", "random", "latin_hypercube",
#'   "max_entropy", or "audze_eglais".
#' @param original Logical. Should original parameter ranges be used?
#' @param levels Integer. Levels per parameter for regular grids.
#' @param variogram_range Numeric. Range for audze_eglais design.
#' @param iter Integer. Iterations for max_entropy optimization.
#'
#' @details
#' This function is specifically for `{kindling}` models. The `n_hlayer` parameter
#' determines network depth and creates list columns for `hidden_neurons` and
#' `activations`, where each element is a vector of length matching the sampled depth.
#'
#' When `n_hlayer` is a parameter object (created with `n_hlayers()`), it will be
#' treated as a tunable parameter and sampled according to its defined range.
#'
#' @return A tibble with list columns for `hidden_neurons` and `activations`,
#'   where each element is a vector of length `n_hlayer`.
#'
#' @examples
#' \donttest{
#' \dontrun{
#' library(dials)
#' library(workflows)
#' library(tune)
#'
#' # Method 1: Fixed depth
#' grid = grid_depth(
#'     hidden_neurons(c(32L, 128L)),
#'     activations(c("relu", "elu")),
#'     epochs(c(50L, 200L)),
#'     n_hlayer = 2:3,
#'     type = "random",
#'     size = 20
#' )
#'
#' # Method 2: Tunable depth using parameter object
#' grid = grid_depth(
#'     hidden_neurons(c(32L, 128L)),
#'     activations(c("relu", "elu")),
#'     epochs(c(50L, 200L)),
#'     n_hlayer = n_hlayers(range = c(2L, 4L)),
#'     type = "random",
#'     size = 20
#' )
#'
#' # Method 3: From workflow
#' wf = workflow() |>
#'     add_model(mlp_kindling(hidden_neurons = tune(), activations = tune())) |>
#'     add_formula(y ~ .)
#' grid = grid_depth(wf, n_hlayer = 2:4, type = "latin_hypercube", size = 15)
#' }
#' }
#'
#' @rdname grid_depth
#' @export
grid_depth = 
    function(
        x,
        ...,
        n_hlayer = 2L,
        size = 5L,
        type = c("regular", "random", "latin_hypercube", "max_entropy", "audze_eglais"),
        original = TRUE,
        levels = 3L,
        variogram_range = 0.5,
        iter = 1000L
    ) {
        UseMethod("grid_depth")
    }

#' @export
#' @rdname grid_depth
grid_depth.parameters = 
    function(
        x,
        ...,
        n_hlayer = 2L,
        size = 5L,
        type = c("regular", "random", "latin_hypercube", "max_entropy", "audze_eglais"),
        original = TRUE,
        levels = 3L,
        variogram_range = 0.5,
        iter = 1000L
    ) {
        
        type = rlang::arg_match(type)
        param_list = setNames(x$object, x$name)
        
        # Extract depth parameter if present
        depth_info = extract_depth_param(n_hlayer, param_list, levels)
        n_hlayer_vals = depth_info$values
        
        # Remove n_hlayers from param_list if it was there
        param_list = param_list[names(param_list) != "n_hlayers"]
        
        has_neurons = "hidden_neurons" %in% names(param_list)
        has_activations = "activations" %in% names(param_list)
        
        neuron_param = if (has_neurons) param_list[["hidden_neurons"]] else NULL
        activation_param = if (has_activations) param_list[["activations"]] else NULL
        
        scalar_names = setdiff(names(param_list), c("hidden_neurons", "activations"))
        scalar_params = param_list[scalar_names]
        
        generate_grid(
            neuron_param = neuron_param,
            activation_param = activation_param,
            n_hlayer = n_hlayer_vals,
            scalar_params = scalar_params,
            type = type,
            size = size,
            levels = levels,
            original = original,
            variogram_range = variogram_range,
            iter = iter
        )
    }

#' @export
#' @rdname grid_depth
grid_depth.list = 
    function(
        x,
        ...,
        n_hlayer = 2L,
        size = 5L,
        type = c("regular", "random", "latin_hypercube", "max_entropy", "audze_eglais"),
        original = TRUE,
        levels = 3L,
        variogram_range = 0.5,
        iter = 1000L
    ) {
        params = rlang::exec(dials::parameters, !!!x)
        grid_depth.parameters(
            params,
            n_hlayer = n_hlayer,
            size = size,
            type = type,
            original = original,
            levels = levels,
            variogram_range = variogram_range,
            iter = iter
        )
    }

#' @export
#' @rdname grid_depth
grid_depth.workflow = 
    function(
        x,
        ...,
        n_hlayer = 2L,
        size = 5L,
        type = c("regular", "random", "latin_hypercube", "max_entropy", "audze_eglais"),
        original = TRUE,
        levels = 3L,
        variogram_range = 0.5,
        iter = 1000L
    ) {
        params = workflows::extract_parameter_set_dials(x)
        grid_depth.parameters(
            params,
            n_hlayer = n_hlayer,
            size = size,
            type = type,
            original = original,
            levels = levels,
            variogram_range = variogram_range,
            iter = iter
        )
    }

#' @export
#' @rdname grid_depth
grid_depth.model_spec = 
    function(
        x,
        ...,
        n_hlayer = 2L,
        size = 5L,
        type = c("regular", "random", "latin_hypercube", "max_entropy", "audze_eglais"),
        original = TRUE,
        levels = 3L,
        variogram_range = 0.5,
        iter = 1000L
    ) {
        tunable_params = tune::tunable(x)
        param_list = purrr::map(seq_len(nrow(tunable_params)), function(i) {
            call_info = tunable_params$call_info[[i]]
            rlang::exec(call_info$fun, !!!call_info$args, .env = asNamespace(call_info$pkg))
        })
        names(param_list) = tunable_params$name
        params = rlang::exec(dials::parameters, !!!param_list)
        
        grid_depth.parameters(
            params,
            n_hlayer = n_hlayer,
            size = size,
            type = type,
            original = original,
            levels = levels,
            variogram_range = variogram_range,
            iter = iter
        )
    }

#' @export
#' @rdname grid_depth
grid_depth.param = 
    function(
        x,
        ...,
        n_hlayer = 2L,
        size = 5L,
        type = c("regular", "random", "latin_hypercube", "max_entropy", "audze_eglais"),
        original = TRUE,
        levels = 3L,
        variogram_range = 0.5,
        iter = 1000L
    ) {
        
        dots = rlang::list2(...)
        all_params = c(list(x), dots)
        param_objects = purrr::keep(all_params, ~ inherits(.x, "param"))
        
        if (length(param_objects) == 0) {
            cli::cli_abort(
                c(
                    "Could not find any param objects.",
                    "i" = "Provide param objects like {.code hidden_neurons()}, {.code epochs()}, etc."
                )
            )
        }
        
        params = rlang::exec(dials::parameters, !!!param_objects)
        grid_depth.parameters(
            params,
            n_hlayer = n_hlayer,
            size = size,
            type = type,
            original = original,
            levels = levels,
            variogram_range = variogram_range,
            iter = iter
        )
    }

#' @export
#' @rdname grid_depth
grid_depth.default = 
    function(
        x,
        ...,
        n_hlayer = 2L,
        size = 5L,
        type = c("regular", "random", "latin_hypercube", "max_entropy", "audze_eglais"),
        original = TRUE,
        levels = 3L,
        variogram_range = 0.5,
        iter = 1000L
    ) {
        
        cli::cli_abort(
            c(
                "No method for object of class {.cls {class(x)}}",
                "i" = "Provide param objects, a {.cls parameters} object, {.cls workflow}, or {.cls model_spec}."
            )
        )
    }

#' Extract depth parameter values from n_hlayer argument
#'
#' @param n_hlayer Either an integer vector or a param object
#' @param param_list List of parameters (for extracting n_hlayers if present)
#' @param levels Number of levels for regular grids
#'
#' @return List with `values` component containing integer vector of depths
#' @keywords internal
extract_depth_param = function(n_hlayer, param_list = list(), levels = 3L) {
    if ("n_hlayers" %in% names(param_list)) {
        depth_param = param_list[["n_hlayers"]]
        depth_vals = extract_param_range(depth_param, levels)
        return(list(values = as.integer(depth_vals)))
    }
    
    if (inherits(n_hlayer, "param")) {
        depth_vals = extract_param_range(n_hlayer, levels)
        return(list(values = as.integer(depth_vals)))
    }
    
    out = list(values = as.integer(n_hlayer))
    out
}

generate_grid = 
    function(
        neuron_param, activation_param, n_hlayer,
        scalar_params, type, size, levels, original,
        variogram_range, iter
    ) {
    if (is.null(neuron_param) && is.null(activation_param)) {
        if (length(scalar_params) == 0) {
            cli::cli_abort("No parameters provided for grid generation.")
        }
        return(make_scalar_grid(scalar_params, size, levels, type, original))
    }
    
    n_hlayer = as.integer(n_hlayer)
    out = switch(
        type,
        regular = generate_regular_grid(
            neuron_param, activation_param, n_hlayer,
            scalar_params, levels, original
        ),
        random = generate_random_grid(
            neuron_param, activation_param, n_hlayer,
            scalar_params, size, original
        ),
        latin_hypercube = generate_lhs_grid(
            neuron_param, activation_param, n_hlayer,
            scalar_params, size, original
        ),
        max_entropy = generate_sfd_grid(
            neuron_param, activation_param, n_hlayer,
            scalar_params, size, "max_entropy", variogram_range, iter, original
        ),
        audze_eglais = generate_sfd_grid(
            neuron_param, activation_param, n_hlayer,
            scalar_params, size, "audze_eglais", variogram_range, iter, original
        )
    )
    
    # Force scalar, not list
    if (all(n_hlayer == 1)) {
        if ("hidden_neurons" %in% names(out)) {
            out$hidden_neurons = purrr::map_int(out$hidden_neurons, 1)
        }
        if ("activations" %in% names(out)) {
            out$activations = purrr::map_chr(out$activations, 1)
        }
    }
    
    out
}

generate_regular_grid = 
    function(
        neuron_param, activation_param, n_hlayer,
        scalar_params, levels, original
    ) {
    neuron_vals = extract_param_range(neuron_param, levels, original)
    activation_vals = extract_param_values(activation_param)
    
    arch_grid = purrr::map_dfr(n_hlayer, function(depth) {
        if (!is.null(neuron_vals) && !is.null(activation_vals)) {
            expand_architecture(neuron_vals, activation_vals, depth)
        } else if (!is.null(neuron_vals)) {
            expand_neurons_only(neuron_vals, depth)
        } else {
            expand_activations_only(activation_vals, depth)
        }
    })
    
    if (length(scalar_params) > 0) {
        scalar_grid = dials::grid_regular(
            dials::parameters(scalar_params),
            levels = levels,
            original = original
        )
        tidyr::crossing(arch_grid, scalar_grid)
    } else {
        arch_grid
    }
}

generate_random_grid = 
    function(
        neuron_param, activation_param, n_hlayer,
        scalar_params, size, original
    ) {
    activation_vals = extract_param_values(activation_param)
    
    purrr::map_dfr(seq_len(size), function(i) {
        depth = safe_sample(n_hlayer, 1)
        row_data = list()
        
        if (!is.null(neuron_param)) {
            neurons = sample_from_param(neuron_param, depth, original)
            row_data$hidden_neurons = list(as.integer(neurons))
        }
        
        if (!is.null(activation_vals)) {
            activations = sample(activation_vals, depth, replace = TRUE)
            row_data$activations = list(as.character(activations))
        }
        
        if (length(scalar_params) > 0) {
            scalars = dials::grid_random(
                dials::parameters(scalar_params),
                size = 1,
                original = original
            )
            row_data = c(row_data, as.list(scalars))
        }
        
        tibble::as_tibble(row_data)
    })
}

generate_lhs_grid = 
    function(
        neuron_param, activation_param, n_hlayer,
        scalar_params, size, original
    ) {
    if (!requireNamespace("lhs", quietly = TRUE)) {
        cli::cli_abort("Package {.pkg lhs} required for Latin Hypercube sampling.")
    }
    
    has_neurons = !is.null(neuron_param)
    activation_vals = extract_param_values(activation_param)
    max_depth = max(n_hlayer)
    
    n_numeric_arch = if (has_neurons) max_depth else 0
    n_numeric_scalars = count_numeric_params(scalar_params)
    n_dims = n_numeric_arch + n_numeric_scalars
    
    if (n_dims == 0) {
        return(generate_random_grid(
            neuron_param, activation_param, n_hlayer,
            scalar_params, size, original
        ))
    }
    
    design = lhs::randomLHS(size, n_dims)
    depths = safe_sample(n_hlayer, size, replace = TRUE)
    
    results = vector("list", size)
    
    for (i in seq_len(size)) {
        depth = depths[i]
        row = design[i, ]
        row_data = list()
        
        if (has_neurons) {
            neuron_indices = row[seq_len(n_numeric_arch)]
            neurons = decode_neurons_from_design(
                neuron_param, 
                neuron_indices[seq_len(depth)],
                original
            )
            row_data$hidden_neurons = list(as.integer(neurons))
        }
        
        if (!is.null(activation_vals)) {
            activations = sample(activation_vals, depth, replace = TRUE)
            row_data$activations = list(as.character(activations))
        }
        
        if (length(scalar_params) > 0) {
            if (n_numeric_scalars > 0) {
                scalar_indices = row[(n_numeric_arch + 1):n_dims]
                scalars = decode_scalars(scalar_params, scalar_indices, original)
            } else {
                scalars = decode_scalars(scalar_params, numeric(0), original)
            }
            row_data = c(row_data, as.list(scalars))
        }
        
        results[[i]] = tibble::as_tibble(row_data)
    }
    
    dplyr::bind_rows(results)
}

generate_sfd_grid = 
    function(
        neuron_param, activation_param, n_hlayer,
        scalar_params, size, sfd_type, variogram_range, iter, original
    ) {
    if (!requireNamespace("sfd", quietly = TRUE)) {
        cli::cli_abort("Package {.pkg sfd} required for space-filling designs.")
    }
    
    has_neurons = !is.null(neuron_param)
    activation_vals = extract_param_values(activation_param)
    
    max_depth = max(n_hlayer)
    n_numeric_arch = if (has_neurons) max_depth else 0
    n_numeric_scalars = count_numeric_params(scalar_params)
    n_dims = n_numeric_arch + n_numeric_scalars
    
    if (n_dims == 0) {
        return(generate_random_grid(
            neuron_param, activation_param, n_hlayer,
            scalar_params, size, original
        ))
    }
    
    has_premade = sfd::sfd_available(n_dims, size, sfd_type)
    
    if (has_premade) {
        design = sfd::get_design(n_dims, num_points = size, type = sfd_type)
        design = apply(design, 2, function(col) {
            col_range = max(col) - min(col)
            if (col_range > 0) {
                (col - min(col)) / col_range
            } else {
                rep(0.5, length(col))
            }
        })
    } else {
        if (!requireNamespace("DiceDesign", quietly = TRUE)) {
            cli::cli_abort("Package {.pkg DiceDesign} required when pre-made designs are not available.")
        }
        design = DiceDesign::dmaxDesign(
            size,
            n_dims,
            range = 1,
            niter_max = iter
        )$design
    }
    
    depths = safe_sample(n_hlayer, size, replace = TRUE)
    
    results = vector("list", size)
    
    for (i in seq_len(size)) {
        depth = depths[i]
        row = design[i, ]
        row_data = list()
        
        if (has_neurons) {
            neuron_indices = row[seq_len(n_numeric_arch)]
            neurons = decode_neurons_from_design(
                neuron_param, 
                neuron_indices[seq_len(depth)],
                original
            )
            row_data$hidden_neurons = list(as.integer(neurons))
        }
        
        if (!is.null(activation_vals)) {
            activations = sample(activation_vals, depth, replace = TRUE)
            row_data$activations = list(as.character(activations))
        }
        
        if (length(scalar_params) > 0) {
            if (n_numeric_scalars > 0) {
                scalar_indices = row[(n_numeric_arch + 1):n_dims]
                scalars = decode_scalars(scalar_params, scalar_indices, original)
            } else {
                scalars = decode_scalars(scalar_params, numeric(0), original)
            }
            row_data = c(row_data, as.list(scalars))
        }
        
        results[[i]] = tibble::as_tibble(row_data)
    }
    
    dplyr::bind_rows(results)
}

extract_param_range = function(param, levels, original = TRUE) {
    if (is.null(param)) return(NULL)
    
    if (!is.null(param$values)) {
        return(param$values)
    }
    
    if (param$type %in% c("integer", "double")) {
        lower = param$range$lower
        upper = param$range$upper
        
        has_trans = !is.null(param$trans)
        
        if (!is.null(levels)) {
            if (has_trans) {
                vals_trans = seq(lower, upper, length.out = levels)
                vals = param$trans$inverse(vals_trans)
            } else {
                vals = seq(lower, upper, length.out = levels)
            }
            
            if (param$type == "integer") {
                unique(as.integer(round(vals)))
            } else {
                vals
            }
        } else {
            if (param$type == "integer") {
                seq.int(lower, upper)
            } else {
                c(lower, upper)
            }
        }
    } else {
        NULL
    }
}

extract_param_bounds = function(param) {
    if (is.null(param)) return(NULL)
    
    if (param$type %in% c("integer", "double")) {
        c(param$range$lower, param$range$upper)
    } else {
        NULL
    }
}

extract_param_values = function(param) {
    if (is.null(param)) return(NULL)
    
    if (param$type == "character" || param$type == "logical") {
        param$values
    } else {
        NULL
    }
}

sample_from_param = function(param, n, original = TRUE) {
    if (is.null(param)) return(NULL)
    if (!is.null(param$values)) {
        return(sample(param$values, n, replace = TRUE))
    }
    
    lower = param$range$lower
    upper = param$range$upper
    
    if (!is.null(param$trans)) {
        vals_trans = stats::runif(n, lower, upper)
        vals = param$trans$inverse(vals_trans)
    } else {
        vals = stats::runif(n, lower, upper)
    }
    
    if (param$type == "integer") {
        as.integer(round(vals))
    } else {
        vals
    }
}

decode_neurons_from_design = function(param, design_vals, original = TRUE) {
    lower = param$range$lower
    upper = param$range$upper
    
    if (!is.null(param$values)) {
        values = param$values
        indices = pmax(
            1L, 
            pmin(length(values), ceiling(design_vals * length(values)))
        )
        return(values[indices])
    }
    # discrete_vals = attr(param, "discrete_values")
    # if (!is.null(discrete_vals)) {
    #     indices = pmax(
    #         1, 
    #         pmin(length(discrete_vals), ceiling(design_vals * length(discrete_vals)))
    #     )
    #     return(discrete_vals[indices])
    # }
    
    if (!is.null(param$trans)) {
        vals_trans = lower + design_vals * (upper - lower)
        vals = param$trans$inverse(vals_trans)
    } else {
        vals = lower + design_vals * (upper - lower)
    }
    
    if (param$type == "integer") {
        as.integer(round(vals))
    } else {
        vals
    }
}

expand_architecture = function(neuron_vals, activation_vals, depth) {
    neuron_cols = stats::setNames(
        rep(list(neuron_vals), depth),
        paste0("n", seq_len(depth))
    )
    activation_cols = stats::setNames(
        rep(list(activation_vals), depth),
        paste0("a", seq_len(depth))
    )
    
    # Generate all combinations
    grid = tidyr::expand_grid(!!!neuron_cols, !!!activation_cols)
    
    # Convert to list columns
    neuron_col_names = paste0("n", seq_len(depth))
    activation_col_names = paste0("a", seq_len(depth))
    
    tibble::tibble(
        hidden_neurons = purrr::pmap(
            dplyr::select(grid, dplyr::all_of(neuron_col_names)),
            ~ as.integer(c(...))
        ),
        activations = purrr::pmap(
            dplyr::select(grid, dplyr::all_of(activation_col_names)),
            ~ as.character(c(...))
        )
    )
}

expand_neurons_only = function(neuron_vals, depth) {
    neuron_cols = stats::setNames(
        rep(list(neuron_vals), depth),
        paste0("n", seq_len(depth))
    )
    
    grid = tidyr::expand_grid(!!!neuron_cols)
    
    tibble::tibble(
        hidden_neurons = purrr::pmap(grid, ~ as.integer(c(...)))
    )
}

expand_activations_only = function(activation_vals, depth) {
    activation_cols = stats::setNames(
        rep(list(activation_vals), depth),
        paste0("a", seq_len(depth))
    )
    
    grid = tidyr::expand_grid(!!!activation_cols)
    
    tibble::tibble(
        activations = purrr::pmap(grid, ~ as.character(c(...)))
    )
}

make_scalar_grid = function(scalar_params, size, levels, type, original) {
    if (length(scalar_params) == 0) {
        return(tibble::tibble())
    }
    
    params_obj = dials::parameters(scalar_params)
    
    if (type == "regular") {
        dials::grid_regular(params_obj, levels = levels, original = original)
    } else {
        dials::grid_random(params_obj, size = size, original = original)
    }
}

count_numeric_params = function(scalar_params) {
    sum(purrr::map_lgl(scalar_params, ~ .x$type %in% c("double", "integer")))
}

decode_scalars = function(scalar_params, design_vals, original = TRUE) {
    if (length(scalar_params) == 0) {
        return(tibble::tibble())
    }
    
    numeric_params = purrr::keep(scalar_params, ~ .x$type %in% c("double", "integer"))
    categorical_params = purrr::keep(scalar_params, ~ .x$type %in% c("character", "logical"))
    
    decoded_numeric = if (length(numeric_params) > 0 && length(design_vals) > 0) {
        purrr::imap_dfc(numeric_params, function(param, param_name) {
            idx = which(names(numeric_params) == param_name)
            
            lower = param$range$lower
            upper = param$range$upper
            
            if (!is.null(param$trans)) {
                val_trans = lower + design_vals[idx] * (upper - lower)
                val = param$trans$inverse(val_trans)
            } else {
                val = lower + design_vals[idx] * (upper - lower)
            }
            
            if (param$type == "integer") {
                val = as.integer(round(val))
            }
            
            tibble::tibble(!!param_name := val)
        })
    } else {
        NULL
    }
    
    decoded_categorical = if (length(categorical_params) > 0) {
        purrr::imap_dfc(categorical_params, function(param, param_name) {
            tibble::tibble(!!param_name := sample(param$values, 1))
        })
    } else {
        NULL
    }
    
    if (!is.null(decoded_numeric) && !is.null(decoded_categorical)) {
        dplyr::bind_cols(decoded_numeric, decoded_categorical)
    } else if (!is.null(decoded_numeric)) {
        decoded_numeric
    } else if (!is.null(decoded_categorical)) {
        decoded_categorical
    } else {
        tibble::tibble()
    }
}

`%||%` = function(x, y) if (is.null(x)) y else x

#' Safe sampling function
#' 
#' R's sample() has quirky behavior: sample(5, 1) samples from 1:5, not from c(5).
#' This function ensures we sample from the actual vector provided.
#'
#' @param x Vector to sample from
#' @param size Number of samples
#' @param replace Sample with replacement?
#' @keywords internal
safe_sample = function(x, size, replace = FALSE) {
    if (length(x) == 1) {
        rep(x, size)
    } else {
        sample(x, size, replace = replace)
    }
}
