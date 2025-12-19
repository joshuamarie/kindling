#' Validate device and get default device
#'
#' Check if requested device is available. And auto-detect available GPU device
#' or fallback to CPU.
#'
#' @param device Character. Requested device.
#' @return Character string of validated device.
#' @keywords internal
validate_device = function(device) {
    device = tolower(device)

    if (device == "cuda" && !torch::cuda_is_available()) {
        cli::cli_warn("CUDA not available. Falling back to CPU.")
        return("cpu")
    }

    if (device == "mps" && !torch::backends_mps_is_available()) {
        cli::cli_warn("MPS not available. Falling back to CPU.")
        return("cpu")
    }

    if (!device %in% c("cpu", "cuda", "mps")) {
        cli::cli_abort("Invalid device: {device}. Must be 'cpu', 'cuda', or 'mps'.")
    }

    device
}

get_default_device = function() {
    if (torch::cuda_is_available()) {
        return("cuda")
    } else if (torch::backends_mps_is_available()) {
        return("mps")
    } else {
        return("cpu")
    }
}
