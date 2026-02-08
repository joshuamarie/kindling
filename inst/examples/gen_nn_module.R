box::use(
    kindling[nn_module_generator, act_funs, args]
)

nn_module_generator(
    nn_name = "CNN1DClassifier",
    nn_layer = "nn_conv1d",
    layer_arg_fn = ~ if (.is_output) {
        list(.in, .out)
    } else {
        list(
            in_channels = .in,
            out_channels = .out,
            kernel_size = 3L,
            stride = 1L,
            padding = 1L 
        )
    },
    after_output_transform = ~ .$mean(dim = 2),
    last_layer_args = list(kernel_size = 1, stride = 2),
    hd_neurons = c(16, 32, 64),
    no_x = 1,
    no_y = 10,
    activations = "relu"
)

nn_module_generator(
    nn_name = "CNN1DClassifier",
    nn_layer = "nn_conv1d",
    layer_arg_fn = ~ if (.is_output) {
        list(.in, .out, kernel_size = 1, stride = 2)
    } else {
        list(
            in_channels = .in,
            out_channels = .out,
            kernel_size = 3L,
            stride = 1L,
            padding = 1L 
        )
    },
    after_output_transform = ~ .$mean(dim = 2),
    # last_layer_args = list(kernel_size = 1, stride = 2),
    hd_neurons = c(16, 32, 64),
    no_x = 1,
    no_y = 10,
    activations = "relu"
)
