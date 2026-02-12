box::use(
    kindling[
        mlp_kindling, hidden_neurons,
        activations, output_activation,
        grid_depth
    ],
    parsnip[mlp, fit, augment, set_engine, set_mode],
    recipes[
        recipe, step_YeoJohnson,
        step_normalize, all_numeric_predictors
    ],
    workflows[workflow, add_recipe, add_model],
    rsample[
        initial_split, training, testing,
        vfold_cv
    ],
    tune[
        tune_grid, tune,
        select_best, finalize_workflow
    ],
    dials[
        grid_random, penalty, hidden_units
    ],
    yardstick[
        metric_set, accuracy, roc_auc
    ],
    dplyr[
        select, bind_rows, mutate
    ]
)

data(Ionosphere, package = "mlbench")

set.seed(123)

# =========================================================
# 1. TRAIN / TEST SPLIT
# =========================================================

split =
    initial_split(
        Ionosphere,
        prop = 0.8,
        strata = Class
    )

train = training(split)
test = testing(split)

folds =
    vfold_cv(
        train,
        v = 5,
        strata = Class
    )

# =========================================================
# 2. RECIPE
# =========================================================

data_rec =
    train |>
    select(!(1:2)) |>
    recipe(Class ~ .) |>
    step_YeoJohnson(all_numeric_predictors()) |>
    step_normalize(all_numeric_predictors())

# =========================================================
# 3A. KINDLING MODEL
# =========================================================

kindling_spec =
    mlp_kindling(
        mode = "classification",
        hidden_neurons = tune(),
        activations = tune(),
        output_activation = tune()
    )

kindling_wf =
    workflow() |>
    add_recipe(data_rec) |>
    add_model(kindling_spec)

kindling_grid =
    grid_depth(
        hidden_neurons(c(16L, 64L)),
        activations(c("relu", "elu", "softshrink(lambd = 0.2)")),
        output_activation(c("sigmoid", "relu")),
        n_hlayer = 2,
        size = 10,
        type = "latin_hypercube"
    )

kindling_tune =
    tune_grid(
        kindling_wf,
        resamples = folds,
        grid = kindling_grid,
        metrics = metric_set(accuracy, roc_auc)
    )

best_kindling =
    select_best(
        kindling_tune,
        metric = "roc_auc"
    )

final_kindling_wf =
    finalize_workflow(
        kindling_wf,
        best_kindling
    )

final_kindling_model =
    fit(
        final_kindling_wf,
        data = train
    )
final_kindling_model

kindling_train_metrics =
    final_kindling_model |>
    augment(new_data = train) |>
    metric_set(accuracy, roc_auc)(
        truth = Class,
        estimate = .pred_class,
        .pred_good
    )

kindling_test_metrics =
    final_kindling_model |>
    augment(new_data = test) |>
    metric_set(accuracy, roc_auc)(
        truth = Class,
        estimate = .pred_class,
        .pred_good
    )

# =========================================================
# 3B. BRULEE VIA PARSNIP MLP
# =========================================================

brulee_spec =
    mlp(
        hidden_units = tune(),
        penalty = tune(),
        epochs = 100
    ) |>
    set_mode("classification") |>
    set_engine("brulee")

brulee_wf =
    workflow() |>
    add_recipe(data_rec) |>
    add_model(brulee_spec)

brulee_grid =
    grid_random(
        hidden_units(c(32L, 128L)),
        penalty(),
        size = 10
    )

brulee_tune =
    tune_grid(
        brulee_wf,
        resamples = folds,
        grid = brulee_grid,
        metrics = metric_set(accuracy, roc_auc)
    )

best_brulee =
    select_best(
        brulee_tune,
        metric = "roc_auc"
    )

final_brulee_wf =
    finalize_workflow(
        brulee_wf,
        best_brulee
    )

final_brulee_model =
    fit(
        final_brulee_wf,
        data = train
    )
final_brulee_model

brulee_train_metrics =
    final_brulee_model |>
    augment(new_data = train) |>
    metric_set(accuracy, roc_auc)(
        truth = Class,
        estimate = .pred_class,
        .pred_good
    )

brulee_test_metrics =
    final_brulee_model |>
    augment(new_data = test) |>
    metric_set(accuracy, roc_auc)(
        truth = Class,
        estimate = .pred_class,
        .pred_good
    )

# =========================================================
# 4. FINAL COMPARISON
# =========================================================

train_comparison = 
    bind_rows(
        kindling_train_metrics |>
            mutate(model = "kindling"),
        brulee_train_metrics |>
            mutate(model = "brulee")
    )

test_comparison =
    bind_rows(
        kindling_test_metrics |>
            mutate(model = "kindling"),
        brulee_test_metrics |>
            mutate(model = "brulee")
    )

train_comparison
test_comparison
