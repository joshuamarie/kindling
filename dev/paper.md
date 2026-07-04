# Summary

[kindling](https://kindling.joshuamarie.com) is an R \[@rcoreteam2025\]
package that provides a higher-level interface to the
[torch](https://torch.mlverse.org/docs) package
\[@falbelluraschi2025torch\], R’s native implementation of PyTorch, for
defining, training, and tuning neural networks. This package supports
MLPs with the same topology, including the standard deep feedforward
neural networks and recurrent variants (RNN, LSTM, GRU), while reducing
the boilerplate typically required to write
[torch](https://torch.mlverse.org/docs) neural network architecture
expression and training loops by hand.

The package is organized around three levels of abstraction. At the
lowest level, `_generator` functions (for example
[`ffnn_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_gens.md))
return unevaluated
[`torch::nn_module()`](https://torch.mlverse.org/docs/reference/nn_module.html)
expressions that users can inspect or modify directly, since
[kindling](https://kindling.joshuamarie.com) builds its models through
code generation rather than opaque wrapper objects. At an intermediate
level, functions such as
[`ffnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md)
and
[`rnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md)
train a model directly from a formula and data frame, handling data
preparation, the optimization loop, and, optionally, early stopping and
a validation split. At the highest level,
[`mlp_kindling()`](https://kindling.joshuamarie.com/dev/reference/mlp_kindling.md)
and
[`rnn_kindling()`](https://kindling.joshuamarie.com/dev/reference/rnn_kindling.md)
register these models as
[parsnip](https://github.com/tidymodels/parsnip) model specifications
\[@kuhnvaughan2026parsnip\], so they can be fit, tuned, and evaluated
using the rest of the [tidymodels](https://tidymodels.tidymodels.org)
ecosystem \[@kuhnwickham2020tidymodels\]:
[recipes](https://github.com/tidymodels/recipes) for preprocessing,
[workflows](https://github.com/tidymodels/workflows) for bundling
preprocessing and modeling steps, and
[tune](https://tune.tidymodels.org/)/[dials](https://dials.tidymodels.org)
for hyperparameter search over layer widths, network depth, activation
functions, the output activation, the optimizer, and other architectural
choices. Fitted models can also be inspected with variable-importance
methods from `{NeuralNetTools}` \[@beck2018neuralnettools\],
implementing the algorithms of Garson \[@garson1991\] and Olden and
Jackson \[@oldenjackson2002\], and with the
[vip](https://github.com/koalaverse/vip/) package. shows an example of
this last capability applied to the feedforward network trained in the
package’s own README usage example.

![Garson’s algorithm variable-importance scores \[@garson1991\],
computed with {kindling}’s garson() wrapper around {NeuralNetTools}
\[@beck2018neuralnettools\] for a feedforward network trained on the
iris dataset (the same model used in the package’s README “Direct
Training Interface” example). Generated with: model = ffnn(Species ~ .,
data = iris, hidden_neurons = c(10, 15, 7), activations = act_funs(relu,
softshrink\[lambd = 0.5\], elu), loss = "cross_entropy", epochs = 100);
garson(model, bar_plot = TRUE).](paper-figure.png)

Garson’s algorithm variable-importance scores \[@garson1991\], computed
with [kindling](https://kindling.joshuamarie.com)’s
[`garson()`](https://rdrr.io/pkg/NeuralNetTools/man/garson.html) wrapper
around `{NeuralNetTools}` \[@beck2018neuralnettools\] for a feedforward
network trained on the `iris` dataset (the same model used in the
package’s README “Direct Training Interface” example). Generated with:
`model = ffnn(Species ~ ., data = iris, hidden_neurons = c(10, 15, 7), activations = act_funs(relu, softshrink[lambd = 0.5], elu), loss = "cross_entropy", epochs = 100); garson(model, bar_plot = TRUE)`.

# Statement of need

[torch](https://torch.mlverse.org/docs) gives R users direct access to
tensors, automatic differentiation, and GPU acceleration, but writing a
[`torch::nn_module()`](https://torch.mlverse.org/docs/reference/nn_module.html),
a training loop, and the surrounding data-handling code by hand is
repetitive and error-prone, particularly for users who want to compare
several architectures or run a hyperparameter search rather than fit one
fixed model. At the same time, R’s dominant applied machine learning
framework, [tidymodels](https://tidymodels.tidymodels.org), historically
has had comparatively narrow neural network support, and most existing
higher-level [torch](https://torch.mlverse.org/docs) wrappers do not
expose the generated model code or integrate with the tuning and
resampling infrastructure that
[tidymodels](https://tidymodels.tidymodels.org) users already rely on
for other model types.

[kindling](https://kindling.joshuamarie.com) addresses this gap for R
users who want deep learning to sit inside a
[tidymodels](https://tidymodels.tidymodels.org) workflow rather than
beside it. Because model specifications built with
[`mlp_kindling()`](https://kindling.joshuamarie.com/dev/reference/mlp_kindling.md)
or
[`rnn_kindling()`](https://kindling.joshuamarie.com/dev/reference/rnn_kindling.md)
behave like any other [parsnip](https://github.com/tidymodels/parsnip)
model, an analyst who already tunes a random forest or a boosted tree
with
[`tune::tune_grid()`](https://tune.tidymodels.org/reference/tune_grid.html)
and [rsample](https://rsample.tidymodels.org) resamples can point the
same workflow at a neural network with only the model specification
changed. The code generation layer additionally lets users audit or
extend the underlying [torch](https://torch.mlverse.org/docs) module
instead of treating it as a black box, which is useful both for teaching
and for architectures that need small custom modifications.

# State of the field

Several R packages sit above [torch](https://torch.mlverse.org/docs) to
reduce this boilerplate, and each targets a different point on the
tradeoff between flexibility and convenience.
[brulee](https://github.com/tidymodels/brulee) \[@kuhnfalbel2025brulee\]
is the official [tidymodels](https://tidymodels.tidymodels.org) package
for [torch](https://torch.mlverse.org/docs)-based models; it offers
production-oriented, batteries-included implementations of linear,
logistic, and multinomial regression and a multi-layer perceptron whose
depth and per-layer activations (from a fixed built-in list) can be set
manually but are not exposed as
[dials](https://dials.tidymodels.org)-tunable search dimensions the way
[kindling](https://kindling.joshuamarie.com)’s
[`n_hlayers()`](https://kindling.joshuamarie.com/dev/reference/dials-kindling.md)/[`grid_depth()`](https://kindling.joshuamarie.com/dev/reference/grid_depth.md)
and
[`activations()`](https://kindling.joshuamarie.com/dev/reference/dials-kindling.md)
are; it also does not support recurrent or convolutional architectures,
custom activation functions, or inspectable generated code.
[cito](https://citoverse.github.io/cito/) \[@amesoder2024cito\]
emphasizes statistical inference and explainability for fully-connected
networks and convolutional networks through a formula interface, with an
extensive set of interpretation tools (partial dependence, accumulated
local effects, bootstrap confidence intervals), but it is a standalone
package rather than a [tidymodels](https://tidymodels.tidymodels.org)
engine. [luz](https://mlverse.github.io/luz/) provides a general,
high-level training-loop abstraction for arbitrary
[`torch::nn_module()`](https://torch.mlverse.org/docs/reference/nn_module.html)
objects \[@falbel2025luz\]; it is architecture-agnostic and reduces
boilerplate at the loop level, but does not offer a formula interface,
code generation, or [tidymodels](https://tidymodels.tidymodels.org)
integration.

[kindling](https://kindling.joshuamarie.com) is positioned alongside,
not in place of, these packages: it combines inspectable code
generation, both feedforward and recurrent architecture families, and
full [tidymodels](https://tidymodels.tidymodels.org) integration
([parsnip](https://github.com/tidymodels/parsnip) specifications,
[tune](https://tune.tidymodels.org/)/[dials](https://dials.tidymodels.org)
search spaces, and
[recipes](https://github.com/tidymodels/recipes)/[workflows](https://github.com/tidymodels/workflows)
pipelines) in one package. Its code generation layer is also more
versatile than a feedforward/recurrent split suggests: the generalized
[`nn_module_generator()`](https://kindling.joshuamarie.com/dev/reference/nn_module_generator.md)/[`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md)
interface (currently experimental) can generate and train arbitrary
sequential [torch](https://torch.mlverse.org/docs) architectures, for
example one-dimensional convolutional networks, beyond the fixed model
families exposed by [brulee](https://github.com/tidymodels/brulee) and
[cito](https://citoverse.github.io/cito/).
[mlr3](https://mlr3.mlr-org.com) \[@lang2019mlr3\] is the other major R
modeling framework and is a planned integration target for
[kindling](https://kindling.joshuamarie.com), alongside its current
[tidymodels](https://tidymodels.tidymodels.org) support. In practice,
the choice between these packages depends on the task:
[cito](https://citoverse.github.io/cito/) for statistical inference and
explainability, [luz](https://mlverse.github.io/luz/) for custom
training loops, and [brulee](https://github.com/tidymodels/brulee) or
[kindling](https://kindling.joshuamarie.com) for production-oriented
[tidymodels](https://tidymodels.tidymodels.org) workflows, where
[kindling](https://kindling.joshuamarie.com)’s own full
[tidymodels](https://tidymodels.tidymodels.org) integration makes it
just as production-ready as
[brulee](https://github.com/tidymodels/brulee) while additionally
offering architectural flexibility, broader architecture support, and
inspectable generated code.

# Software design

[kindling](https://kindling.joshuamarie.com)’s central design decision
is to separate code generation from execution. The `_generator`
functions build quoted
[`torch::nn_module()`](https://torch.mlverse.org/docs/reference/nn_module.html)
expressions through R’s metaprogramming facilities rather than
instantiating an opaque model object internally; the same expression can
be printed, copied, hand-edited, and run outside
[kindling](https://kindling.joshuamarie.com). This costs an extra layer
of implementation complexity (an expression-building step that a direct
wrapper would not need), not a runtime cost, since the generated code
executes as ordinary [torch](https://torch.mlverse.org/docs) once built;
in exchange, a user, reviewer, or student can verify exactly what
network is being fit rather than trust a black box, and can extend a
generated architecture for a case the higher-level wrappers do not
cover.

The three-level API (code generation, direct training,
[tidymodels](https://tidymodels.tidymodels.org) engine) follows from
that same trade-off, applied at increasing levels of convenience and
constraint. The generalized
[`train_nn()`](https://kindling.joshuamarie.com/dev/reference/gen-nn-train.md)
interface dispatches via S3 methods on the class of its input (`matrix`,
`data.frame`, `formula`, or [torch](https://torch.mlverse.org/docs)
`dataset`);
[`ffnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md)
and
[`rnn()`](https://kindling.joshuamarie.com/dev/reference/kindling-basemodels.md)
are separate, architecture-specific entry points that accept a formula
or `x`/`y` arguments directly rather than going through that dispatch (a
possible target for consolidation once the package has matured further
post-review). All three delegate preprocessing to
[`hardhat::mold()`](https://hardhat.tidymodels.org/reference/mold.html)/`forge()`
rather than reimplementing it, so
[kindling](https://kindling.joshuamarie.com) models inherit the same
predictor-role, factor-encoding, and missing-data handling as other
[parsnip](https://github.com/tidymodels/parsnip)/[recipes](https://github.com/tidymodels/recipes)
engines. Tunable parameters
([`hidden_neurons()`](https://kindling.joshuamarie.com/dev/reference/dials-kindling.md),
[`activations()`](https://kindling.joshuamarie.com/dev/reference/dials-kindling.md),
[`output_activation()`](https://kindling.joshuamarie.com/dev/reference/dials-kindling.md),
[`grid_depth()`](https://kindling.joshuamarie.com/dev/reference/grid_depth.md))
are implemented as [dials](https://dials.tidymodels.org) parameter
objects dispatched over `default`/`list`/`model_spec`/`param` classes,
so they plug directly into
[`tune::tune_grid()`](https://tune.tidymodels.org/reference/tune_grid.html)’s
search-space machinery instead of a bespoke tuning loop. This constrains
[kindling](https://kindling.joshuamarie.com) to
[tidymodels](https://tidymodels.tidymodels.org) conventions, but it is
precisely what lets a [tidymodels](https://tidymodels.tidymodels.org)
user reuse existing resampling and tuning code for a neural network
instead of learning a separate API.

Within the generated code itself, activation functions are specified
through
[`act_funs()`](https://kindling.joshuamarie.com/dev/reference/act_funs.md),
a small embedded domain specific language (eDSL) function (akin to
[dplyr](https://dplyr.tidyverse.org)’s `pick()` function) based on R’s
non-standard evaluation (NSE), built with
[rlang](https://rlang.r-lib.org), that accepts bracket syntax such as
`softshrink[lambd = 0.5]` for parametric activations, and
[`new_act_fn()`](https://kindling.joshuamarie.com/dev/reference/new_act_fn.md)
as an escape hatch for activations with no `torch::nnf_*()` equivalent.
This trades some NSE complexity in the implementation for a call-like
syntax at the user level, avoiding nested lists of function names and
parameter values.

# Research impact statement

[kindling](https://kindling.joshuamarie.com) is a young package
(development began with its first commit around early November 2025) and
does not yet have external citations, publications, or third-party
integrations to report; the evidence below documents community-readiness
rather than realized adoption, as of this writing (July 2026). It is
available on CRAN (initial release 0.1.0 on 2026-01-31; current release
0.3.1), with 1,434 downloads recorded since that release and 278
downloads in the last month, and the GitHub repository has 27 stars and
5 forks. Correctness is backed by a test suite of 206
[testthat](https://testthat.r-lib.org) test blocks covering the code
generators, the direct-training wrappers, the
[tidymodels](https://tidymodels.tidymodels.org) engines, and the
tuning/[dials](https://dials.tidymodels.org) integration, run
continuously via GitHub Actions `R-CMD-check` and tracked with Codecov.

# AI usage disclosure

The authors used Claude (Anthropic) to obtain suggestions for English
phrasing and to improve clarity and readability during the writing
process. All content was critically reviewed and finalized by the
authors.

# Acknowledgements

We thank the maintainers of [torch](https://torch.mlverse.org/docs) for
R and of the [tidymodels](https://tidymodels.tidymodels.org) ecosystem,
on which [kindling](https://kindling.joshuamarie.com) is built.

# References
