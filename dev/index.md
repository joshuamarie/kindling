# kindling

## Package overview

Title: ***Higher-Level Interface of ‘torch’ Package to Auto-Train Neural
Networks***

Whether you’re generating neural network architecture expressions or
directly fitting/training models,
[kindling](https://kindling.joshuamarie.com) minimizes boilerplate code
while preserving [torch](https://torch.mlverse.org/docs). Since this
package uses [torch](https://torch.mlverse.org/docs) as its backend, GPU
acceleration is supported.

[kindling](https://kindling.joshuamarie.com) also bridges the gap
between [torch](https://torch.mlverse.org/docs) and
[tidymodels](https://tidymodels.tidymodels.org). It works seamlessly
with [parsnip](https://github.com/tidymodels/parsnip),
[recipes](https://github.com/tidymodels/recipes), and
[workflows](https://github.com/tidymodels/workflows) to bring deep
learning into your existing
[tidymodels](https://tidymodels.tidymodels.org) modeling pipeline. This
enables a streamlined interface for building, training, and tuning deep
learning models within the familiar
[tidymodels](https://tidymodels.tidymodels.org) ecosystem.

### Main Features

- Code generation of [torch](https://torch.mlverse.org/docs) expression

- Multiple architectures available

  - Base models interface: feedforward networks (MLP/DNN/FFNN) and
    recurrent variants (RNN, LSTM, GRU)
  - Generalized neural network trainer that has the same topology as
    MLPs

- Native support for R ML workflows and pipelines (currently
  [tidymodels](https://tidymodels.tidymodels.org);
  [mlr3](https://mlr3.mlr-org.com) planned)

- Fine-grained control over network depth, layer sizes, and activation
  functions

- GPU acceleration support via [torch](https://torch.mlverse.org/docs)
  tensors

## Installation

You can install [kindling](https://kindling.joshuamarie.com) on CRAN:

``` r

install.packages('kindling')
```

Or install the development version from GitHub:

``` r

# install.packages("pak")
pak::pak("joshuamarie/kindling")
## devtools::install_github("joshuamarie/kindling")
```

## Learn more

- [Getting Started with
  kindling](https://kindling.joshuamarie.com/articles/kindling.html)
- [Tuning
  Capabilities](https://kindling.joshuamarie.com/articles/tuning-capabilities.html)
- [Custom Activation
  Function](https://kindling.joshuamarie.com/articles/custom-act-fn.html)
- [Special Cases: Linear and Logistic
  Regression](https://kindling.joshuamarie.com/articles/special-cases.html)
- [Similar Packages and
  Comparison](https://kindling.joshuamarie.com/articles/similar-packages.html)

## References

Falbel D, Luraschi J (2023). *torch: Tensors and Neural Networks with
‘GPU’ Acceleration*. R package version 0.13.0,
<https://torch.mlverse.org>, <https://github.com/mlverse/torch>.

Wickham H (2019). *Advanced R*, 2nd edition. Chapman and Hall/CRC. ISBN
978-0815384571, <https://adv-r.hadley.nz/>.

Goodfellow I, Bengio Y, Courville A (2016). *Deep Learning*. MIT Press.
<https://www.deeplearningbook.org/>.

## Citation

If you use [kindling](https://kindling.joshuamarie.com) in a
publication, please cite it. Run `citation("kindling")` in R to get the
current citation, or see the [CITATION
file](https://github.com/joshuamarie/kindling/blob/main/inst/CITATION).

## License

MIT + file LICENSE

## Code of Conduct

Please note that the kindling project is released with a [Contributor
Code of
Conduct](https://contributor-covenant.org/version/2/1/CODE_OF_CONDUCT.html).
By contributing to this project, you agree to abide by its terms.
