# Safe sampling function

R's sample() has quirky behavior: sample(5, 1) samples from 1:5, not
from c(5). This function ensures we sample from the actual vector
provided.

## Usage

``` r
safe_sample(x, size, replace = FALSE)
```

## Arguments

- x:

  Vector to sample from

- size:

  Number of samples

- replace:

  Sample with replacement?
