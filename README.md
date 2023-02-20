# Pytorch2Jax

[![PyPI version](https://badge.fury.io/py/pytorch2jax.svg)](https://badge.fury.io/py/pytorch2jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pytorch2Jax is a small Python library that provides functions to wrap PyTorch models into Jax functions and Flax modules. It uses `dlpack` to convert between Pytorch and Jax tensors in-memory and executes Pytorch backend inside Jax wrapped functions. The wrapped functions are compaitible Jax backward-mode autodiff via `functorch.vjp`.

## Installation

You can install the Pytorch2Jax package from PyPI via pip:
```
pip install pytorch2jax
```

## Usage
### Example 1: Wrap a Pytorch function to a function that accepts Jax tensors

```python
import torch
import jax.numpy as jnp
from pytorch2jax import py_to_jax_wrapper

# Define a PyTorch function that multiples an input tensor with another tensor
# and wrap it with the py_to_jax_wrapper decorator
@py_to_jax_wrapper
def fn(x):
    return torch.rand((10,10))*x


# Call the wrapped function on a JAX array
x = jnp.ones((10,10))
output = fn(x)

# Print the output
print(output)

```

### Example 2: Convert a PyTorch model to a JAX function and differentiate with grad

The converted Jax function can be used seamlessly with Jax's `grad` function to compute gradients.
```python
import jax.numpy as jnp
import jax

import torch.nn as pnn

from pytorch2jax import convert_pytnn_to_jax

# Create a PyTorch model
pyt_model = pnn.Linear(10, 10)

# Convert PyTorch model to a JAX function
jax_fn, params = convert_pytnn_to_jax(pyt_model)

# Define a function that uses the JAX function and returns the sum of its output
def fx(x):
    return jax_fn(params, x).sum()

# Compute the gradient of the function `fx` with respect to `x`
grad_fx = jax.grad(fx)
x = jnp.ones((10,))
print(grad_fx(x))  # Prints the gradient of fx at x

```

### Example 3: Convert a PyTorch model to a Flax model class and do forward pass inside another Flax module

```python
import jax.numpy as jnp
import jax
import torch.nn as pnn
import flax.linen as jnn

from pytorch2jax import convert_pytnn_to_flax
from typing import Any

# Convert the PyTorch model to a Flax model using the 'convert_pytnn_to_flax' function
# flax_module is the converted Flax model and params are the parameters of the converted Flax model
pyt_model = pnn.Linear(10, 10)
flax_module, params = convert_pytnn_to_flax(pyt_model)

# Define a new Flax module and define the flax_module attribute as the converted Flax model
# The __call__ method of this module will call the __call__ method of the flax_module attribute
class SampleFlaxModule(jnn.Module):
    flax_module: Any

    @jnn.compact
    def __call__(self, x):
        return self.flax_module()(x)

# Create an instance of the new Flax module
flax_model = SampleFlaxModule(flax_module)

# Initialize the parameters of the Flax model using random key and a 10x10 array of ones as input
params = flax_model.init(jax.random.PRNGKey(0), jnp.ones((10, 10)))

# Apply the Flax model to the input to get the output
flax_model.apply(params, jnp.ones((10, 10)))
```

# Contributing

If you encounter any bugs or issues while using pytorch2jax, or if you have any suggestions for improvements or new features, please open an issue on the GitHub repository at https://github.com/subho406/Pytorch2Jax.

# License

Pytorch2Jax is released under the MIT License. See LICENSE for more information.
