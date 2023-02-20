import jax.numpy as jnp
import functorch
import torch
import jax
from flax import linen as jnn
from torch import nn as pnn
from torch.utils.dlpack import from_dlpack as pyt_from_dlpack
from torch.utils.dlpack import to_dlpack as pyt_to_dlpack
from jax.dlpack import to_dlpack as jax_to_dlpack
from jax.dlpack import from_dlpack as jax_from_dlpack
from jax import custom_vjp
from typing import Any, Callable, Iterable, Optional, Tuple, Union


#Convert a Pytorch model to a Flax model
# Define functions for converting data between PyTorch and JAX representations
def convert_to_pyt(x):
    # If x is a JAX ndarray, convert it to a DLPack and then to a PyTorch tensor
    if isinstance(x,jnp.ndarray):
        x=jax_to_dlpack(x)
        x=pyt_from_dlpack(x)
    return x

def convert_to_jax(x):
    # If x is a PyTorch tensor, convert it to a DLPack and then to a JAX ndarray
    if isinstance(x,torch.Tensor):
        x=pyt_to_dlpack(x)
        x=jax_from_dlpack(x)
    return x


def convert_pytnn_to_jax(model):
    # Convert the PyTorch model to a functional representation and extract the model function and parameters
    model_fn,model_params=functorch.make_functional(model)



    # Convert the model parameters from PyTorch to JAX representations
    model_params=jax.tree_map(convert_to_jax,model_params)

    # Define the apply function using a custom VJP
    @custom_vjp
    def apply(params,*args,**kwargs):
        # Convert the input data from PyTorch to JAX representations
        params=jax.tree_map(convert_to_pyt,params)
        args=jax.tree_map(convert_to_pyt,args)
        kwargs=jax.tree_map(convert_to_pyt,kwargs)
        # Apply the model function to the input data
        out=model_fn(params,*args,**kwargs)
        # Convert the output data from JAX to PyTorch representations
        out=jax.tree_map(convert_to_jax,out)
        return out

    # Define the forward and backward passes for the VJP
    def apply_fwd(params,*args,**kwargs):
        return apply(params,*args,**kwargs),(params,args,kwargs)
    
    def apply_bwd(res,grads):
        params,args,kwargs=res
        # Convert the input data and gradients from PyTorch to JAX representations
        params=jax.tree_map(convert_to_pyt,params)
        args=jax.tree_map(convert_to_pyt,args)
        kwargs=jax.tree_map(convert_to_pyt,kwargs)
        grads=jax.tree_map(convert_to_pyt,grads)
        # Compute the gradients using the model function and convert them from JAX to PyTorch representations
        grads=functorch.vjp(model_fn,params,*args,**kwargs)[1](grads)
        grads=jax.tree_map(convert_to_jax,grads)
        return grads
    apply.defvjp(apply_fwd,apply_bwd)
    
    # Return the apply function and the converted model parameters
    return apply,model_params


def convert_pytnn_to_flax(model):
    # Define a Flax module that wraps the JAX-converted PyTorch model
    jax_fn,params=convert_pytnn_to_jax(model)
    class FlaxModule(jnn.Module):
        # Convert the PyTorch model to a JAX-converted version and set it up as a Flax parameter
        def setup(self):
            self.jax_param=self.param('jax_params',lambda x:params)
        
        # Define the __call__ method to apply the JAX-converted model to the input data
        def __call__(self,x):
            return jax_fn(self.jax_param,x)
    return FlaxModule,params


def py_to_jax_wrapper(fun):
    def wrapper(*args,**kwargs):
        # Convert the input data from PyTorch to JAX representations
        args=jax.tree_map(convert_to_pyt,args)
        kwargs=jax.tree_map(convert_to_pyt,kwargs)
        # Apply the function to the input data
        out=fun(*args,**kwargs)
        # Convert the output data from JAX to PyTorch representations
        out=jax.tree_map(convert_to_jax,out)
        return out
    return wrapper