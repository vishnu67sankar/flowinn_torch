
"""
Author of the tensorflow repo is Jon Errasti Odriozola (tensorflow) | github-id: https://github.com/errasti13
Vishnu Sankar has converted tf to pytorch and re-built the same package but in pytroch
"""

import torch
from typing import Tuple, List
from abc import ABC, abstractmethod


class NavierStokes(ABC):
    """Base class for Navier-Stokes equations."""
    
    def _compute_first_derivatives(self, variables: list, coords: list) -> list:
        """Compute first-order derivatives for each variable with respect to each coordinate."""
        derivatives = []
      
        for var_item in variables:
            var = var_item.reshape(-1) # 1d tensor

            for coords_item in coords:
                if not var.requires_grad or var.grad_fn is None:
                    print(f"Warning: Variable '{var_item}' (reshaped to {var.shape}) ", 
                      f"does not require grad or has no grad_fn for differentiation w.r.t coord '{coords_item.shape}'.")
                
                grad = torch.autograd.grad(outputs=var, 
                inputs=coords_item, 
                grad_outputs=torch.ones_like(var),
                retain_graph=True, create_graph=True, allow_unused=True)[0]

                if grad is None:
                    print(f"CRITICAL WARNING (_compute_first_derivatives): grad is None for variable {i} w.r.t. coord {j}.")
                    print(f"    Var info: shape={current_var_for_grad.shape}, requires_grad={current_var_for_grad.requires_grad}, grad_fn_exists={current_var_for_grad.grad_fn is not None}")
                    print(f"    Coord info: shape={coord_tensor.shape}, requires_grad={coord_tensor.requires_grad}, is_leaf={coord_tensor.is_leaf}")
                    grad = torch.zeros_like(var)

                derivatives.append(grad.reshape(-1))
        return derivatives


    def _compute_second_derivatives(self, first_derivatives: list, coords: list) -> list:
        """Compute second-order derivatives."""
        derivatives = []

        if coords and coords[0] is not None:
            default_dtype = coords[0].dtype
            default_device = coords[0].device
            num_points_for_zeros = coords[0].shape[0]

        elif first_derivatives and first_derivatives[0] is not None:
            sample_d = torch.as_tensor(first_derivatives[0]) 
            default_dtype = sample_d.dtype
            default_device = sample_d.device
            num_points_for_zeros = sample_d.shape[0]
        
        else: 
            default_dtype = torch.float32
            default_device = torch.device('cpu')
            

        for i, d_tensor_orig in enumerate(first_derivatives): # d_tensor_orig is a first derivative
            if d_tensor_orig is not None:
                d_tensor = d_tensor_orig.reshape(-1)

                for j, coord_tensor in enumerate(coords):
                    grad_result = torch.autograd.grad(
                        outputs=d_tensor,
                        inputs=coord_tensor,
                        grad_outputs=torch.ones_like(d_tensor),
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True)
                    grad = grad_result[0]

                    if grad is None:
                        print(f"CRITICAL WARNING (_compute_second_derivatives): grad is None for 1st-deriv {i} w.r.t. coord {j}.")
                        print(f"    1st Deriv info: shape={d_tensor.shape}, requires_grad={d_tensor.requires_grad}, grad_fn_exists={d_tensor.grad_fn is not None}")
                        print(f"    Coord info: shape={coord_tensor.shape}, requires_grad={coord_tensor.requires_grad}, is_leaf={coord_tensor.is_leaf}")
                        grad = torch.zeros_like(d_tensor)
                    derivatives.append(grad.reshape(-1))
            else:
                print(f"INFO (_compute_second_derivatives): First derivative {i} was None, appending zeros for all its second derivatives.")
                derivatives.append(torch.zeros(num_points_for_zeros, dtype=default_dtype, device=default_device))
        return derivatives
    
    @abstractmethod
    def get_residuals(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]: # Changed tf.Tensor to torch.Tensor
        """Calculate Navier-Stokes residuals."""
        pass