
"""
primary author: Jon Errasti Odriozola (tensorflow)
secondary author: Vishnu Sankar (converted tf to pytorch)
github-id: https://github.com/errasti13
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union

class BoundaryCondition(ABC):
    """Base class for all boundary conditions."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(self, coords: List[torch.Tensor], values: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Apply the boundary condition.
        
        Args:
            coords: List of coordinate tensors [x, y] or [x, y, z]
            values: Dictionary of boundary values and conditions
            tape: GradientTape for automatic differentiation
            
        Returns:
            Dictionary of variable names to their boundary conditions
        """
        pass



class GradientBC(BoundaryCondition):
    """Boundary condition for gradients of variables."""
    def apply(self, coords: List[torch.Tensor], values: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        result = {}
        n_dims = len(coords)  # Get number of dimensions
        
        
        if coords[0] is not None:
            default_dtype = coords[0].dtype # it is good practice to specify the device for pytorch tensors
            default_device = coords[0].device
                

        for var_name, grad_info in values.items():
            if grad_info is None:
                result[var_name] = None
                continue

            if isinstance(grad_info, dict):
                direction = grad_info.get('direction', 'normal')
                value = grad_info.get('value', 0.0)
                
                if direction == 'normal':
                    # Get normal direction components for actual dimensions
                    normal_components = [
                        grad_info.get(f'n{dim}', 0.0)
                        for dim in ['x', 'y', 'z'][:n_dims]
                    ]
                    direction = tuple(normal_components)
                
                
                if isinstance(value, torch.Tensor):
                    torch_value = value.to(dtype=default_dtype, device=default_device)
                
                else:
                    torch_value = torch.tensor(value, dtype=default_dtype, device=default_device)
                
                result[var_name] = {'gradient': torch_value, 'direction': direction}
            
        return result

class DirichletBC(BoundaryCondition): # Ensure PyTorch BoundaryCondition base
    """Dirichlet boundary condition."""
    def apply(self, coords: List[torch.Tensor], values: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        result = {}
        # Determine default_dtype and default_device safely
        default_dtype = torch.float32
        default_device = torch.device('cpu')
        if coords and coords[0] is not None: 
            default_dtype = coords[0].dtype
            default_device = coords[0].device

        for var_name, var_info in values.items():
            if var_info is None:
                result[var_name] = None # No condition for this variable
                continue

            # Skip 'w' component in 2D, specific to this project's conventions
            if var_name == 'w' and coords and len(coords) == 2:
                continue

            actual_value_data = None
            is_value_spec = False

            if isinstance(var_info, dict):
                if 'value' in var_info:
                    actual_value_data = var_info['value']
                    is_value_spec = True
                # If var_info is a dict but has no 'value', this DirichletBC will not process it.
                # It might be a spec for another BC type (e.g. gradient) to be handled by compute_boundary_loss.
                # Or it's a malformed condition from the user.
                # For DirichletBC, we ONLY care if 'value' is present.
            else: # var_info is not a dict, so it's assumed to be the direct value
                actual_value_data = var_info
                is_value_spec = True

            if is_value_spec:
                if actual_value_data is None: # if var_info={'value': None} or var_info=None (already handled)
                    result[var_name] = None
                else:
                    try:
                        if isinstance(actual_value_data, torch.Tensor):
                            torch_value = actual_value_data.to(dtype=default_dtype, device=default_device)
                        else:
                            torch_value = torch.tensor(actual_value_data, dtype=default_dtype, device=default_device)
                        result[var_name] = {'type': 'dirichlet', 'value': torch_value}
                    except Exception as e_conv:
                        print(f"Warning (DirichletBC {self.name}): Could not convert value for '{var_name}' to tensor: {actual_value_data}. Error: {e_conv}")
                        result[var_name] = None # Failed to process
            # If var_info was a dict without 'value', result[var_name] is not set here by DirichletBC.
            # This means compute_boundary_loss won't find 'value' in bc_results[var_name] for this var_name.
            # If it also has a 'gradient' key, compute_boundary_loss would pick that up.

        return result


class WallBC(DirichletBC):
    """No-slip wall boundary condition."""
    def apply(self, coords: List[torch.Tensor], values: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        base_values = {'u': {'value': 0.0}, 'v': {'value': 0.0}}

        if len(coords) > 2:  # 3D case
            base_values['w'] = {'value': 0.0}
            
        if 'p' in values:
            base_values['p'] = values['p']
            
        # Override with any provided values
        base_values.update(values)
        return super().apply(coords, base_values)


class InletBC(DirichletBC):
    """Inlet boundary condition."""
    def apply(self, coords: List[torch.Tensor], values: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        base_values = {
            'u': {'value': 1.0},
            'v': {'value': 0.0}
        }
        if len(coords) > 2:  # 3D case
            base_values['w'] = {'value': 0.0}
            
        # Override with provided values
        base_values.update(values)
        return super().apply(coords, base_values)


class OutletBC(GradientBC):
    """Outlet boundary condition."""
    def apply(self, coords: List[torch.Tensor], values: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        base_values = {
            'u': {'gradient': 0.0, 'direction': 'x'},
            'v': {'gradient': 0.0, 'direction': 'x'}
        }
        if len(coords) > 2:  # 3D case
            base_values['w'] = {'gradient': 0.0, 'direction': 'x'}
            
        if 'p' in values:
            base_values['p'] = {'value': 0.0}  # Default pressure value at outlet
            
        # Override with provided values
        base_values.update(values)
        return super().apply(coords, base_values)


class MovingWallBC(DirichletBC):
    """Moving wall boundary condition."""
    def apply(self, coords: List[torch.Tensor], values: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        base_values = {
            'u': {'value': 1.0},
            'v': {'value': 0.0}
        }
        if len(coords) > 2:  # 3D case
            base_values['w'] = {'value': 0.0}
            
        if 'p' in values:
            base_values['p'] = values['p']
            
        # Override with provided values
        base_values.update(values)
        return super().apply(coords, base_values)
    
## have to implement periodic boundary conditions