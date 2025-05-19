
"""
Author of the tensorflow repo is Jon Errasti Odriozola (tensorflow) | github-id: https://github.com/errasti13
Vishnu Sankar has converted tf to pytorch and re-built the same package but in pytroch
"""

import torch 
import numpy as np 
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple 
from flowinn_torch.physics.steady_2D import SteadyNavierStokes2D
from flowinn_torch.physics.steady_3D import SteadyNavierStokes3D
from flowinn_torch.training.base_loss import NavierStokesBaseLoss 

class SteadyNavierStokesLoss(NavierStokesBaseLoss):
    def __init__(self, mesh, model, Re: float = 1000.0, physics_model='NS2D', weights: List[float]=[0.7, 0.3]) -> None:
        super().__init__(mesh, model, Re, weights) 
        if physics_model == 'NS2D':
            self._physics_loss = SteadyNavierStokes2D(Re)
        elif physics_model == 'NS3D':
            self._physics_loss = SteadyNavierStokes3D(Re)
        else:
            raise ValueError(f"Unknown physics model: {physics_model}")

    @property
    def physics_loss(self):
        return self._physics_loss

    def _to_tensor_and_reshape(self, data: Any, dtype: torch.dtype = torch.float32, shape: Tuple[int,...] = (-1, 1), device: torch.device = None) -> torch.Tensor:
        """ Helper to convert to tensor and reshape, ensuring correct device. """
        if device is None:
            device = self._model.device if hasattr(self._model, 'device') else torch.device('cpu')
        
        if not isinstance(data, torch.Tensor):
            tensor = torch.as_tensor(data, dtype=dtype, device=device)
        else:
            tensor = data.to(dtype=dtype, device=device)
        return tensor.reshape(shape)

    def loss_function(self, batch_data: torch.Tensor = None) -> torch.Tensor:
        """Compute combined physics and boundary condition losses"""
       
        device = self._model.device if hasattr(self._model, 'device') else torch.device('cpu')
        prepared_coords = [] 
       
        if batch_data is None:
           
            num_coord_dims = 3 if isinstance(self._physics_loss, SteadyNavierStokes3D) else 2
            coord_names = ['x', 'y', 'z'][:num_coord_dims]
          
            for coord_name in coord_names:
               
                coord_data = getattr(self.mesh, coord_name) 
                coord_tensor = self._to_tensor_and_reshape(coord_data, shape=(-1,1), device=device)
                coord_tensor.requires_grad_(True)
                prepared_coords.append(coord_tensor)

        else:
            processed_batch_data = batch_data.to(device).detach().requires_grad_(True)
            prepared_coords = list(torch.split(processed_batch_data, split_size_or_sections=1, dim=-1))

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        input_tensor = torch.cat(prepared_coords, dim=1)
        predictions = self._model.model(input_tensor) 
        current_physics_loss = self.compute_physics_loss(predictions, prepared_coords)
        total_loss += self.physicsWeight*current_physics_loss

        current_boundary_loss_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
        if hasattr(self.mesh, 'boundaries') and self.mesh.boundaries:
            for boundary_name, boundary_data in self.mesh.boundaries.items():
                try:
                    num_bc_coord_dims = 3 if isinstance(self._physics_loss, SteadyNavierStokes3D) else 2
                    bc_coord_names = ['x', 'y', 'z'][:num_bc_coord_dims]
                    
                    current_bc_coords_list = []
                    present_bc_coords = [name for name in bc_coord_names if name in boundary_data]
                    if not present_bc_coords: continue

                    for coord_name in present_bc_coords:
                        coord_tensor = self._to_tensor_and_reshape(boundary_data[coord_name], shape=(-1,1), device=device)
                        coord_tensor.requires_grad_(True) 
                        current_bc_coords_list.append(coord_tensor)
                    
                    if not current_bc_coords_list: continue

                    bc_type = boundary_data['bc_type']
                    conditions = boundary_data['conditions']

                    bc_input_tensor = torch.cat(current_bc_coords_list, dim=1)
                    bc_pred_output = self._model.model(bc_input_tensor)
                    
                    vel_pred = bc_pred_output[:, :-1]
                    p_pred = bc_pred_output[:, -1]
                    bc_results = bc_type.apply(current_bc_coords_list, conditions)
                    current_boundary_loss_sum += self.compute_boundary_loss(bc_results, vel_pred, p_pred, current_bc_coords_list)

                except Exception as e:
                    print(f"Warning: Error processing boundary {boundary_name}: {str(e)}")
                    continue
        
        total_loss += self.boundaryWeight*current_boundary_loss_sum


        current_interior_loss_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
        if hasattr(self.mesh, 'interiorBoundaries') and self.mesh.interiorBoundaries:
            for boundary_name, boundary_data in self.mesh.interiorBoundaries.items():
                try:
                    num_bc_coord_dims = 3 if isinstance(self._physics_loss, SteadyNavierStokes3D) else 2
                    bc_coord_names = ['x', 'y', 'z'][:num_bc_coord_dims]
                    
                    current_int_coords_list = []
                    present_int_coords = [name for name in bc_coord_names if name in boundary_data]
                    if not present_int_coords: continue

                    for coord_name in present_int_coords:
                        coord_tensor = self._to_tensor_and_reshape(boundary_data[coord_name], shape=(-1,1), device=device)
                        coord_tensor.requires_grad_(True)
                        current_int_coords_list.append(coord_tensor)
                    
                    if not current_int_coords_list: continue

                    bc_type = boundary_data['bc_type']
                    conditions = boundary_data['conditions']

                    int_input_tensor = torch.cat(current_int_coords_list, dim=1)
                    int_pred_output = self._model.model(int_input_tensor)
                    vel_pred = int_pred_output[:, :-1]
                    p_pred = int_pred_output[:, -1].unsqueeze(-1)

                    bc_results = bc_type.apply(current_int_coords_list, conditions)
                    current_interior_loss_sum += self.compute_boundary_loss(bc_results, vel_pred, p_pred, current_int_coords_list)
                except Exception as e:
                    print(f"Warning: Error processing interior boundary {boundary_name}: {str(e)}")
                    continue
            total_loss += self.boundaryWeight*current_interior_loss_sum

        if hasattr(self.mesh, 'periodicBoundaries') and self.mesh.periodicBoundaries:
            periodic_loss_val = self.compute_periodic_loss() 
            total_loss += self.boundaryWeight*periodic_loss_val 

        self.physics_loss_history.append(current_physics_loss.item())
        combined_bc_actual_loss = torch.tensor(0.0, device=device) 
        if 'current_boundary_loss_sum' in locals() and isinstance(current_boundary_loss_sum, torch.Tensor):
            combined_bc_actual_loss += current_boundary_loss_sum
        if 'current_interior_loss_sum' in locals() and isinstance(current_interior_loss_sum, torch.Tensor):
            combined_bc_actual_loss += current_interior_loss_sum
        if hasattr(self.mesh, 'periodicBoundaries') and self.mesh.periodicBoundaries and 'periodic_loss_val' in locals() and isinstance(periodic_loss_val, torch.Tensor):
            combined_bc_actual_loss += periodic_loss_val

        self.boundary_loss_history.append(combined_bc_actual_loss.item())
        
        self.update_weights()

        if len(self.physics_loss_history) > self.lookback_window*2:
            self.physics_loss_history = self.physics_loss_history[-self.lookback_window:]
            self.boundary_loss_history = self.boundary_loss_history[-self.lookback_window:]
        return total_loss


    def compute_physics_loss(self, predictions: torch.Tensor, coords_for_physics: List[torch.Tensor]) -> torch.Tensor:
        """Compute physics-based loss terms for flow equations."""
        device = predictions.device 
        is_3d = isinstance(self._physics_loss, SteadyNavierStokes3D)
        n_vel_components = 3 if is_3d else 2
        
        predictions = predictions.to(dtype=torch.float32)
        velocities = predictions[:, :n_vel_components].reshape(-1, n_vel_components)
        pressure = predictions[:, n_vel_components].reshape(-1) 
        
        coords_to_use = coords_for_physics if is_3d else coords_for_physics[:2]
        
        residuals_tuple = self._physics_loss.get_residuals(velocities, pressure, coords_to_use)
        
        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        for residual_tensor in residuals_tuple:
            if residual_tensor is not None:
                residual_tensor = residual_tensor.reshape(-1)
                loss += torch.mean(torch.square(residual_tensor))
        return loss

    def compute_boundary_loss(self, bc_results: Dict[str, Any], vel_pred: torch.Tensor, 
                              p_pred: torch.Tensor, coords: List[torch.Tensor]) -> torch.Tensor:
        """Compute boundary condition losses."""
        device = vel_pred.device
        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        is_3d = isinstance(self._physics_loss, SteadyNavierStokes3D)
        n_vel_components = 3 if is_3d else 2
        
        p_pred_flat = p_pred.reshape(-1) 

        for var_name, bc_info in bc_results.items():
            if bc_info is None or not isinstance(bc_info, dict):
                continue
                
            if 'value' in bc_info:
                target_value = torch.as_tensor(bc_info['value'], dtype=torch.float32, device=device)
                if var_name == 'p':
                    loss += torch.mean(torch.square(p_pred_flat - target_value.reshape(-1)))
                else:
                    component_idx = {'u': 0, 'v': 1, 'w': 2}.get(var_name)
                    if component_idx is not None and component_idx < n_vel_components:
                        loss += torch.mean(torch.square(vel_pred[:, component_idx] - target_value.reshape(-1)))
                            
            if 'gradient' in bc_info:
                loss += self.compute_gradient_loss(bc_info, vel_pred, p_pred, coords, var_name, n_vel_components)
                
        return loss

    def compute_gradient_loss(self, bc_info: Dict[str, Any], vel_pred: torch.Tensor, p_pred: torch.Tensor, 
                              coords: List[torch.Tensor], var_name: str, n_vel_components: int) -> torch.Tensor:
        """Compute gradient-based boundary condition losses using PyTorch."""
        device = vel_pred.device
        target_gradient = torch.as_tensor(bc_info['gradient'], dtype=torch.float32, device=device)
        direction_spec = bc_info['direction'] 
        loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        if var_name == 'p':
            var_tensor_to_diff = p_pred 
        else:
            component_idx = {'u': 0, 'v': 1, 'w': 2}.get(var_name)
            if component_idx is None or component_idx >= n_vel_components:
                return loss 
            var_tensor_to_diff = vel_pred[:, component_idx] 
        
        if isinstance(direction_spec, tuple): 
            normal_vector_components = [torch.as_tensor(comp, dtype=torch.float32, device=device) for comp in direction_spec]
            
            computed_normal_grad = torch.zeros_like(var_tensor_to_diff, device=device)

            for i, coord_tensor in enumerate(coords):
                if i < len(normal_vector_components) and normal_vector_components[i] != 0:
                    print(f"var_tensor_to_diff: shape={var_tensor_to_diff.shape}, requires_grad={var_tensor_to_diff.requires_grad}, grad_fn={var_tensor_to_diff.grad_fn is not None}")
                    print(f"coord_tensor: shape={coord_tensor.shape}, requires_grad={coord_tensor.requires_grad}, is_leaf={coord_tensor.is_leaf}")
                    grad_wrt_coord = torch.autograd.grad(outputs=var_tensor_to_diff, inputs=coord_tensor,
                        grad_outputs=torch.ones_like(var_tensor_to_diff),
                        retain_graph=True, create_graph=False, allow_unused=True)[0]
                    
                    if grad_wrt_coord is not None:
                        computed_normal_grad += normal_vector_components[i]*grad_wrt_coord.reshape(computed_normal_grad.shape)
            
            loss += torch.mean(torch.square(computed_normal_grad - target_gradient.reshape(computed_normal_grad.shape)))
            
        else:
            coord_map = {'x': 0, 'y': 1, 'z': 2}
            coord_idx = coord_map.get(direction_spec)
            
            if coord_idx is not None and coord_idx < len(coords):
                coord_tensor_to_diff_against = coords[coord_idx]
                
                grad_wrt_coord = torch.autograd.grad(outputs=var_tensor_to_diff, inputs=coord_tensor_to_diff_against,
                    grad_outputs=torch.ones_like(var_tensor_to_diff),
                    retain_graph=True, create_graph=False, allow_unused=True)[0]
                
                if grad_wrt_coord is not None:
                    loss += torch.mean(torch.square(grad_wrt_coord - target_gradient)) 
        return loss

    def compute_periodic_loss(self) -> torch.Tensor: 
        """Compute loss for periodic boundary conditions using PyTorch."""
        device = self._model.device if hasattr(self._model, 'device') else torch.device('cpu')
        periodic_loss_sum = torch.tensor(0.0, dtype=torch.float32, device=device)

        if not hasattr(self.mesh, 'periodicBoundaries') or not self.mesh.periodicBoundaries:
            return periodic_loss_sum

        for boundary_name, boundary_data in self.mesh.periodicBoundaries.items():
            try:
                coupled_boundary_name = boundary_data['coupled_boundary']
                all_boundaries_data = {**getattr(self.mesh, 'boundaries', {}), **getattr(self.mesh, 'interiorBoundaries', {})}
                coupled_data = all_boundaries_data.get(coupled_boundary_name)

                if coupled_data is None:
                    print(f"Warning: Coupled boundary {coupled_boundary_name} not found for periodic boundary {boundary_name}")
                    continue

                # determine coordinate names (e.g. ['x', 'y'] or ['x', 'y', 'z'])
                coord_names = ['x', 'y', 'z'] if isinstance(self._physics_loss, SteadyNavierStokes3D) else ['x', 'y']
                present_base_coords = [name for name in coord_names if name in boundary_data]
                present_coupled_coords = [name for name in coord_names if name in coupled_data]

                if not present_base_coords or not present_coupled_coords or len(present_base_coords) != len(present_coupled_coords):
                    print(f"Warning: Mismatched coordinates for periodic pair {boundary_name} and {coupled_boundary_name}")
                    continue

                base_coords_list = []
                for name in present_base_coords:
                    c = self._to_tensor_and_reshape(boundary_data[name], shape=(-1,1), device=device)
                    c.requires_grad_(True)
                    base_coords_list.append(c)

                coupled_coords_list = []
                for name in present_coupled_coords:
                    c = self._to_tensor_and_reshape(coupled_data[name], shape=(-1,1), device=device)
                    c.requires_grad_(True)
                    coupled_coords_list.append(c)
                
                base_input_tensor = torch.cat(base_coords_list, dim=1)
                coupled_input_tensor = torch.cat(coupled_coords_list, dim=1)
                
                base_pred_output = self._model.model(base_input_tensor)
                coupled_pred_output = self._model.model(coupled_input_tensor)

                # match loss for u, v, p
                value_loss = torch.mean(torch.square(base_pred_output - coupled_pred_output))
                periodic_loss_sum += value_loss

                # match gradients
                num_outputs = base_pred_output.shape[1]
                for i in range(num_outputs): 
                    for j in range(len(base_coords_list)): 
                        base_grad_comp = torch.autograd.grad(outputs=base_pred_output[:, i], inputs=base_coords_list[j],
                            grad_outputs=torch.ones_like(base_pred_output[:, i]),
                            retain_graph=True, create_graph=False, allow_unused=True)[0]
                        
                        coupled_grad_comp = torch.autograd.grad(
                            outputs=coupled_pred_output[:, i], inputs=coupled_coords_list[j],
                            grad_outputs=torch.ones_like(coupled_pred_output[:, i]),
                            retain_graph=True, create_graph=False, allow_unused=True)[0]
                        
                        if base_grad_comp is not None and coupled_grad_comp is not None:
                            gradient_match_loss = torch.mean(torch.square(base_grad_comp - coupled_grad_comp))
                            periodic_loss_sum += gradient_match_loss
            
            except Exception as e:
                print(f"Warning: Error processing periodic boundary {boundary_name}: {str(e)}")
                continue
        return periodic_loss_sum