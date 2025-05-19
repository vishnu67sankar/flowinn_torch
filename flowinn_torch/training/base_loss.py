
"""
Author of the tensorflow repo is Jon Errasti Odriozola (tensorflow) | github-id: https://github.com/errasti13
Vishnu Sankar has converted tf to pytorch and re-built the same package but in pytroch
"""

import torch 
import numpy as np 
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional 

class NavierStokesBaseLoss(ABC):
    def __init__(self, mesh, model, Re: float = 1000.0, weights: List[float] = [0.7, 0.3]) -> None: 
        self._mesh = mesh  
        self._model = model 
        self._Re = Re
        
        default_device = model.device if hasattr(model, 'device') else torch.device('cpu')
        self.physicsWeight = torch.tensor(weights[0], dtype=torch.float32, device=default_device)
        self.boundaryWeight = torch.tensor(weights[1], dtype=torch.float32, device=default_device)
        
        self.physics_loss_history: List[float] = [] 
        self.boundary_loss_history: List[float] = [] 
        self.lookback_window = 50
        self.alpha = 0.1 
        self.min_weight = 0.1 
        self.max_weight = 0.9 

    @property
    def mesh(self):
        return self._mesh
    
    @property
    def model(self):
        return self._model
    
    @property
    def Re(self):
        return self._Re
    
    @abstractmethod
    def compute_physics_loss(self, predictions: torch.Tensor, coords: List[torch.Tensor]) -> torch.Tensor: 
        pass

    @abstractmethod
    def compute_boundary_loss(self, bc_results: dict, vel_pred: torch.Tensor, p_pred: torch.Tensor, coords: List[torch.Tensor]) -> torch.Tensor: 
        pass

    @abstractmethod
    def loss_function(self, batch_data=None) -> torch.Tensor:
        """Abstract method for loss function implementation"""
        pass

    def update_weights(self):
        """Update weights using ReLoBraLo algorithm"""
        if len(self.physics_loss_history) < self.lookback_window:
            return

        device = self.physicsWeight.device
        dtype = self.physicsWeight.dtype

        lookback_max_val = min(self.lookback_window, len(self.physics_loss_history))
        if lookback_max_val < 1: 
            lookback_val = 1
        else:
            lookback_tensor = torch.randint(low=1, high=lookback_max_val + 1, size=(), device=device)
            lookback_val = lookback_tensor.item() 
        # physics_change = (self.physics_loss_history[-1]/(tf.reduce_mean(self.physics_loss_history[-lookback:]) + 1e-10))
        physics_loss_lookback_slice = self.physics_loss_history[-lookback_val:]
        if not physics_loss_lookback_slice: physics_loss_lookback_slice = [0.0]
        mean_physics_loss_in_window = torch.mean(torch.tensor(physics_loss_lookback_slice, dtype=dtype, device=device))
        physics_change = self.physics_loss_history[-1]/(mean_physics_loss_in_window + 1e-10)
        
        # boundary_change = (self.boundary_loss_history[-1]/(tf.reduce_mean(self.boundary_loss_history[-lookback:]) + 1e-10))
        boundary_loss_lookback_slice = self.boundary_loss_history[-lookback_val:]
        if not boundary_loss_lookback_slice: boundary_loss_lookback_slice = [0.0] # just avoiding empty tensor here
        mean_boundary_loss_in_window = torch.mean(torch.tensor(boundary_loss_lookback_slice, dtype=dtype, device=device))
        boundary_change = self.boundary_loss_history[-1]/(mean_boundary_loss_in_window + 1e-10)
        
        weight_update = self.alpha*(physics_change - boundary_change)
        
        # self.physicsWeight = tf.clip_by_value(self.physicsWeight - weight_update, self.min_weight, self.max_weight)
        self.physicsWeight = torch.clamp(self.physicsWeight - weight_update, min=self.min_weight, max=self.max_weight)
        self.boundaryWeight = 1.0 - self.physicsWeight

    def convert_and_reshape(self, tensor: torch.Tensor, dtype: torch.dtype = torch.float32, shape: Tuple[int, ...] = (-1, 1)) -> Optional[torch.Tensor]:
        if tensor is not None:
            if not isinstance(tensor, torch.Tensor):
                processed_tensor = torch.as_tensor(tensor, dtype=dtype)

            else:
                processed_tensor = tensor.to(dtype=dtype)
            return processed_tensor.reshape(shape)
        return None