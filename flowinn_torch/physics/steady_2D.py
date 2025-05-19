
"""
Author of the tensorflow repo is Jon Errasti Odriozola (tensorflow) | github-id: https://github.com/errasti13
Vishnu Sankar has converted tf to pytorch and re-built the same package but in pytroch
"""

import torch
from typing import Tuple
from .navier_stokes import NavierStokes


class SteadyNavierStokes2D(NavierStokes):
    """2D Steady Navier-Stokes equations solver."""

    def __init__(self, Re: float = 1000.0):
        """
        Initialize the solver.
        
        Args:
            Re (float): Reynolds number
        """
        self.Re = Re

    
    def get_residuals(self, velocities: torch.Tensor, pressure: torch.Tensor, coords: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate 2D Steady Navier-Stokes residuals.
        
        Args:
            velocities: Tensor of velocity components [u, v]
            pressure: Pressure tensor
            coords: List of coordinate tensors [x, y]
            tape: Gradient tape for automatic differentiation
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (continuity, momentum_x, momentum_y) residuals
        """
        x, y = coords
        u = velocities[:, 0]
        v = velocities[:, 1]
        p = pressure.reshape(-1)
        
        [u_x, u_y, v_x, v_y, p_x, p_y] = self._compute_first_derivatives([u, v, p], [x, y])
        [u_xx, u_xy, u_yx, u_yy, v_xx, v_xy, v_yx, v_yy] = self._compute_second_derivatives([u_x, u_y, v_x, v_y], [x, y])

        # Continuity equation: ∂u/∂x + ∂v/∂y = 0
        continuity = u_x + v_y
        
        # Momentum equation x: u∂u/∂x + v∂u/∂y = -∂p/∂x + 1/Re(∂²u/∂x² + ∂²u/∂y²)
        momentum_x = (u*u_x + v*u_y + p_x - 1/self.Re*(u_xx + u_yy))
        
        # Momentum equation y: u∂v/∂x + v∂v/∂y = -∂p/∂y + 1/Re(∂²v/∂x² + ∂²v/∂y²)
        momentum_y = (u*v_x + v*v_y + p_y- 1/self.Re*(v_xx + v_yy))

        continuity = continuity.reshape(-1)
        momentum_x = momentum_x.reshape(-1)
        momentum_y = momentum_y.reshape(-1)

        return continuity, momentum_x, momentum_y

    def _compute_first_derivatives(self, variables, coords):
        return super()._compute_first_derivatives(variables, coords)

    def _compute_second_derivatives(self, first_derivs, coords):
        return super()._compute_second_derivatives(first_derivs, coords)
