
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

class UnsteadyNavierStokesLoss(NavierStokesBaseLoss):
    pass